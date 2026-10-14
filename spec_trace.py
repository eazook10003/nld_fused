"""Baseline generation loop with per-block records.

linear_spec_generate_traced is a copy of NemotronLabsDiffusionModel.linear_spec_generate from the HF repo
nvidia/Nemotron-Labs-Diffusion-8B (modeling_nemotron_labs_diffusion.py). Lines added for recording are marked
# [TRACE]; all other lines are unchanged, so the generated tokens and nfe match model.linear_spec_generate.
Here it gives the forward shapes of the baseline for FLOP counting (flops_record.py).

Per block it records draft confidences / entropies / tokens, the AR verify tokens and probabilities,
matched / accepted / full, and the wall time of the draft and verify forwards.
"""
import sys
import time
from typing import Optional

import torch
from transformers.cache_utils import DynamicCache


def _upstream_module(model):
    """The transformers_modules module the model class was loaded from."""
    return sys.modules[type(model).__module__]


def _sync_ms(t0):  # [TRACE]
    torch.cuda.synchronize()  # [TRACE]
    return (time.perf_counter() - t0) * 1000.0  # [TRACE]


@torch.no_grad()
def linear_spec_generate_traced(
    self,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    mask_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    max_thinking_tokens: Optional[int] = None,
    end_think_token_id: Optional[int] = None,
    threshold: float = 0.0,
    topk: int = 0,  # [TRACE] >0: also record the draft top-k token ids/probs per drafted cell (no effect on generation)
):
    """Linear speculative decoding: diffusion draft + AR verify.

    Each iteration: (1) draft the next block under bidirectional attention,
    (2) verify the drafted block under causal attention, accept the longest
    prefix where draft matches AR + one bonus token, advance the KV cache.

    LoRA-aware: when a PEFT adapter is attached to the model (e.g.
    ``linear_spec_lora``), it is toggled ON for the bidirectional draft
    phase and OFF for the causal prefill / verify phases — so the adapter
    only specializes the diffusion-mode forward and AR semantics are
    preserved. With no adapter loaded the calls are no-ops.

    Returns ``(output_ids, nfe)`` — ``output_ids`` includes the prompt.
    """
    _crop_dynamic_cache = _upstream_module(self)._crop_dynamic_cache  # [TRACE] upstream helper
    trace = []  # [TRACE] one dict per block

    if prompt_ids.shape[0] != 1:
        raise ValueError("Linear speculative decoding requires batch_size == 1")

    token_mask_id = mask_token_id if mask_token_id is not None else self.config.mask_token_id
    if eos_token_id is None:
        eos_token_id = getattr(self.config, "eos_token_id", None)

    device = prompt_ids.device

    def _set_diffusion_lm(val: bool):
        for layer in self.encoder.layers:
            if hasattr(layer.self_attn, "diffusion_lm"):
                layer.self_attn.diffusion_lm = val

    def _toggle_adapters(enable: bool):
        # No-op when no PEFT/LoRA modules are attached.
        for module in self.modules():
            if hasattr(module, "_disable_adapters"):
                module._disable_adapters = not enable

    # Prefill (causal, LoRA OFF).
    _set_diffusion_lm(False)
    _toggle_adapters(False)
    t0 = time.perf_counter()  # [TRACE]
    enc_out = self.encoder(
        input_ids=prompt_ids,
        past_key_values=DynamicCache(),
        use_cache=True,
        use_causal_mask=True,
    )
    past_key_values = enc_out.past_key_values
    last_logit = self.diffusion_head(enc_out.last_hidden_state[:, -1:, :]).squeeze(1)
    nfe = 1
    t_prefill_ms = _sync_ms(t0)  # [TRACE]

    if temperature > 0:
        next_token = torch.multinomial(torch.softmax(last_logit / temperature, dim=-1), num_samples=1)
    else:
        next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

    if eos_token_id is not None and next_token.item() == eos_token_id:
        return torch.cat([prompt_ids, next_token], dim=1), nfe, {"prefill_ms": t_prefill_ms, "blocks": trace}  # [TRACE]

    generated = [next_token]
    total_gen = 1

    while total_gen < max_new_tokens:
        cache_len = past_key_values.get_seq_length()

        block = torch.full((1, block_length), token_mask_id, dtype=torch.long, device=device)
        block[0, 0] = next_token.item()

        # Draft phase (bidirectional, LoRA ON) — iterate at threshold>0 so
        # that even low-confidence blocks make progress.
        _set_diffusion_lm(True)
        _toggle_adapters(True)
        conf_at_commit = torch.zeros(block_length, device=device)  # [TRACE]
        tk_ids_commit = torch.zeros(block_length, max(topk, 1), dtype=torch.long, device=device)  # [TRACE]
        tk_p_commit = torch.zeros(block_length, max(topk, 1), device=device)  # [TRACE]
        ent_at_commit = torch.zeros(block_length, device=device)  # [TRACE]
        n_draft_fwd = 0  # [TRACE]
        t0 = time.perf_counter()  # [TRACE]
        while True:
            is_mask = block == token_mask_id
            if not is_mask.any():
                break

            enc_out = self.encoder(input_ids=block, past_key_values=past_key_values, use_cache=False)
            nfe += 1
            n_draft_fwd += 1  # [TRACE]

            draft_logits = self.diffusion_head(enc_out.last_hidden_state)
            # LLaDA: logit[i] directly predicts position i — no shift needed.

            if temperature > 0:
                draft_probs = torch.softmax(draft_logits / temperature, dim=-1)
                draft_tokens = torch.multinomial(
                    draft_probs.view(-1, draft_probs.shape[-1]), num_samples=1
                ).view(1, block_length)
            else:
                draft_tokens = draft_logits.argmax(dim=-1)
                draft_probs = torch.softmax(draft_logits, dim=-1)

            _p = torch.gather(draft_probs, -1, draft_tokens.unsqueeze(-1)).squeeze(-1)[0].float()  # [TRACE] top-1 prob
            _lp = torch.log_softmax(draft_logits.float(), dim=-1)[0]  # [TRACE]
            _ent = -(_lp.exp() * _lp).sum(-1)  # [TRACE] entropy (nats)
            if topk > 0:  # [TRACE] top-k of the same draft distribution: ids from logits (consistent with argmax), probs from draft_probs
                _tk_ids = torch.topk(draft_logits[0].float(), topk, dim=-1).indices
                _tk_p = torch.gather(draft_probs[0].float(), -1, _tk_ids)

            if threshold > 0:
                draft_conf = torch.gather(draft_probs, -1, draft_tokens.unsqueeze(-1)).squeeze(-1)
                draft_conf = torch.where(is_mask, draft_conf, -torch.inf)
                unmask = draft_conf >= threshold
                # Force progress even when every masked position is below threshold.
                if not unmask.any():
                    best_idx = draft_conf.view(-1).argmax()
                    unmask = torch.zeros_like(is_mask, dtype=torch.bool)
                    unmask.view(-1)[best_idx] = True
                block[unmask] = draft_tokens[unmask]
                conf_at_commit[unmask[0]] = _p[unmask[0]]  # [TRACE]
                ent_at_commit[unmask[0]] = _ent[unmask[0]]  # [TRACE]
                if topk > 0: tk_ids_commit[unmask[0]] = _tk_ids[unmask[0]]; tk_p_commit[unmask[0]] = _tk_p[unmask[0]]  # [TRACE]
            else:
                block[is_mask] = draft_tokens[is_mask]
                conf_at_commit[is_mask[0]] = _p[is_mask[0]]  # [TRACE]
                ent_at_commit[is_mask[0]] = _ent[is_mask[0]]  # [TRACE]
                if topk > 0: tk_ids_commit[is_mask[0]] = _tk_ids[is_mask[0]]; tk_p_commit[is_mask[0]] = _tk_p[is_mask[0]]  # [TRACE]
                break
        t_draft_ms = _sync_ms(t0)  # [TRACE]
        draft_tok_list = block[0].tolist()  # [TRACE] full block incl. seed at [0]

        # Verify phase (causal, LoRA OFF).
        _set_diffusion_lm(False)
        _toggle_adapters(False)
        t0 = time.perf_counter()  # [TRACE]
        enc_out = self.encoder(
            input_ids=block,
            past_key_values=past_key_values,
            use_cache=True,
            use_causal_mask=True,
        )
        past_key_values = enc_out.past_key_values
        nfe += 1

        verify_logits = self.diffusion_head(enc_out.last_hidden_state)
        if temperature > 0:
            ar_tokens = torch.multinomial(
                torch.softmax(verify_logits / temperature, dim=-1).view(-1, verify_logits.shape[-1]),
                num_samples=1,
            ).view(1, block_length)
        else:
            ar_tokens = verify_logits.argmax(dim=-1)
        t_verify_ms = _sync_ms(t0)  # [TRACE]
        ar_p = torch.gather(torch.softmax(verify_logits.float(), dim=-1), -1, ar_tokens.unsqueeze(-1)).squeeze(-1)[0]  # [TRACE]

        # Accept consecutive matches; AR also gives one bonus token at the tail.
        accepted = 0
        for i in range(block_length - 1):
            if ar_tokens[0, i].item() == block[0, i + 1].item():
                accepted += 1
            else:
                break
        accepted += 1

        trace.append(dict(  # [TRACE]
            blk=len(trace), cache_len=int(cache_len), n_draft_fwd=n_draft_fwd,
            draft_p=[round(v, 5) for v in conf_at_commit[1:].tolist()],
            draft_ent=[round(v, 4) for v in ent_at_commit[1:].tolist()],
            draft_tok=draft_tok_list[1:], seed_tok=draft_tok_list[0],
            ar_tok=ar_tokens[0].tolist(), ar_p=[round(v, 5) for v in ar_p.tolist()],
            matched=accepted - 1, accepted=accepted, full=(accepted == block_length),
            t_draft_ms=round(t_draft_ms, 3), t_verify_ms=round(t_verify_ms, 3),
        ))
        if topk > 0:  # [TRACE]
            trace[-1]["topk_ids"] = tk_ids_commit[1:].tolist()
            trace[-1]["topk_p"] = [[float(f"{v:.4g}") for v in row] for row in tk_p_commit[1:].tolist()]

        accepted_toks = ar_tokens[:, :accepted]
        generated.append(accepted_toks)
        total_gen += accepted

        _crop_dynamic_cache(past_key_values, cache_len + accepted)
        next_token = ar_tokens[:, accepted - 1 : accepted]

        if eos_token_id is not None:
            eos_pos = (accepted_toks[0] == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                first_eos = eos_pos[0].item()
                generated[-1] = accepted_toks[:, : first_eos + 1]
                total_gen = total_gen - accepted + first_eos + 1
                trace[-1]["eos_in_block"] = int(first_eos)  # [TRACE]
                break

        # Thinking-budget enforcement: force end-think as next seed if budget exhausted.
        if end_think_token_id is not None and max_thinking_tokens is not None:
            if total_gen > max_thinking_tokens:
                all_gen = torch.cat(generated, dim=1)
                if not (all_gen == end_think_token_id).any():
                    next_token = torch.tensor([[end_think_token_id]], device=device)

        if total_gen >= max_new_tokens:
            break

    all_generated = torch.cat(generated, dim=1)
    output_ids = torch.cat([prompt_ids, all_generated], dim=1)
    return output_ids, nfe, {"prefill_ms": t_prefill_ms, "blocks": trace}  # [TRACE]
