"""Model loading, prompts and small helpers shared by the scripts."""
import os, subprocess
import torch
from transformers import AutoModel, AutoTokenizer
from spec_fused import install_gated_lora

# HF repo id or a local directory with the checkpoint (the LoRA adapter is in its linear_spec_lora/ subfolder)
MODEL = os.environ.get("NLD_MODEL", "nvidia/Nemotron-Labs-Diffusion-8B")
BL = 32

# instruction for GSM8K and MATH-500 (NeMo-Skills generic/math wording, used by the authors' evaluation)
V2 = "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.\n\n"
# GSM8K instruction of the upstream evaluate.py (github.com/NVlabs/Nemotron-Labs-Diffusion); only for the cost-table prompt
UPSTREAM_GSM8K = "Solve the following math problem. Put the final numerical answer inside \\boxed{} at the very end.\n\n"


def load_model(path=MODEL):
    """bf16 model + linear_spec_lora with the position-gated LoRA installed."""
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModel.from_pretrained(path, trust_remote_code=True).to("cuda").to(torch.bfloat16)
    from peft import PeftModel
    if os.path.isdir(path):
        model = PeftModel.from_pretrained(model, os.path.join(path, "linear_spec_lora")).eval().model
    else:
        model = PeftModel.from_pretrained(model, path, subfolder="linear_spec_lora").eval().model
    model.eval(); install_gated_lora(model)
    return tok, model


def load_prompts(dataset, limit=None):
    """[dict(id, user, gold)] in dataset order (first `limit` problems if given)."""
    from datasets import load_dataset
    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        rows = [dict(id=f"gsm8k-{i}", user=V2 + r["question"], gold=r["answer"].split("####")[-1].strip().replace(",", ""))
                for i, r in enumerate(ds)]
    elif dataset == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        rows = [dict(id=f"math500-{i}", user=V2 + r["problem"], gold=str(r["answer"])) for i, r in enumerate(ds)]
    else:
        raise ValueError(dataset)
    return rows[:limit] if limit else rows


def chat_ids(tok, user, thinking=False):
    think = dict(enable_thinking=True) if thinking else {}
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, **think)
    return tok(text, return_tensors="pt").input_ids.to("cuda")


def round_to_block(n, block=BL):  # same as upstream evaluate.py
    return max(block, (n // block) * block)


def parse_cost_table(s, arm):
    """'v=..,d=..,1=..,..,fa0=..,..' (ms, written by cost_table.py) -> the dyn argument of linear_spec_generate_fused.
    Arm dynfat gets the full-accept entries as a 4th element; dyn / dynfa ignore them."""
    c = dict(x.split("=") for x in s.split(","))
    dyn = (float(c.pop("v")), float(c.pop("d")), {int(k): float(x) for k, x in c.items() if not k.startswith("fa")})
    if arm == "dynfat":
        ffa = {int(k[2:]): float(x) for k, x in c.items() if k.startswith("fa")}
        assert ffa, "the cost table needs fa<M>=... entries for arm dynfat"
        dyn = dyn + (ffa,)
    return dyn


def smi():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip() + " | apps: " + \
               subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().replace("\n", "; ")
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi failed: {e}"
