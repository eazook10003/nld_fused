"""Accuracy.
GSM8K    : the GSM8K scorer of the authors' SGLang branch (copied below without changes).
MATH-500 : math_verify on the text after </think>.
An output that did not end with EOS (stopped at the length cap) counts as wrong."""
import re


# ---- from github.com/hutm/sglang, benchmark/gsm8k/eval_sglang.py @ 1c9f365a (Apache-2.0), unchanged
def search_boxed(string: str) -> str | None:

    if "\\boxed" not in string:
        return None
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    retval = string[idx : right_brace_idx + 1]
    left = "\\boxed{"
    if retval[: len(left)] == left and retval[-1] == "}":
        return retval[len(left) : -1]
    return None


def extract_gsm8k_answer(text: str) -> str | None:
    if text is None:
        return None
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    match = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    boxed = search_boxed(text)
    if boxed is not None:
        return boxed.replace(",", "").strip()
    numbers = re.findall(r"[+-]?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return None


def extract_gsm8k_gold(answer_text: str) -> str:
    match = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    return answer_text.strip()


def normalize_number(s: str) -> str:
    if s is None:
        return ""
    s = re.sub(r"\\(?:text|mathrm|textbf|mathbf)\{([^}]*)\}", r"\1", s)
    s = s.replace("\\$", "").replace("$", "")
    s = re.sub(r"\\[,;:\s]", " ", s)
    s = re.sub(r"\\%", "%", s)
    s = s.strip().rstrip("%")
    s = s.rstrip("\\").strip()
    s = re.sub(
        r"\s*(?:cubic|square)?\s*(?:minutes|grams|mph|hours|dollars|cents|feet|meters|inches|pounds|ounces|gallons|liters|days|weeks|months|years|people|students|books|miles|cm|kg|lb|oz|ft|in|m|s)\s*$",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = s.strip()
    try:
        val = float(s.replace(",", ""))
        if not (val == val) or val == float("inf") or val == float("-inf"):
            return s
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, TypeError, OverflowError):
        return s


def score_gsm8k(pred_text: str, gold_answer_text: str) -> dict:
    pred = extract_gsm8k_answer(pred_text)
    gold = extract_gsm8k_gold(gold_answer_text)
    gold_norm = normalize_number(gold)
    pred_norm = normalize_number(pred) if pred else ""
    return {
        "is_correct": gold_norm == pred_norm,
        "pred_answer": pred,
        "pred_normalized": pred_norm,
        "gold_answer": gold,
        "gold_normalized": gold_norm,
    }


# ---- end of copied code


def is_correct(dataset, text, gold, term):
    if term != "eos":
        return False
    if dataset == "gsm8k":
        return bool(score_gsm8k(text, gold)["is_correct"])
    from math_verify import parse, verify
    try:
        return bool(verify(parse("$" + gold + "$"), parse(text.split("</think>")[-1])))
    except Exception:
        return False
