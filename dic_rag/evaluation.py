"""Question/answer parsing, prompt construction, answer extraction, grading and IPR aggregation."""

import json
import os
import re
from typing import Dict, List, Optional

from .ssc import ROLES

N_OPTIONS = 5
# Overall information preservation rate = weighted mean of the three safeguarded roles (manuscript Sec. 2.6)
IPR_WEIGHTS = {"exception": 0.40, "condition": 0.35, "numeric": 0.25}

# The prompt was written in Korean for the experiments and is kept verbatim.
PROMPT_HEADER = "당신은 치과 건강보험 청구 전문가입니다. 다음 객관식 문제의 정답을 선택하세요.\n\n"
PROMPT_FORMAT = (
    "\n답변 형식 (반드시 준수):\n"
    "정답: [1~5 중 하나의 숫자만]\n"
    "이유: [선택한 이유를 1-2문장으로 간단히]\n\n"
    "예시:\n"
    "정답: 3\n"
    "이유: 치면열구전색술은 예방적 조치에 해당하므로 Z29.8 코드가 적합합니다.\n"
)


# ---------------------------------------------------------------- input parsing
def parse_questions(path: str) -> List[Dict]:
    """Parse a question file. One question per line:

        <no>. <question text>  1: <option>  2: <option> ... 5: <option>

    The question text and the option list are separated by two or more spaces before "1:".
    Lines that do not contain exactly five options are skipped with a warning.
    """
    questions = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            m = re.match(r"^(\d+)([.:])\s+(.+)$", line)
            if not m:
                continue
            number, body = int(m.group(1)), m.group(3)
            parts = re.split(r"\s{2,}1:\s+", body, maxsplit=1)
            if len(parts) != 2:
                parts = re.split(r"\s+1:\s+", body, maxsplit=1)
            if len(parts) != 2:
                print(f"  warning: question {number}: options not found, skipped")
                continue
            text, options_str = parts[0].strip(), "1: " + parts[1]
            options = re.findall(r"(\d+):\s+(.+?)(?=\s+\d+:\s+|$)", options_str)
            if len(options) != N_OPTIONS:
                print(f"  warning: question {number}: {len(options)} options (expected {N_OPTIONS}), skipped")
                continue
            questions.append({"number": number, "question": text, "options": [(int(n), t.strip()) for n, t in options]})
    return questions


def parse_correct_answers(path: str) -> Dict[int, int]:
    """Answer file: one '<question no>. <answer 1-5>' pair per line."""
    answers = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.search(r"(\d+)[.:\s]+(\d+)", line)
            if m:
                answers[int(m.group(1))] = int(m.group(2))
    return answers


def build_prompt(q: Dict) -> str:
    options = "".join(f"{n}: {t}\n" for n, t in q["options"])
    return f"{PROMPT_HEADER}문제: {q['question']}\n\n선택지:\n{options}{PROMPT_FORMAT}"


# --------------------------------------------------------------- answer parsing
_KOREAN_NUMERALS = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5}
_ANSWER_PATTERNS = [
    r"정답\s*[:：]\s*\**\s*([1-5])",
    r"답\s*[:：]\s*\**\s*([1-5])",
    r"선택\s*[:：]\s*\**\s*([1-5])",
    r"answer\s*(?:is)?\s*[:：]?\s*\**\s*([1-5])",
    r"정답은\s*\**\s*([1-5])",
    r"([1-5])\s*번",
]


def extract_answer_number(text: Optional[str]) -> Optional[int]:
    """Extract the selected option (1-5) from a model reply; None if not found."""
    if not text:
        return None
    for pattern in _ANSWER_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    m = re.search(r"정답\s*[:：은]?\s*([일이삼사오])\s*번?", text)
    if m:
        return _KOREAN_NUMERALS[m.group(1)]
    m = re.search(r"\b([1-5])\b", text)  # last resort
    return int(m.group(1)) if m else None


def extract_reasoning(text: Optional[str]) -> str:
    if not text:
        return ""
    m = re.search(r"이유\s*[:：]\s*(.+)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


# ---------------------------------------------------------------------- grading
def grade_letter(accuracy: float) -> str:
    for threshold, letter in ((90, "A"), (80, "B"), (70, "C"), (60, "D")):
        if accuracy >= threshold:
            return letter
    return "F"


def aggregate_ipr(records: List[Dict]) -> Optional[Dict]:
    """Pooled IPR: for each role, total kept sentences / total original sentences over all questions."""
    totals = {role: [0, 0] for role in ROLES}
    seen = False
    for r in records:
        counts = r.get("role_counts")
        if not counts:
            continue
        seen = True
        for role in ROLES:
            totals[role][0] += counts[role][0]
            totals[role][1] += counts[role][1]
    if not seen:
        return None
    ipr = {role: (kept / orig if orig else None) for role, (orig, kept) in totals.items()}
    if all(ipr[r] is not None for r in IPR_WEIGHTS):
        ipr["overall"] = sum(IPR_WEIGHTS[r] * ipr[r] for r in IPR_WEIGHTS)
    else:
        ipr["overall"] = None
    ipr["sentence_counts"] = {role: {"original": o, "kept": k} for role, (o, k) in totals.items()}
    return ipr


# ------------------------------------------------------------------- the driver
def run_evaluation(system, questions: List[Dict], correct: Dict[int, int], output_dir: str, tag: str = "") -> Dict:
    """Query `system` for every question, grade the replies and write result files to `output_dir`."""
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    records = []
    for i, q in enumerate(questions, start=1):
        print(f"[{system.name}] question {q['number']} ({i}/{len(questions)})")
        out = system.query(build_prompt(q))
        predicted = extract_answer_number(out["answer"])
        truth = correct.get(q["number"])
        records.append(
            {
                "question_number": q["number"],
                "predicted": predicted,
                "correct_answer": truth,
                "is_correct": predicted is not None and predicted == truth,
                "reasoning": extract_reasoning(out["answer"]),
                "raw_answer": out["answer"],
                "ntrial": out["ntrial"],
                "n_chunks": out["n_chunks"],
                "tokens_used": out["tokens_used"],
                "tokens_prompt": out["tokens_prompt"],
                "tokens_completion": out["tokens_completion"],
                "role_counts": out["role_counts"],
            }
        )

    n = len(records)
    n_correct = sum(r["is_correct"] for r in records)
    accuracy = 100.0 * n_correct / n if n else 0.0
    summary = {
        "method": system.name,
        "n_questions": n,
        "n_correct": n_correct,
        "accuracy_percent": accuracy,
        "grade": grade_letter(accuracy),
        "mean_total_tokens": sum(r["tokens_used"] for r in records) / n if n else 0.0,
        "mean_prompt_tokens": sum(r["tokens_prompt"] for r in records) / n if n else 0.0,
        "mean_completion_tokens": sum(r["tokens_completion"] for r in records) / n if n else 0.0,
        "ipr": aggregate_ipr(records),
    }
    compressor = getattr(system, "compressor", None)
    if compressor is not None and hasattr(compressor, "ppl_cap_rate"):
        summary["ppl_cap_rate"] = compressor.ppl_cap_rate

    with open(os.path.join(output_dir, f"results_{system.name}{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, f"answers_{system.name}{suffix}.txt"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(f"{r['question_number']}. {r['predicted']}\n")

    print(f"\n[{system.name}] accuracy {accuracy:.2f}% ({n_correct}/{n}), grade {summary['grade']}")
    if summary["ipr"]:
        print(f"[{system.name}] overall IPR {summary['ipr']['overall']}")
    if "ppl_cap_rate" in summary:
        print(f"[{system.name}] PPL cap rate {summary['ppl_cap_rate']:.3f}")
    return summary
