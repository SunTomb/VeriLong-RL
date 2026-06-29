import re

from benchmark.schemas.prediction import ParsedPrediction


_EVIDENCE_LINE_RE = re.compile(r"^\s*Evidence\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_ANSWER_LINE_RE = re.compile(r"^\s*Answer\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE | re.DOTALL)
_STEPS_LINE_RE = re.compile(r"^\s*Steps\s*:\s*$", re.IGNORECASE | re.MULTILINE)
_EVIDENCE_ID_RE = re.compile(r"E\d+")
_NUMBERED_STEP_RE = re.compile(r"^\s*\d+\.\s*(.+?)\s*$")


def parse_model_output(text: str, valid_evidence_ids: set[str]) -> ParsedPrediction:
    error_flags: list[str] = []
    pred_evidence_ids = _parse_evidence_ids(text)
    pred_steps = _parse_steps(text)
    pred_answer = _parse_answer(text)

    if not pred_evidence_ids:
        error_flags.append("missing_evidence")
    for evidence_id in pred_evidence_ids:
        if evidence_id not in valid_evidence_ids:
            error_flags.append(f"invalid_evidence_id:{evidence_id}")

    if not pred_steps:
        error_flags.append("missing_steps")

    if pred_answer is None or pred_answer == "":
        error_flags.append("missing_answer")
        pred_answer = None

    return ParsedPrediction(
        pred_answer=pred_answer,
        pred_evidence_ids=pred_evidence_ids,
        pred_steps=pred_steps,
        format_valid=not error_flags,
        unparsed_text=text,
        error_flags=error_flags,
    )


def _parse_evidence_ids(text: str) -> list[str]:
    match = _EVIDENCE_LINE_RE.search(text)
    if match is None:
        return []
    return _EVIDENCE_ID_RE.findall(match.group(1))


def _parse_answer(text: str) -> str | None:
    match = _ANSWER_LINE_RE.search(text)
    if match is None:
        return None
    return match.group(1).strip()


def _parse_steps(text: str) -> list[str]:
    steps_match = _STEPS_LINE_RE.search(text)
    if steps_match is None:
        return []

    answer_match = _ANSWER_LINE_RE.search(text, steps_match.end())
    steps_block = text[steps_match.end() : answer_match.start() if answer_match else len(text)]
    steps: list[str] = []
    for line in steps_block.splitlines():
        step_match = _NUMBERED_STEP_RE.match(line)
        if step_match:
            steps.append(step_match.group(1))
    return steps
