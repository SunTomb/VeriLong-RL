from dataclasses import dataclass

from benchmark.schemas.prediction import ParsedPrediction
from benchmark.schemas.task import VeriLongTask


@dataclass(frozen=True)
class FormatScores:
    format_valid: float
    step_count_valid: float


def format_scores(parsed: ParsedPrediction, task: VeriLongTask) -> FormatScores:
    """Score output format, using task-derived step bounds.

    The parser is task-agnostic and only flags structural problems
    (missing evidence/steps/answer, invalid evidence IDs). Step-count validity
    depends on the task family / structure, so it is evaluated here against
    :meth:`VeriLongTask.step_bounds`. ``format_valid`` requires both the parser
    to be clean and the step count to fall within bounds.
    """

    min_steps, max_steps = task.step_bounds()
    step_count_valid = 1.0 if min_steps <= len(parsed.pred_steps) <= max_steps else 0.0
    format_valid = 1.0 if (parsed.format_valid and step_count_valid == 1.0) else 0.0
    return FormatScores(format_valid=format_valid, step_count_valid=step_count_valid)
