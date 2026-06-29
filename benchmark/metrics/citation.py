from benchmark.schemas.metrics import CitationScores


def citation_scores(
    pred_evidence_ids: list[str],
    gold_evidence_ids: list[str],
    distractor_evidence_ids: list[str],
    stale_evidence_ids: list[str],
    valid_evidence_ids: set[str],
) -> CitationScores:
    pred_set = set(pred_evidence_ids)
    gold_set = set(gold_evidence_ids)
    distractor_set = set(distractor_evidence_ids)
    stale_set = set(stale_evidence_ids)

    pred_count = len(pred_evidence_ids)
    true_positive_count = len(pred_set & gold_set)
    precision = _safe_divide(true_positive_count, pred_count)
    recall = _safe_divide(true_positive_count, len(gold_set))
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    all_gold_evidence_recall = 1.0 if gold_set and gold_set.issubset(pred_set) else 0.0

    invalid_count = sum(1 for evidence_id in pred_evidence_ids if evidence_id not in valid_evidence_ids)
    distractor_count = sum(1 for evidence_id in pred_evidence_ids if evidence_id in distractor_set)
    stale_count = sum(1 for evidence_id in pred_evidence_ids if evidence_id in stale_set)
    overcitation_count = sum(1 for evidence_id in pred_evidence_ids if evidence_id not in gold_set)

    return CitationScores(
        precision=precision,
        recall=recall,
        f1=f1,
        all_gold_evidence_recall=all_gold_evidence_recall,
        distractor_citation_rate=_safe_divide(distractor_count, pred_count),
        stale_citation_rate=_safe_divide(stale_count, pred_count),
        invalid_citation_rate=_safe_divide(invalid_count, pred_count),
        overcitation_rate=_safe_divide(overcitation_count, pred_count),
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
