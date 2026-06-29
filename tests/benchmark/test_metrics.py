from benchmark.metrics.answer import exact_match, normalized_match
from benchmark.metrics.citation import citation_scores
from benchmark.metrics.format import format_scores
from benchmark.parser.output_parser import parse_model_output
from benchmark.generator.retrieval import generate_retrieval_task
from benchmark.generator.temporal import generate_temporal_task
from benchmark.generator.multihop import generate_multihop_task


def test_normalized_match_ignores_case_and_articles():
    assert normalized_match("The Team Delta", "team delta") == 1.0


def test_exact_match_requires_same_trimmed_text():
    assert exact_match(" Team Delta ", "Team Delta") == 1.0
    assert exact_match("team delta", "Team Delta") == 0.0


def test_citation_scores_penalize_distractor_and_stale():
    scores = citation_scores(
        pred_evidence_ids=["E01", "E02", "E03"],
        gold_evidence_ids=["E01", "E04"],
        distractor_evidence_ids=["E02"],
        stale_evidence_ids=["E03"],
        valid_evidence_ids={"E01", "E02", "E03", "E04"},
    )
    assert scores.precision == 1 / 3
    assert scores.recall == 1 / 2
    assert scores.f1 == 0.4
    assert scores.all_gold_evidence_recall == 0.0
    assert scores.distractor_citation_rate == 1 / 3
    assert scores.stale_citation_rate == 1 / 3


def test_citation_scores_track_invalid_and_overcitation():
    scores = citation_scores(
        pred_evidence_ids=["E01", "E99", "E02"],
        gold_evidence_ids=["E01"],
        distractor_evidence_ids=["E02"],
        stale_evidence_ids=[],
        valid_evidence_ids={"E01", "E02"},
    )
    assert scores.invalid_citation_rate == 1 / 3
    assert scores.overcitation_rate == 2 / 3


def test_format_scores_require_valid_format_and_step_count():
    task = generate_temporal_task("vlr_pilot_000001", seed=1, target_context_tokens=8000)
    parsed = parse_model_output(
        "Evidence: E01\nSteps:\n1. E01 states it.\n2. Therefore answer.\nAnswer: A17",
        valid_evidence_ids={"E01"},
    )
    scores = format_scores(parsed, task)
    assert scores.format_valid == 1.0
    assert scores.step_count_valid == 1.0


def test_single_step_valid_for_retrieval_but_not_temporal():
    single_step = "Evidence: E01\nSteps:\n1. E01 states the answer.\nAnswer: A17"
    parsed = parse_model_output(single_step, valid_evidence_ids={"E01"})

    retrieval = generate_retrieval_task("vlr_pilot_000001", seed=1, target_context_tokens=8000)
    temporal = generate_temporal_task("vlr_pilot_000002", seed=2, target_context_tokens=8000)

    # One grounded step is a complete rationale for single-hop retrieval ...
    assert format_scores(parsed, retrieval).step_count_valid == 1.0
    assert format_scores(parsed, retrieval).format_valid == 1.0
    # ... but temporal_update needs at least two steps.
    assert format_scores(parsed, temporal).step_count_valid == 0.0
    assert format_scores(parsed, temporal).format_valid == 0.0


def test_multihop_accepts_two_steps_for_three_hops():
    # A model may legitimately express 3 hops in 2 grounded steps; step count
    # must not punish this. Whether all hops are used is a citation concern.
    task = generate_multihop_task("vlr_pilot_000001", seed=1, hop_count=3, target_context_tokens=8000)
    assert task.metadata.hop_count == 3
    two_step = (
        "Evidence: E01, E02, E03\n"
        "Steps:\n"
        "1. E01 and E02 establish the protocol and condition.\n"
        "2. E03 maps that condition to the destination.\n"
        "Answer: Vault K"
    )
    parsed = parse_model_output(two_step, valid_evidence_ids={"E01", "E02", "E03"})
    scores = format_scores(parsed, task)
    assert scores.step_count_valid == 1.0
    assert scores.format_valid == 1.0
