from benchmark.generator.retrieval import generate_retrieval_task
from benchmark.generator.multihop import generate_multihop_task
from benchmark.generator.temporal import generate_temporal_task
from benchmark.parser.output_parser import parse_model_output
from benchmark.metrics.citation import citation_scores
from benchmark.metrics.format import format_scores
from experiments.sft.build_sft_data import build_sft_example
from experiments.sft.distill_steps import steps_are_gold_perfect


def _assistant_text(example):
    assert example["messages"][0]["role"] == "system"
    assert example["messages"][1]["role"] == "user"
    assert example["messages"][2]["role"] == "assistant"
    return example["messages"][2]["content"]


def _assert_self_consistent(task):
    example = build_sft_example(task)
    text = _assistant_text(example)
    parsed = parse_model_output(text, valid_evidence_ids=task.evidence_ids())
    fmt = format_scores(parsed, task)
    cit = citation_scores(
        pred_evidence_ids=parsed.pred_evidence_ids,
        gold_evidence_ids=task.gold_evidence_ids,
        distractor_evidence_ids=task.distractor_evidence_ids,
        stale_evidence_ids=task.stale_evidence_ids,
        valid_evidence_ids=task.evidence_ids(),
    )
    assert fmt.format_valid == 1.0
    assert cit.f1 == 1.0
    assert cit.distractor_citation_rate == 0.0
    assert cit.stale_citation_rate == 0.0
    assert parsed.pred_answer == task.gold_answer
    assert example["metadata"]["task_id"] == task.id
    assert example["metadata"]["task_family"] == task.task_family


def test_retrieval_sft_example_is_self_consistent():
    _assert_self_consistent(
        generate_retrieval_task(task_id="vlr_pilot_000001", seed=1, target_context_tokens=8000)
    )


def test_multihop_sft_example_is_self_consistent():
    _assert_self_consistent(
        generate_multihop_task(task_id="vlr_pilot_000002", seed=2, hop_count=3, target_context_tokens=8000)
    )


def test_temporal_sft_example_is_self_consistent():
    _assert_self_consistent(
        generate_temporal_task(task_id="vlr_pilot_000003", seed=3, update_count=2, target_context_tokens=8000)
    )


def test_hard_examples_are_self_consistent():
    _assert_self_consistent(
        generate_retrieval_task(
            task_id="vlr_pilot_000010", seed=10, target_context_tokens=16000,
            difficulty="hard", distractor_strength="adversarial", distractor_count=12,
            evidence_position="random",
        )
    )
    _assert_self_consistent(
        generate_temporal_task(
            task_id="vlr_pilot_000040", seed=40, update_count=4, stale_count=8,
            target_context_tokens=16000, difficulty="hard", evidence_position="mixed",
        )
    )


def test_distilled_steps_replaces_gold_steps_but_keeps_evidence_and_answer():
    task = generate_retrieval_task(task_id="vlr_pilot_000001", seed=1, target_context_tokens=8000)
    custom_steps = ["E01 directly states the access code.", "The code is therefore confirmed."]
    example = build_sft_example(task, distilled_steps={task.id: custom_steps})
    assistant = example["messages"][2]["content"]
    assert "1. E01 directly states the access code." in assistant
    assert "2. The code is therefore confirmed." in assistant
    assert f"Evidence: {', '.join(task.gold_evidence_ids)}" in assistant
    assert f"Answer: {task.gold_answer}" in assistant
    assert example["metadata"]["distilled_steps"] is True


def test_distilled_steps_falls_back_to_gold_when_not_in_map():
    task = generate_retrieval_task(task_id="vlr_pilot_000001", seed=1, target_context_tokens=8000)
    example_distilled = build_sft_example(task, distilled_steps={"other_id": ["irrelevant"]})
    example_gold = build_sft_example(task)
    assert example_distilled["messages"][2]["content"] == example_gold["messages"][2]["content"]
    assert example_distilled["metadata"]["distilled_steps"] is False


def test_steps_are_gold_perfect_accepts_exact_gold_output():
    task = generate_multihop_task(task_id="vlr_pilot_000002", seed=2, hop_count=3, target_context_tokens=8000)
    gold_ids = ", ".join(task.gold_evidence_ids)
    good_output = f"Evidence: {gold_ids}\nSteps:\n1. Use E01.\n2. Use E02.\n3. Use E03.\nAnswer: {task.gold_answer}"
    result = steps_are_gold_perfect(task, good_output)
    assert result is not None
    assert len(result) == 3


def test_steps_are_gold_perfect_rejects_distractor_citation():
    task = generate_retrieval_task(task_id="vlr_pilot_000001", seed=1, target_context_tokens=8000)
    distractor = task.distractor_evidence_ids[0]
    bad_output = f"Evidence: {task.gold_evidence_ids[0]}, {distractor}\nSteps:\n1. step.\nAnswer: {task.gold_answer}"
    assert steps_are_gold_perfect(task, bad_output) is None


def test_steps_are_gold_perfect_rejects_wrong_answer():
    task = generate_retrieval_task(task_id="vlr_pilot_000001", seed=1, target_context_tokens=8000)
    gold_ids = ", ".join(task.gold_evidence_ids)
    bad_output = f"Evidence: {gold_ids}\nSteps:\n1. step.\nAnswer: WRONG"
    assert steps_are_gold_perfect(task, bad_output) is None
