import json

from benchmark.generator.retrieval import generate_retrieval_task
from benchmark.reward.score import score_output_record
from experiments.rlvr.reward import make_reward_fn


def _task_and_json():
    task = generate_retrieval_task(task_id="vlr_pilot_000001", seed=1, target_context_tokens=8000)
    return task, task.model_dump_json()


def test_reward_fn_matches_score_output_record_for_gold_perfect():
    task, task_json = _task_and_json()
    gold_ids = ", ".join(task.gold_evidence_ids)
    text = (
        f"Evidence: {gold_ids}\nSteps:\n1. {task.gold_evidence_ids[0]} states the fact.\n"
        f"Answer: {task.gold_answer}"
    )
    reward_fn = make_reward_fn()
    got = reward_fn(prompts=["ignored"], completions=[text], task_json=[task_json])
    expected = score_output_record(task, {"output_text": text})["reward_total"]
    assert abs(got[0] - expected) < 1e-6


def test_reward_fn_penalizes_distractor_citation():
    task, task_json = _task_and_json()
    distractor = task.distractor_evidence_ids[0]
    good = f"Evidence: {task.gold_evidence_ids[0]}\nSteps:\n1. ok.\nAnswer: {task.gold_answer}"
    bad = f"Evidence: {task.gold_evidence_ids[0]}, {distractor}\nSteps:\n1. ok.\nAnswer: {task.gold_answer}"
    reward_fn = make_reward_fn()
    rewards = reward_fn(prompts=["x", "x"], completions=[good, bad], task_json=[task_json, task_json])
    assert rewards[0] > rewards[1]


def test_reward_fn_accepts_chat_style_completion():
    task, task_json = _task_and_json()
    text = f"Evidence: {task.gold_evidence_ids[0]}\nSteps:\n1. ok.\nAnswer: {task.gold_answer}"
    reward_fn = make_reward_fn()
    chat = [[{"role": "assistant", "content": text}]]
    got = reward_fn(prompts=["x"], completions=chat, task_json=[task_json])
    assert got[0] > 0.0
