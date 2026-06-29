from benchmark.reward.programmatic import compute_reward


def test_reward_rewards_answer_and_gold_citation():
    reward = compute_reward(
        answer_score=1.0,
        citation_f1=1.0,
        reasoning_score=0.8,
        format_score=1.0,
        distractor_rate=0.0,
        stale_rate=0.0,
        invalid_rate=0.0,
    )
    assert reward.total > 0.8
    assert reward.components["answer"] == 0.40
    assert reward.components["citation"] == 0.25


def test_reward_penalizes_distractor_and_invalid_citations():
    clean = compute_reward(1.0, 1.0, 0.8, 1.0, 0.0, 0.0, 0.0)
    noisy = compute_reward(1.0, 0.5, 0.8, 1.0, 0.5, 0.0, 0.5)
    assert noisy.total < clean.total
    assert noisy.components["distractor_penalty"] < 0.0
    assert noisy.components["invalid_penalty"] < 0.0


def test_score_output_record_importable_from_benchmark_reward():
    # 抽取后必须能从新位置 import，且对一个 gold-perfect 输出打满 reward。
    from benchmark.reward.score import score_output_record
    from benchmark.generator.retrieval import generate_retrieval_task

    task = generate_retrieval_task(task_id="vlr_pilot_000001", seed=1, target_context_tokens=8000)
    gold_ids = ", ".join(task.gold_evidence_ids)
    output_text = (
        f"Evidence: {gold_ids}\nSteps:\n1. {task.gold_evidence_ids[0]} states the fact.\n"
        f"Answer: {task.gold_answer}"
    )
    scored = score_output_record(task, {"task_id": task.id, "output_text": output_text})
    assert scored["citation_precision"] == 1.0
    assert scored["answer_normalized_match"] == 1.0
    assert scored["reward_total"] > 0.8
