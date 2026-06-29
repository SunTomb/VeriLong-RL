import json

from benchmark.generator.multihop import generate_multihop_task
from benchmark.generator.retrieval import generate_retrieval_task
from benchmark.generator.temporal import generate_temporal_task
from experiments.eval_api.claude_client import (
    NO_SAMPLING_MODELS,
    STREAM_MAX_TOKENS_THRESHOLD,
    ClaudeClient,
    _extract_text,
)
from experiments.eval_api.run_api_eval import (
    PROMPT_VERSION,
    build_user_prompt,
    cache_path,
    compute_task_hash,
    load_cache,
    load_tasks,
    run,
)
from scripts.score_outputs import score_output_record


def _task():
    return generate_retrieval_task("vlr_pilot_000001", seed=1, target_context_tokens=8000)


def _write_tasks(path, tasks):
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.model_dump(mode="json"), ensure_ascii=False) + "\n")


# -- prompt contract ----------------------------------------------------------


def test_user_prompt_exposes_only_neutral_fields():
    task = _task()
    prompt = build_user_prompt(task)

    assert task.question in prompt
    for document in task.documents:
        assert document.evidence_id in prompt
        assert document.text in prompt

    # Must not leak supervision signals as labeled fields. The answer string
    # itself legitimately appears inside the gold evidence text (the task must
    # be answerable from the documents); what we forbid is exposing the labels
    # gold_answer / gold_evidence / distractor / stale / reward.
    lowered = prompt.lower()
    assert "gold_answer" not in lowered
    assert "gold answer" not in lowered
    assert "gold_evidence" not in lowered
    assert "distractor" not in lowered
    assert "stale" not in lowered
    assert "reward" not in lowered
    # The role labels carried on each document must not be serialized.
    for role in ("'role'", '"role"', "role=", "role:"):
        assert role not in lowered


def test_task_hash_is_stable_and_input_bound():
    task = _task()
    prompt = build_user_prompt(task)
    assert compute_task_hash(task, prompt) == compute_task_hash(task, prompt)
    assert compute_task_hash(task, prompt) != compute_task_hash(task, prompt + " x")


# -- client constraints -------------------------------------------------------


def test_default_model_omits_sampling_and_budget_params():
    client = ClaudeClient(model="claude-opus-4-8")
    assert client.model in NO_SAMPLING_MODELS
    kwargs = client.build_request_kwargs("sys", "user")

    for forbidden in ("temperature", "top_p", "top_k", "budget_tokens"):
        assert forbidden not in kwargs
    assert kwargs["thinking"] == {"type": "adaptive"}
    # No assistant prefill: only a single user message.
    assert [m["role"] for m in kwargs["messages"]] == ["user"]


def test_streaming_threshold():
    assert ClaudeClient(max_tokens=STREAM_MAX_TOKENS_THRESHOLD).should_stream()
    assert not ClaudeClient(max_tokens=STREAM_MAX_TOKENS_THRESHOLD - 1).should_stream()


def test_extract_text_skips_thinking_blocks():
    message = {
        "content": [
            {"type": "thinking", "text": "hidden reasoning"},
            {"type": "text", "text": "Answer: A17"},
        ]
    }
    assert _extract_text(message) == "Answer: A17"


# -- cache round-trip ---------------------------------------------------------


def test_cache_round_trip_and_invalidation(tmp_path):
    task = _task()
    tasks_path = tmp_path / "tasks.jsonl"
    out_path = tmp_path / "out.jsonl"
    cache_root = tmp_path / "cache"
    _write_tasks(tasks_path, [task])

    # Dry run does not write cache (it is not a real prediction).
    run(
        tasks_path=tasks_path,
        out_path=out_path,
        model="claude-opus-4-8",
        split=None,
        limit=None,
        cache_root=cache_root,
        dry_run=True,
        max_tokens=2048,
    )
    assert not cache_path(cache_root, "claude-opus-4-8", task.id).exists()

    # A hand-written cache record with a matching hash is treated as a hit.
    prompt = build_user_prompt(task)
    task_hash = compute_task_hash(task, prompt)
    cpath = cache_path(cache_root, "claude-opus-4-8", task.id)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(
        json.dumps(
            {
                "task_id": task.id,
                "model": "claude-opus-4-8",
                "prompt_version": PROMPT_VERSION,
                "task_hash": task_hash,
                "output_text": "Evidence: E01\nSteps:\n1. x\nAnswer: y",
            }
        ),
        encoding="utf-8",
    )
    assert load_cache(cpath, model="claude-opus-4-8", task_hash=task_hash) is not None
    # Mismatched hash / model / version invalidate the entry.
    assert load_cache(cpath, model="claude-opus-4-8", task_hash="sha256:other") is None
    assert load_cache(cpath, model="other-model", task_hash=task_hash) is None


# -- pipeline + scorer wiring -------------------------------------------------


def test_dry_run_output_is_scorable(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    out_path = tmp_path / "out.jsonl"
    cache_root = tmp_path / "cache"
    tasks = [
        _task(),
        generate_temporal_task("vlr_pilot_000002", seed=2, target_context_tokens=8000),
    ]
    _write_tasks(tasks_path, tasks)

    summary = run(
        tasks_path=tasks_path,
        out_path=out_path,
        model="claude-opus-4-8",
        split=None,
        limit=None,
        cache_root=cache_root,
        dry_run=True,
        max_tokens=2048,
    )
    assert summary["tasks"] == 2
    assert summary["source"] == "dry_run"
    assert summary["api_calls"] == 0

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    for task, row in zip(tasks, rows, strict=True):
        assert row["task_id"] == task.id
        assert row["metadata"]["source"] == "dry_run"
        scored = score_output_record(task, row)
        # Dry-run stub echoes gold, so it must parse and score cleanly.
        assert scored["format_valid"] == 1
        assert scored["citation_f1"] == 1.0


def test_split_filter_limits_tasks(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    out_path = tmp_path / "out.jsonl"
    cache_root = tmp_path / "cache"

    tasks = []
    for i in range(1, 5):
        task = generate_retrieval_task(f"vlr_pilot_{i:06d}", seed=i, target_context_tokens=8000)
        task.metadata.split = "dev" if i % 2 == 0 else "train"
        tasks.append(task)
    _write_tasks(tasks_path, tasks)

    summary = run(
        tasks_path=tasks_path,
        out_path=out_path,
        model="claude-opus-4-8",
        split="dev",
        limit=None,
        cache_root=cache_root,
        dry_run=True,
        max_tokens=2048,
    )
    assert summary["tasks"] == 2


def test_stratified_sample_spans_families(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    # Grouped by family, as the real dataset is laid out within a split.
    tasks = []
    for i in range(1, 7):
        tasks.append(generate_retrieval_task(f"vlr_pilot_{i:06d}", seed=i, target_context_tokens=8000))
    for i in range(7, 13):
        tasks.append(generate_multihop_task(f"vlr_pilot_{i:06d}", seed=i, hop_count=3, target_context_tokens=8000))
    for i in range(13, 19):
        tasks.append(generate_temporal_task(f"vlr_pilot_{i:06d}", seed=i, target_context_tokens=8000))
    _write_tasks(tasks_path, tasks)

    # Non-stratified head slice stays within the first family.
    head = load_tasks(tasks_path, split=None, limit=3, stratify=False)
    assert {t.task_family for t in head} == {"anti_distractor_retrieval"}

    # Stratified sampling spans all three families.
    sampled = load_tasks(tasks_path, split=None, limit=6, stratify=True)
    assert len(sampled) == 6
    assert {t.task_family for t in sampled} == {
        "anti_distractor_retrieval",
        "multi_hop_reasoning",
        "temporal_update",
    }
