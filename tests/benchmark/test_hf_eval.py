import json

from benchmark.generator.multihop import generate_multihop_task
from benchmark.generator.retrieval import generate_retrieval_task
from benchmark.generator.temporal import generate_temporal_task
from experiments.eval_open_source.run_hf_eval import build_chat_messages, run
from experiments.eval_api.run_api_eval import SYSTEM_PROMPT
from scripts.score_outputs import score_output_record


def _write_tasks(path, tasks):
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.model_dump(mode="json"), ensure_ascii=False) + "\n")


def test_chat_messages_use_shared_system_prompt():
    messages = build_chat_messages("hello")
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "hello"}


def test_fake_run_is_scorable_across_families(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    out_path = tmp_path / "out.jsonl"
    tasks = [
        generate_retrieval_task("vlr_pilot_000001", seed=1, target_context_tokens=8000),
        generate_multihop_task("vlr_pilot_000002", seed=2, hop_count=3, target_context_tokens=8000),
        generate_temporal_task("vlr_pilot_000003", seed=3, target_context_tokens=8000),
    ]
    _write_tasks(tasks_path, tasks)

    summary = run(
        tasks_path=tasks_path,
        out_path=out_path,
        model_path="unused",
        split=None,
        limit=None,
        stratify=False,
        max_new_tokens=128,
        fake=True,
    )
    assert summary["tasks"] == 3
    assert summary["fake"] is True

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 3
    for task, row in zip(tasks, rows, strict=True):
        assert row["task_id"] == task.id
        assert row["metadata"]["source"] == "fake"
        scored = score_output_record(task, row)
        assert scored["format_valid"] == 1
        assert scored["citation_f1"] == 1.0
