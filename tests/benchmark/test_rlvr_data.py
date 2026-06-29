import json

from experiments.rlvr.data import build_prompt_messages, iter_grpo_rows
from benchmark.generator.retrieval import generate_retrieval_task
from experiments.eval_api.run_api_eval import SYSTEM_PROMPT


def test_build_prompt_messages_uses_system_and_user():
    task = generate_retrieval_task(task_id="vlr_pilot_000001", seed=1, target_context_tokens=8000)
    messages = build_prompt_messages(task)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"
    assert task.question in messages[1]["content"]


def test_iter_grpo_rows_filters_family_and_context(tmp_path):
    keep = generate_retrieval_task(task_id="vlr_pilot_000001", seed=1, target_context_tokens=8000)
    keep.metadata.split = "train"
    drop_ctx = generate_retrieval_task(task_id="vlr_pilot_000002", seed=2, target_context_tokens=16000)
    drop_ctx.metadata.split = "train"
    path = tmp_path / "tasks.jsonl"
    path.write_text(keep.model_dump_json() + "\n" + drop_ctx.model_dump_json() + "\n", encoding="utf-8")

    rows = list(iter_grpo_rows(path, family="anti_distractor_retrieval", max_context_tokens=8000, split="train"))
    assert len(rows) == 1
    row = rows[0]
    assert set(row.keys()) == {"prompt", "task_json"}
    assert isinstance(row["prompt"], list)
    assert json.loads(row["task_json"])["id"] == "vlr_pilot_000001"


def test_iter_grpo_rows_empty_when_nothing_matches(tmp_path):
    task = generate_retrieval_task(task_id="vlr_pilot_000001", seed=1, target_context_tokens=8000)
    task.metadata.split = "train"
    path = tmp_path / "tasks.jsonl"
    path.write_text(task.model_dump_json() + "\n", encoding="utf-8")
    # wrong split -> nothing matches
    rows = list(iter_grpo_rows(path, family="anti_distractor_retrieval", max_context_tokens=8000, split="test"))
    assert rows == []
