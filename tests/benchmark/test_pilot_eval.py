import json

from benchmark.eval.run_pilot_eval import run_pilot_eval
from benchmark.generator.multihop import generate_multihop_task
from benchmark.generator.retrieval import generate_retrieval_task
from benchmark.generator.temporal import generate_temporal_task


def _write_tasks(path, tasks):
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.model_dump(mode="json"), ensure_ascii=False) + "\n")


def test_oracle_baseline_scores_perfect_citations(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    out_dir = tmp_path / "oracle"
    tasks = [generate_retrieval_task("vlr_pilot_000001", seed=1, target_context_tokens=8000)]
    _write_tasks(tasks_path, tasks)

    summary = run_pilot_eval(tasks_path=tasks_path, baseline="oracle_format_baseline", out_dir=out_dir)

    assert summary["count"] == 1
    assert summary["format_valid_mean"] == 1.0
    assert summary["citation_f1_mean"] == 1.0
    assert (out_dir / "outputs.jsonl").exists()
    assert (out_dir / "scored.jsonl").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "cases_for_demo.json").exists()


def test_corrupted_baseline_cites_distractor_when_available(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    out_dir = tmp_path / "corrupted"
    tasks = [
        generate_retrieval_task("vlr_pilot_000001", seed=1, target_context_tokens=8000),
        generate_temporal_task("vlr_pilot_000002", seed=2, target_context_tokens=8000),
    ]
    _write_tasks(tasks_path, tasks)

    summary = run_pilot_eval(tasks_path=tasks_path, baseline="corrupted_distractor_baseline", out_dir=out_dir)

    assert summary["count"] == 2
    assert summary["distractor_citation_rate_mean"] > 0.0
    cases = json.loads((out_dir / "cases_for_demo.json").read_text(encoding="utf-8"))
    assert cases[0]["question"]
    assert cases[0]["documents"]
    assert cases[0]["model_output"]
    assert cases[0]["metric_breakdown"]


def test_demo_cases_cover_every_task_family(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    out_dir = tmp_path / "oracle"
    # Order tasks grouped by family, as build_pilot does, with enough of the
    # first family to fill the demo limit. A naive head-of-list export would
    # then surface only anti_distractor_retrieval.
    tasks = [
        generate_retrieval_task(f"vlr_pilot_{i:06d}", seed=i, target_context_tokens=8000)
        for i in range(1, 31)
    ]
    tasks.append(generate_multihop_task("vlr_pilot_000031", seed=31, hop_count=3, target_context_tokens=8000))
    tasks.append(generate_temporal_task("vlr_pilot_000032", seed=32, target_context_tokens=8000))
    _write_tasks(tasks_path, tasks)

    run_pilot_eval(tasks_path=tasks_path, baseline="oracle_format_baseline", out_dir=out_dir)

    cases = json.loads((out_dir / "cases_for_demo.json").read_text(encoding="utf-8"))
    families = {case["task_family"] for case in cases}
    assert families == {
        "anti_distractor_retrieval",
        "multi_hop_reasoning",
        "temporal_update",
    }
