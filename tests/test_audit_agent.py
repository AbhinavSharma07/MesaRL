import json
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from audit_agent.graph import build_audit_graph, ingest_metrics, run_audit


def test_ingest_metrics_reads_only_existing_files(tmp_path: Path):
    (tmp_path / "adaptation_eval.json").write_text(json.dumps({"a": 1}))
    (tmp_path / "regime_probe.json").write_text(json.dumps({"b": 2}))
    metrics = ingest_metrics(tmp_path)
    assert metrics == {"adaptation_eval": {"a": 1}, "regime_probe": {"b": 2}}


def test_ingest_metrics_empty_run_dir_returns_empty_dict(tmp_path: Path):
    assert ingest_metrics(tmp_path) == {}


def test_ingest_metrics_strips_long_per_trial_curves(tmp_path: Path):
    long_curve = list(range(100))
    short_list = [1, 2, 3]
    (tmp_path / "candidate_comparison.json").write_text(
        json.dumps({"summary": {"most_similar_candidate": "Epsilon-greedy"}, "mean_kl_per_trial": long_curve})
    )
    (tmp_path / "regime_probe.json").write_text(json.dumps({"short_list": short_list}))

    metrics = ingest_metrics(tmp_path)

    assert metrics["candidate_comparison"]["summary"] == {"most_similar_candidate": "Epsilon-greedy"}
    assert isinstance(metrics["candidate_comparison"]["mean_kl_per_trial"], str)
    assert "100-point curve" in metrics["candidate_comparison"]["mean_kl_per_trial"]
    assert metrics["regime_probe"]["short_list"] == short_list  # untouched, below the threshold


def test_ingest_metrics_drops_bulky_sweep_field(tmp_path: Path):
    (tmp_path / "patching_sweep.json").write_text(
        json.dumps({"interpretation": "the key finding", "sweep": {"favored_arm_1": [{"scale": 4.0}]}})
    )
    metrics = ingest_metrics(tmp_path)
    assert metrics["patching_sweep"]["interpretation"] == "the key finding"
    assert isinstance(metrics["patching_sweep"]["sweep"], str)


def test_run_audit_pipeline_with_fake_llm(tmp_path: Path):
    (tmp_path / "adaptation_eval.json").write_text(
        json.dumps({"trained_policy": {"improvement_ratio": 0.2}, "random_baseline": {"improvement_ratio": 0.0}})
    )

    fake_llm = FakeListChatModel(responses=["draft report text", "critique text", "final report text"])
    report = run_audit(tmp_path, llm=fake_llm, save=True)

    assert report == "final report text"
    assert (tmp_path / "audit_report.md").read_text() == "final report text"


def test_build_audit_graph_state_flows_through_all_stages(tmp_path: Path):
    (tmp_path / "regime_probe.json").write_text(json.dumps({"best_layer": "layer_1"}))
    fake_llm = FakeListChatModel(responses=["DRAFT", "CRITIQUE", "FINAL"])
    app = build_audit_graph(llm=fake_llm)

    from audit_agent.graph import AuditState

    final_state = app.invoke(AuditState(run_dir=str(tmp_path)))
    assert final_state["metrics"] == {"regime_probe": {"best_layer": "layer_1"}}
    assert final_state["draft_report"] == "DRAFT"
    assert final_state["critique"] == "CRITIQUE"
    assert final_state["final_report"] == "FINAL"
