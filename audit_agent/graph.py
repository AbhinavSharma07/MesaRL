"""LangGraph pipeline: ingest all analysis JSON outputs -> LLM drafts an
audit report -> LLM critiques it against the raw metrics -> LLM revises."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from audit_agent.llm_factory import build_chat_llm
from audit_agent.prompts import (
    CRITIQUE_SYSTEM_PROMPT,
    CRITIQUE_USER_TEMPLATE,
    DRAFT_SYSTEM_PROMPT,
    DRAFT_USER_TEMPLATE,
    REVISE_SYSTEM_PROMPT,
    REVISE_USER_TEMPLATE,
)

METRIC_FILES = [
    "adaptation_eval.json",
    "candidate_comparison.json",
    "probe_results.json",
    "patching_results.json",
    "patching_sweep.json",
    "attention_circuits.json",
    "distribution_shift.json",
    "regime_probe.json",
]


_MAX_LIST_LEN = 20  # longer lists are per-trial curves, not summary stats -- drop for the LLM prompt
_DROP_KEYS = {"sweep"}  # bulky nested detail already summarized elsewhere (e.g. patching_sweep's "interpretation")


def _drop_long_lists(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: (f"<omitted: see the actual {k} field in the JSON file>" if k in _DROP_KEYS else _drop_long_lists(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return f"<omitted: {len(obj)}-point curve, see the actual JSON file>" if len(obj) > _MAX_LIST_LEN else obj
    return obj


def ingest_metrics(run_dir: Path) -> Dict[str, Any]:
    """Per-trial curves (e.g. 100-point KL/regret arrays) are stripped before
    this reaches the LLM prompt -- they blow past small-model token limits
    and add nothing a summary stat doesn't already say."""
    run_dir = Path(run_dir)
    metrics = {}
    for filename in METRIC_FILES:
        path = run_dir / filename
        if path.exists():
            metrics[filename.removesuffix(".json")] = _drop_long_lists(json.loads(path.read_text(encoding="utf-8")))
    return metrics


class AuditState(BaseModel):
    run_dir: str
    metrics: Dict[str, Any] = {}
    draft_report: Optional[str] = None
    critique: Optional[str] = None
    final_report: Optional[str] = None


def ingest_node(state: AuditState) -> AuditState:
    state.metrics = ingest_metrics(Path(state.run_dir))
    return state


DRAFT_MAX_TOKENS = 1200
CRITIQUE_MAX_TOKENS = 400
REVISE_MAX_TOKENS = 2500


def make_draft_node(llm):
    bound_llm = llm.bind(max_tokens=DRAFT_MAX_TOKENS)

    def draft_node(state: AuditState) -> AuditState:
        response = bound_llm.invoke([
            ("system", DRAFT_SYSTEM_PROMPT),
            ("user", DRAFT_USER_TEMPLATE.format(
                run_dir=state.run_dir, metrics_json=json.dumps(state.metrics, indent=2)
            )),
        ])
        state.draft_report = response.content
        return state

    return draft_node


def make_critique_node(llm):
    bound_llm = llm.bind(max_tokens=CRITIQUE_MAX_TOKENS)

    def critique_node(state: AuditState) -> AuditState:
        response = bound_llm.invoke([
            ("system", CRITIQUE_SYSTEM_PROMPT),
            ("user", CRITIQUE_USER_TEMPLATE.format(
                metrics_json=json.dumps(state.metrics, indent=2), draft_report=state.draft_report
            )),
        ])
        state.critique = response.content
        return state

    return critique_node


def make_revise_node(llm):
    bound_llm = llm.bind(max_tokens=REVISE_MAX_TOKENS)

    def revise_node(state: AuditState) -> AuditState:
        response = bound_llm.invoke([
            ("system", REVISE_SYSTEM_PROMPT),
            ("user", REVISE_USER_TEMPLATE.format(
                metrics_json=json.dumps(state.metrics, indent=2),
                draft_report=state.draft_report,
                critique=state.critique,
            )),
        ])
        state.final_report = response.content
        return state

    return revise_node


def build_audit_graph(llm=None, config: Optional[dict] = None):
    llm = llm or build_chat_llm(config)
    workflow = StateGraph(AuditState)
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("draft", make_draft_node(llm))
    workflow.add_node("critique", make_critique_node(llm))
    workflow.add_node("revise", make_revise_node(llm))

    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "draft")
    workflow.add_edge("draft", "critique")
    workflow.add_edge("critique", "revise")
    workflow.add_edge("revise", END)
    return workflow.compile()


def run_audit(run_dir: Path, llm=None, config: Optional[dict] = None, save: bool = True) -> str:
    app = build_audit_graph(llm=llm, config=config)
    final_state = app.invoke(AuditState(run_dir=str(run_dir)))
    final_report = final_state.get("final_report") if isinstance(final_state, dict) else final_state.final_report
    if save:
        (Path(run_dir) / "audit_report.md").write_text(final_report, encoding="utf-8")
    return final_report


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the LangGraph audit agent over a run directory's metrics")
    parser.add_argument("--run-dir", type=str, default="runs/main")
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if Path(args.config).exists() else {}
    report = run_audit(Path(args.run_dir), config=config)
    print(report)
