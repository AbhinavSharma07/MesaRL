"""Gradio dashboard: browse a run directory's saved analysis results."""

import json
from pathlib import Path

import gradio as gr

DEFAULT_RUN_DIR = "runs/main"


def _load_json(run_dir: str, filename: str) -> str:
    path = Path(run_dir) / filename
    if not path.exists():
        return f"(no {filename} found in {run_dir} -- run the corresponding CLI command first)"
    return json.dumps(json.loads(path.read_text()), indent=2)


def _load_image(run_dir: str, filename: str):
    path = Path(run_dir) / filename
    return str(path) if path.exists() else None


def load_run(run_dir: str):
    return (
        _load_json(run_dir, "adaptation_eval.json"),
        _load_image(run_dir, "adaptation_regret_curve.png"),
        _load_json(run_dir, "candidate_comparison.json"),
        _load_image(run_dir, "candidate_comparison_kl.png"),
        _load_json(run_dir, "probe_results.json"),
        _load_json(run_dir, "patching_results.json"),
        _load_image(run_dir, "attention_circuits_heatmap.png"),
        _load_image(run_dir, "distribution_shift_curves.png"),
        _load_image(run_dir, "distribution_shift_shock.png"),
        _load_json(run_dir, "distribution_shift.json"),
        _load_image(run_dir, "regime_probe_accuracy.png"),
        _load_json(run_dir, "regime_probe.json"),
        (
            (Path(run_dir) / "audit_report.md").read_text()
            if (Path(run_dir) / "audit_report.md").exists()
            else "(no audit_report.md found -- run `python cli.py audit` first, requires an LLM API key)"
        ),
    )


with gr.Blocks(title="MesaRL Dashboard") as demo:
    gr.Markdown("# MesaRL — Mesa-Optimizer Analysis Dashboard")
    run_dir_input = gr.Textbox(value=DEFAULT_RUN_DIR, label="Run directory")
    load_button = gr.Button("Load", variant="primary")

    with gr.Tab("Adaptation"):
        adaptation_plot = gr.Image(label="Regret curve: trained policy vs random baseline")
        adaptation_json = gr.Code(label="Summary", language="json")

    with gr.Tab("Candidate comparison"):
        compare_plot = gr.Image(label="KL divergence to known bandit algorithms")
        compare_json = gr.Code(label="Summary", language="json")

    with gr.Tab("Belief probing & patching"):
        probe_json = gr.Code(label="Probe accuracy per layer", language="json")
        patching_json = gr.Code(label="Activation patching result", language="json")

    with gr.Tab("Attention circuits"):
        attention_heatmap = gr.Image(label="Same-arm attention bias (induction-head search)")

    with gr.Tab("Distribution shift"):
        shift_curves_plot = gr.Image(label="Regret under distribution shift")
        shift_shock_plot = gr.Image(label="Shock & recovery around a non-stationary change point")
        shift_json = gr.Code(label="Summary", language="json")

    with gr.Tab("Regime probe"):
        regime_plot = gr.Image(label="Decoding train-like vs held-out-signature regime")
        regime_json = gr.Code(label="Summary", language="json")

    with gr.Tab("Audit report"):
        audit_report = gr.Markdown()

    load_button.click(
        load_run,
        inputs=[run_dir_input],
        outputs=[
            adaptation_json, adaptation_plot,
            compare_json, compare_plot,
            probe_json, patching_json,
            attention_heatmap,
            shift_curves_plot, shift_shock_plot, shift_json,
            regime_plot, regime_json,
            audit_report,
        ],
    )
    demo.load(
        load_run,
        inputs=[run_dir_input],
        outputs=[
            adaptation_json, adaptation_plot,
            compare_json, compare_plot,
            probe_json, patching_json,
            attention_heatmap,
            shift_curves_plot, shift_shock_plot, shift_json,
            regime_plot, regime_json,
            audit_report,
        ],
    )


if __name__ == "__main__":
    demo.launch()
