"""Typer CLI tying every MesaRL phase together."""

import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="MesaRL: reverse-engineering an emergent mesa-optimizer via Transformer meta-RL")


def _print_json(data: dict) -> None:
    typer.echo(json.dumps(data, indent=2))


@app.command()
def train(
    iterations: int = 300,
    batch_size: int = 256,
    run_dir: str = "runs/main",
    resume_from: Optional[str] = None,
    entropy_coef_start: Optional[float] = None,
    entropy_coef_end: Optional[float] = None,
    quick: bool = False,
):
    """Meta-train the Transformer bandit policy via PPO."""
    from training.train_meta_rl import TrainingConfig, train as train_fn

    config = TrainingConfig(num_iterations=iterations, batch_size=batch_size)
    if quick:
        config = TrainingConfig(
            num_iterations=5, batch_size=16, num_trials=20, d_model=16, n_heads=2,
            n_layers=2, d_ff=32, minibatch_size=8, ppo_epochs=2,
        )
    if entropy_coef_start is not None:
        config.entropy_coef_start = entropy_coef_start
    if entropy_coef_end is not None:
        config.entropy_coef_end = entropy_coef_end

    train_fn(config, run_dir=Path(run_dir), resume_from=resume_from)
    typer.echo(f"Training complete. Checkpoint saved to {run_dir}/checkpoint.pt")


@app.command()
def evaluate(run_dir: str = "runs/main", num_episodes: int = 1000, seed: int = 12345):
    """Evaluate frozen-weight within-episode adaptation vs a random baseline."""
    from eval.adaptation import run_adaptation_eval

    _print_json(run_adaptation_eval(Path(run_dir), num_episodes, seed))


@app.command()
def analyze(run_dir: str = "runs/main", num_episodes: int = 300, seed: int = 999):
    """Compare the trained policy against candidate bandit algorithms."""
    from analysis.compare import run_candidate_comparison

    _print_json(run_candidate_comparison(Path(run_dir), num_episodes, seed))


@app.command()
def probe(run_dir: str = "runs/main", num_episodes: int = 400, seed: int = 0):
    """Linear-probe the residual stream for a best-arm belief direction."""
    from analysis.probes import run_probe_analysis

    _print_json(run_probe_analysis(Path(run_dir), num_episodes, seed))


@app.command()
def patch(
    run_dir: str = "runs/main",
    num_episodes: int = 300,
    position_idx: Optional[int] = None,
    scale: float = 4.0,
    seed: int = 2024,
    target_arm: Optional[int] = None,
):
    """Causally test the probed belief direction via activation patching."""
    from analysis.patching import run_patching_analysis

    _print_json(run_patching_analysis(Path(run_dir), num_episodes, position_idx, scale, seed, target_arm))


@app.command()
def attention(run_dir: str = "runs/main", num_episodes: int = 200, seed: int = 0, top_k: int = 3):
    """Search attention heads for induction-head-like circuits."""
    from analysis.attention_circuits import run_attention_analysis

    result = run_attention_analysis(Path(run_dir), num_episodes, seed, top_k)
    _print_json(result["top_candidates"])


@app.command()
def shift(run_dir: str = "runs/main", num_episodes: int = 1000, seed: int = 777):
    """Evaluate the trained policy under distribution shift."""
    from shift.distribution_shift import run_distribution_shift_eval

    _print_json(run_distribution_shift_eval(Path(run_dir), num_episodes, seed))


@app.command(name="regime-probe")
def regime_probe_cmd(
    run_dir: str = "runs/main", num_episodes_per_regime: int = 200, early_window: int = 15, seed: int = 0
):
    """Probe for a train/deploy-regime-distinguishing signal."""
    from regime_probe.train_deploy_probe import run_regime_probe_analysis

    _print_json(run_regime_probe_analysis(Path(run_dir), num_episodes_per_regime, early_window, seed))


@app.command()
def audit(run_dir: str = "runs/main", config_path: str = "config.json"):
    """Run the LangGraph audit agent over a run directory's saved metrics."""
    from dotenv import load_dotenv

    from audit_agent.graph import run_audit

    load_dotenv()
    config = json.loads(Path(config_path).read_text()) if Path(config_path).exists() else {}
    report = run_audit(Path(run_dir), config=config)
    typer.echo(report)


@app.command(name="all")
def run_all(run_dir: str = "runs/main"):
    """Run every analysis phase (not training) against an existing checkpoint, in sequence."""
    from analysis.attention_circuits import run_attention_analysis
    from analysis.compare import run_candidate_comparison
    from analysis.patching import run_patching_analysis
    from eval.adaptation import run_adaptation_eval
    from regime_probe.train_deploy_probe import run_regime_probe_analysis
    from shift.distribution_shift import run_distribution_shift_eval

    typer.echo("Running adaptation evaluation...")
    run_adaptation_eval(Path(run_dir))
    typer.echo("Running candidate-algorithm comparison...")
    run_candidate_comparison(Path(run_dir))
    typer.echo("Running belief-state probing + activation patching...")
    run_patching_analysis(Path(run_dir))
    typer.echo("Running attention-circuit analysis...")
    run_attention_analysis(Path(run_dir))
    typer.echo("Running distribution-shift evaluation...")
    run_distribution_shift_eval(Path(run_dir))
    typer.echo("Running regime-probe experiment...")
    run_regime_probe_analysis(Path(run_dir))
    typer.echo(f"All analyses complete. Results saved under {run_dir}/")


if __name__ == "__main__":
    app()
