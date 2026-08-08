from pathlib import Path

import torch
from typer.testing import CliRunner

from cli import app
from training.train_meta_rl import TrainingConfig, build_policy

runner = CliRunner()

TINY_CONFIG = TrainingConfig(
    num_arms=3,
    num_trials=20,
    d_model=16,
    n_heads=2,
    n_layers=2,
    d_ff=32,
    batch_size=8,
)


def make_checkpoint(run_dir: Path) -> None:
    policy = build_policy(TINY_CONFIG)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": policy.state_dict(), "config": TINY_CONFIG.__dict__},
        run_dir / "checkpoint.pt",
    )


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ["train", "evaluate", "analyze", "probe", "patch", "attention", "shift", "audit"]:
        assert command in result.stdout


def test_train_command_respects_seed_and_saves_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run"
    result = runner.invoke(app, ["train", "--quick", "--seed", "7", "--run-dir", str(run_dir)])
    assert result.exit_code == 0, result.output
    assert (run_dir / "checkpoint.pt").exists()

    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    assert checkpoint["config"]["seed"] == 7


def test_evaluate_command(tmp_path: Path):
    run_dir = tmp_path / "run"
    make_checkpoint(run_dir)
    result = runner.invoke(app, ["evaluate", "--run-dir", str(run_dir), "--num-episodes", "10"])
    assert result.exit_code == 0, result.output
    assert (run_dir / "adaptation_eval.json").exists()


def test_analyze_command(tmp_path: Path):
    run_dir = tmp_path / "run"
    make_checkpoint(run_dir)
    result = runner.invoke(app, ["analyze", "--run-dir", str(run_dir), "--num-episodes", "10"])
    assert result.exit_code == 0, result.output
    assert (run_dir / "candidate_comparison.json").exists()


def test_probe_command(tmp_path: Path):
    run_dir = tmp_path / "run"
    make_checkpoint(run_dir)
    result = runner.invoke(app, ["probe", "--run-dir", str(run_dir), "--num-episodes", "20"])
    assert result.exit_code == 0, result.output
    assert (run_dir / "probe_results.json").exists()


def test_patch_command(tmp_path: Path):
    run_dir = tmp_path / "run"
    make_checkpoint(run_dir)
    result = runner.invoke(app, ["patch", "--run-dir", str(run_dir), "--num-episodes", "20"])
    assert result.exit_code == 0, result.output
    assert (run_dir / "patching_results.json").exists()


def test_attention_command(tmp_path: Path):
    run_dir = tmp_path / "run"
    make_checkpoint(run_dir)
    result = runner.invoke(app, ["attention", "--run-dir", str(run_dir), "--num-episodes", "10"])
    assert result.exit_code == 0, result.output
    assert (run_dir / "attention_circuits.json").exists()


def test_shift_command(tmp_path: Path):
    run_dir = tmp_path / "run"
    make_checkpoint(run_dir)
    result = runner.invoke(app, ["shift", "--run-dir", str(run_dir), "--num-episodes", "10"])
    assert result.exit_code == 0, result.output
    assert (run_dir / "distribution_shift.json").exists()


def test_regime_probe_command(tmp_path: Path):
    run_dir = tmp_path / "run"
    make_checkpoint(run_dir)
    result = runner.invoke(
        app, ["regime-probe", "--run-dir", str(run_dir), "--num-episodes-per-regime", "15"]
    )
    assert result.exit_code == 0, result.output
    assert (run_dir / "regime_probe.json").exists()


def test_run_all_command(tmp_path: Path):
    run_dir = tmp_path / "run"
    make_checkpoint(run_dir)
    result = runner.invoke(app, ["all", "--run-dir", str(run_dir)])
    assert result.exit_code == 0, result.output
    for filename in [
        "adaptation_eval.json", "candidate_comparison.json", "patching_results.json",
        "attention_circuits.json", "distribution_shift.json", "regime_probe.json",
    ]:
        assert (run_dir / filename).exists()
