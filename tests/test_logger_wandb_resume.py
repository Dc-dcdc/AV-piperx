from pathlib import Path

import pytest

from lerobot.common.logger import get_wandb_resume_info_from_filesystem


def _make_checkpoint_dir(tmp_path: Path) -> Path:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


def test_wandb_resume_falls_back_to_successful_run_logs(tmp_path: Path) -> None:
    checkpoint_dir = _make_checkpoint_dir(tmp_path)
    run_dir = tmp_path / "wandb" / "run-20260716_162345-kxsihm18"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "files").mkdir()
    (run_dir / "run-kxsihm18.wandb").touch()
    (run_dir / "logs" / "debug.log").write_text(
        "run started, returning control to user process\n"
    )
    (run_dir / "files" / "output.log").write_text(
        "Track this run --> https://wandb.ai/my-entity/my-project/runs/kxsihm18\n"
    )

    assert get_wandb_resume_info_from_filesystem(checkpoint_dir) == (
        "kxsihm18",
        "my-project",
        "my-entity",
    )


def test_wandb_resume_rejects_run_without_success_marker(tmp_path: Path) -> None:
    checkpoint_dir = _make_checkpoint_dir(tmp_path)
    run_dir = tmp_path / "wandb" / "run-20260717_162345-failed123"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "files").mkdir()
    (run_dir / "run-failed123.wandb").touch()
    (run_dir / "logs" / "debug.log").write_text("wandb.init failed\n")
    (run_dir / "files" / "output.log").write_text(
        "Track this run --> https://wandb.ai/my-entity/my-project/runs/failed123\n"
    )

    with pytest.raises(RuntimeError, match="successfully initialized"):
        get_wandb_resume_info_from_filesystem(checkpoint_dir)
