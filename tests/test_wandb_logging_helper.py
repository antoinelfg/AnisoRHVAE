from __future__ import annotations

from typing import Any

from src.utils import wandb_logging


class _DummyWandb:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def init(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return object()


def test_init_wandb_run_forwards_run_resume_job_type(monkeypatch) -> None:
    dummy = _DummyWandb()
    monkeypatch.setattr(wandb_logging, "_wandb", dummy)

    run = wandb_logging.init_wandb_run(
        project="proj",
        entity="ent",
        group="grp",
        tags="a,b",
        run_name="run-name",
        name_mode="manual",
        mode=None,
        config={"x": 1},
        name_prefix="prefix",
        run_id="run-123",
        resume="allow",
        job_type="three_zone_sampling",
    )

    assert run is not None
    assert len(dummy.calls) == 1
    kwargs = dummy.calls[0]
    assert kwargs["project"] == "proj"
    assert kwargs["entity"] == "ent"
    assert kwargs["group"] == "grp"
    assert kwargs["name"] == "run-name"
    assert kwargs["id"] == "run-123"
    assert kwargs["resume"] == "allow"
    assert kwargs["job_type"] == "three_zone_sampling"
    assert kwargs["tags"] == ["a", "b"]


def test_init_wandb_run_uses_env_run_id_and_resume(monkeypatch) -> None:
    dummy = _DummyWandb()
    monkeypatch.setattr(wandb_logging, "_wandb", dummy)
    monkeypatch.setenv("WANDB_RUN_ID", "env-run-id")
    monkeypatch.setenv("WANDB_RESUME", "must")

    run = wandb_logging.init_wandb_run(
        project="proj",
        entity=None,
        group=None,
        tags=None,
        run_name=None,
        name_mode="auto",
        mode=None,
        config=None,
        name_prefix="prefix",
    )

    assert run is not None
    assert len(dummy.calls) == 1
    kwargs = dummy.calls[0]
    assert kwargs["id"] == "env-run-id"
    assert kwargs["resume"] == "must"
