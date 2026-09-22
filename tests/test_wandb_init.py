"""WandB initialization accepts overrides while preserving resume identity."""

import re
import sys

import fxr.experiment.train as module
from tests.test_training_recipe import _ToyTrainExperiment, _toy_config


def test_custom_wandb_keywords_merge_and_identity_stays_stable(tmp_path, monkeypatch):
    monkeypatch.setenv("FXR_DEVICE", "cpu")
    config = _toy_config(tmp_path)
    config["log"]["wandb"].update({
        "name": "custom", "dir": str(tmp_path / "wandb"),
        "config": {"test_tag": "overrides"},
    })
    experiment = _ToyTrainExperiment.from_config(config)
    calls = []
    monkeypatch.setattr(module, "_generate_wandb_id", lambda: "test-id")
    monkeypatch.setattr(module.wandb, "init", lambda **kwargs: calls.append(kwargs))
    try:
        experiment._init_wandb()
        experiment._init_wandb()
        assert [c["id"] for c in calls] == ["test-id", "test-id"]
        assert calls[0]["name"] == "custom"
        assert calls[0]["dir"] == str(tmp_path / "wandb")
        assert calls[0]["config"]["test_tag"] == "overrides"
        assert calls[0]["config"]["optim"]["lr"] == 0.1
        assert calls[0]["resume"] == "allow"
    finally:
        experiment.close()


def test_wandb_id_has_local_fallback_when_sdk_helper_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb.sdk.lib.runid", None)
    assert re.fullmatch("[0-9a-f]{8}", module._generate_wandb_id())
