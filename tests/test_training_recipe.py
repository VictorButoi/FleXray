"""Behavior tests for the current training recipe and runtime policies."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from fxr.augmentation import (
    MinMaxNormalize,
    PercentileMinMaxNormalize,
    Standardize,
    build_input_normalizer,
    resolve_input_normalization_config,
)
import fxr.experiment.train as train_module
from fxr.experiment._device import cuda_compatibility_error, resolve_training_device
from fxr.experiment.compile import maybe_compile_model, resolve_torch_compile_config
from fxr.experiment.ema import EmaPolicy, ExponentialMovingAverage, resolve_ema_policy
from fxr.config import config_digest
from fxr.datasets import TrainingDatasetBundle
from fxr.experiment.train import TrainExperiment


def test_device_policy_rejects_a_cuda_build_without_visible_gpu_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CUDA build that omits Volta must not be selected for a V100."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (7, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Tesla V100")
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_80"])

    error = cuda_compatibility_error()
    assert error is not None
    assert "compute capability 7.0" in error
    assert "CUDA 12.6" in error
    assert resolve_training_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="--device cuda.*CUDA 12.6"):
        resolve_training_device("cuda")


def test_device_policy_accepts_compatible_cuda_and_explicit_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compatible CUDA is selected while an explicit CPU choice always wins."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (7, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Tesla V100")
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_70", "sm_80"])

    assert cuda_compatibility_error() is None
    assert resolve_training_device("auto").type == "cuda"
    assert resolve_training_device("cuda").type == "cuda"
    assert resolve_training_device("cpu").type == "cpu"


class _ToyTrainExperiment(TrainExperiment):
    """Tiny scalar model used to exercise generic training state behavior."""

    def build_model(self) -> None:
        """Build a deterministic one-weight model.

        Args:
            None.

        Returns:
            ``None``.
        """

        self.model = torch.nn.Linear(1, 1, bias=False).to(self.device)
        with torch.no_grad():
            self.model.weight.zero_()

    def build_augmentations(self) -> None:
        """Leave augmentation disabled for scalar behavior tests.

        Args:
            None.

        Returns:
            ``None``.
        """

    def build_loss(self) -> None:
        """Leave loss construction disabled for scalar behavior tests.

        Args:
            None.

        Returns:
            ``None``.
        """

    def build_data(self, load_data: bool) -> None:
        """Build one-item in-memory train and validation iterables.

        Args:
            load_data: Whether the tiny iterables should be populated.

        Returns:
            ``None``.
        """

        if load_data:
            self.train_dl = [torch.ones(1, 1, device=self.device)]
            self.val_dl = [torch.ones(1, 1, device=self.device)]

    def run_step(
        self,
        batch: torch.Tensor,
        *,
        phase: str,
        backward: bool,
    ) -> dict[str, Any]:
        """Return the current scalar prediction without changing weights.

        Args:
            batch: One scalar model input.
            phase: Current train or validation phase.
            backward: Whether an optimization step was requested.

        Returns:
            Output mapping consumed by :meth:`compute_metrics`.
        """

        del phase, backward
        prediction = self.model(batch)
        return {"y_pred": prediction, "value": float(prediction.item())}

    def compute_metrics(self, outputs: dict[str, Any]) -> dict[str, float]:
        """Expose the scalar prediction as a metric.

        Args:
            outputs: Mapping returned by :meth:`run_step`.

        Returns:
            Single ``value`` metric.
        """

        return {"value": float(outputs["value"])}


def _toy_config(root: Path) -> dict[str, Any]:
    """Return a complete tiny training config with EMA enabled.

    Args:
        root: Run-directory parent.

    Returns:
        Config mapping accepted by :class:`_ToyTrainExperiment`.
    """

    return {
        "experiment": {"seed": 0},
        "train": {
            "epochs": 1,
            "ema": {
                "enabled": True,
                "decay": 0.5,
                "start_after_steps": 0,
                "update_every": 1,
            },
        },
        "optim": {"_class": "torch.optim.SGD", "lr": 0.1},
        "log": {
            "root": str(root),
            "wandb": {"mode": "disabled"},
            "model_weights": {"save_freq": 0},
        },
    }


def test_percentile_normalization_clips_outliers_per_sample() -> None:
    image = torch.tensor(
        [
            [[0.0, 1.0, 2.0, 3.0, 100.0]],
            [[10.0, 20.0, 30.0, 40.0, 50.0]],
        ]
    )
    normalizer = PercentileMinMaxNormalize(percentiles=(25.0, 75.0))

    normalized = normalizer(image)

    expected = torch.tensor(
        [
            [[0.0, 0.0, 0.5, 1.0, 1.0]],
            [[0.0, 0.0, 0.5, 1.0, 1.0]],
        ]
    )
    assert torch.allclose(normalized, expected)


def test_normalization_config_builds_current_and_legacy_behaviors() -> None:
    current = build_input_normalizer({"train": {"normalization": {}}})
    legacy = build_input_normalizer({"train": {}})
    minmax = build_input_normalizer(
        {"train": {"normalization": {"scheme": "minmax"}}}
    )
    values = torch.tensor([[[[1.0, 2.0, 3.0]]]])

    assert isinstance(current, PercentileMinMaxNormalize)
    assert isinstance(legacy, Standardize)
    assert isinstance(minmax, MinMaxNormalize)
    assert legacy(values).mean().item() == pytest.approx(0.0, abs=1.0e-7)
    assert legacy(values).std().item() == pytest.approx(1.0)
    assert torch.equal(minmax(values), torch.tensor([[[[0.0, 0.5, 1.0]]]]))
    assert resolve_input_normalization_config(
        {"train": {"normalization": {"scheme": "minmax"}}}
    )["percentiles"] is None

    with pytest.raises(ValueError, match="ordered"):
        PercentileMinMaxNormalize(percentiles=(99.0, 1.0))
    with pytest.raises(ValueError, match="Unexpected"):
        resolve_input_normalization_config(
            {"train": {"normalization": {"unknown": True}}}
        )


def test_grad_scaler_falls_back_for_older_supported_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    calls: list[bool] = []

    def fake_legacy_scaler(*, enabled: bool) -> object:
        calls.append(enabled)
        return sentinel

    monkeypatch.delattr(torch.amp, "GradScaler")
    monkeypatch.setattr(torch.cuda.amp, "GradScaler", fake_legacy_scaler)

    assert train_module._build_grad_scaler("cuda", enabled=True) is sentinel
    assert calls == [True]


def test_torch_compile_is_opt_in_and_keeps_raw_model_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(2, 1)
    compiled_sentinel = object()
    calls: list[tuple[torch.nn.Module, dict[str, Any]]] = []

    def fake_compile(module: torch.nn.Module, **kwargs: Any) -> object:
        calls.append((module, kwargs))
        return compiled_sentinel

    monkeypatch.setattr(torch, "compile", fake_compile)

    assert maybe_compile_model(model, {"enabled": False}) is model
    result = maybe_compile_model(
        model,
        {"enabled": True, "backend": "eager", "dynamic": True},
    )

    assert result is compiled_sentinel
    assert calls == [(model, {"backend": "eager", "dynamic": True})]
    assert set(model.state_dict()) == {"weight", "bias"}


def test_torch_compile_rejects_invalid_or_multiprocess_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="enabled"):
        resolve_torch_compile_config({"enabled": "yes"})
    with pytest.raises(ValueError, match="Unexpected"):
        resolve_torch_compile_config({"enabled": False, "mystery": 1})

    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="single-process"):
        maybe_compile_model(torch.nn.Linear(1, 1), {"enabled": True})


def test_torch_compile_executes_forward_and_backward_outside_autocast() -> None:
    from torch._functorch import config as functorch_config

    model = torch.nn.Linear(2, 1)
    compiled = maybe_compile_model(
        model,
        {"enabled": True, "backend": "eager"},
    )
    inputs = torch.ones(2, 2)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = compiled(inputs).float().square().mean()
    loss.backward()

    assert model.weight.grad is not None
    assert functorch_config.backward_pass_autocast == "off"


def test_cosine_scheduler_horizon_excludes_warmup_epochs(tmp_path: Path) -> None:
    config = _toy_config(tmp_path)
    config["train"]["epochs"] = 6
    config["optim"]["lr_scheduler"] = {
        "_class": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "interval": "epoch",
    }
    config["optim"]["warmup"] = {"num_epochs": 2, "start_factor": 0.1}

    experiment = _ToyTrainExperiment.from_config(
        config, uuid="20260724_150000-WARM-schedule"
    )
    try:
        assert isinstance(
            experiment.lr_scheduler, torch.optim.lr_scheduler.SequentialLR
        )
        cosine = experiment.lr_scheduler._schedulers[1]
        assert cosine.T_max == 4
    finally:
        experiment.close()


def test_ema_policy_validates_update_schedule() -> None:
    policy = resolve_ema_policy(
        {
            "train": {
                "ema": {
                    "enabled": True,
                    "decay": 0.9,
                    "start_after_steps": 3,
                    "update_every": 2,
                }
            }
        }
    )

    assert policy == EmaPolicy(
        enabled=True,
        decay=0.9,
        start_after_steps=3,
        update_every=2,
    )
    with pytest.raises(ValueError, match="0 < decay < 1"):
        resolve_ema_policy({"train": {"ema": {"decay": 1.0}}})
    with pytest.raises(TypeError, match="update_every"):
        resolve_ema_policy({"train": {"ema": {"update_every": 1.5}}})


def test_ema_update_swap_checkpoint_and_resume_round_trip(tmp_path: Path) -> None:
    experiment = _ToyTrainExperiment.from_config(
        _toy_config(tmp_path),
        uuid="20260724_120000-TEST-deadbeef",
    )
    with torch.no_grad():
        experiment.model.weight.fill_(1.0)
    assert experiment.ema.update_after_optimizer_step(1)
    with torch.no_grad():
        experiment.model.weight.fill_(3.0)
    assert experiment.ema.update_after_optimizer_step(2)

    validation_metrics = experiment.run_phase("val", epoch=0)
    callback_values: list[float] = []
    experiment.callbacks = {
        "epoch": [
            lambda **_: callback_values.append(float(experiment.model.weight.item()))
        ]
    }
    experiment.run_callbacks("epoch", epoch=0)
    state = experiment.state

    assert validation_metrics["value"] == pytest.approx(2.0)
    assert callback_values == pytest.approx([2.0])
    assert experiment.model.weight.item() == pytest.approx(3.0)
    assert state["model"]["weight"].item() == pytest.approx(2.0)
    assert state["model_raw"]["weight"].item() == pytest.approx(3.0)
    assert state["_ema_state"]["num_updates"] == 2

    experiment._epoch = 4
    experiment._global_step = 2
    experiment.checkpoint("last")
    resumed = _ToyTrainExperiment(str(experiment.path))
    resumed.load("last")

    assert resumed.model.weight.item() == pytest.approx(3.0)
    assert resumed._epoch == 4
    assert resumed._global_step == 2
    assert resumed.ema.num_updates == 2
    with resumed.ema.use_ema_weights():
        assert resumed.model.weight.item() == pytest.approx(2.0)
    assert resumed.model.weight.item() == pytest.approx(3.0)


def test_ema_respects_burn_in_and_update_interval() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    tracker = ExponentialMovingAverage(
        model,
        EmaPolicy(
            enabled=True,
            decay=0.5,
            start_after_steps=1,
            update_every=2,
        ),
    )

    assert not tracker.update_after_optimizer_step(1)
    assert tracker.update_after_optimizer_step(2)
    assert not tracker.update_after_optimizer_step(3)
    assert tracker.update_after_optimizer_step(4)
    assert tracker.num_updates == 2


def test_ema_checkpoint_shadows_follow_the_live_model_device() -> None:
    model = torch.nn.Linear(2, 1, device="meta")
    ema = ExponentialMovingAverage(
        model,
        EmaPolicy(enabled=True, decay=0.9, start_after_steps=0, update_every=1),
    )
    deployment_state = {
        name: torch.zeros(value.shape, dtype=value.dtype)
        for name, value in model.state_dict().items()
    }

    ema.load_checkpoint_state(
        deployment_state,
        {
            "version": ema.STATE_VERSION,
            "initialized": True,
            "num_updates": 1,
            "decay": 0.9,
            "start_after_steps": 0,
            "update_every": 1,
        },
    )

    assert {value.device.type for value in ema._shadow_state.values()} == {"meta"}


def test_checkpoint_round_trips_rng_state_and_commits_properties(
    tmp_path: Path,
) -> None:
    experiment = _ToyTrainExperiment.from_config(
        _toy_config(tmp_path),
        uuid="20260724_130000-RNGS-deadbeef",
    )

    random.seed(123)
    np.random.seed(456)
    torch.manual_seed(789)
    experiment._epoch = 5
    experiment.checkpoint("last")

    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    resumed = _ToyTrainExperiment(str(experiment.path))
    requested_train_epochs: list[int] = []
    resumed.train_dl = SimpleNamespace(set_epoch=requested_train_epochs.append)
    resumed.load("last")
    assert requested_train_epochs == [6]
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert resumed.properties["epoch"] == 5
    assert not list((experiment.path / "checkpoints").glob(".*.tmp"))


def test_checkpoint_identity_fields_are_persisted_and_legacy_state_loads(
    tmp_path: Path,
) -> None:
    experiment = _ToyTrainExperiment.from_config(
        _toy_config(tmp_path),
        uuid="20260724_131000-IDEN-deadbeef",
    )
    experiment.checkpoint("last")
    checkpoint_path = experiment.path / "checkpoints" / "last.pt"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert state["run_id"] == experiment.path.name
    assert state["config_digest"] == config_digest(experiment.config)

    state.pop("run_id")
    state.pop("config_digest")
    torch.save(state, checkpoint_path)
    resumed = _ToyTrainExperiment(str(experiment.path))
    assert resumed.load("last") is resumed


@pytest.mark.parametrize(
    ("field", "wrong_value", "message"),
    [
        ("run_id", "another-run-id", "Checkpoint run_id"),
        ("config_digest", "wrong", "Checkpoint config_digest"),
    ],
)
def test_checkpoint_identity_mismatch_is_rejected_before_restore(
    tmp_path: Path, field: str, wrong_value: str, message: str
) -> None:
    experiment = _ToyTrainExperiment.from_config(
        _toy_config(tmp_path),
        uuid="20260724_132000-IDEN-deadbeef",
    )
    experiment._epoch = 3
    experiment.checkpoint("last")
    checkpoint_path = experiment.path / "checkpoints" / "last.pt"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state[field] = wrong_value
    torch.save(state, checkpoint_path)

    resumed = _ToyTrainExperiment(str(experiment.path))
    with pytest.raises(ValueError, match=message):
        resumed.load("last")
    assert resumed._epoch == -1


def test_epoch_callback_rng_is_committed_before_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _toy_config(tmp_path)
    config["log"]["model_weights"]["save_freq"] = 1
    experiment = _ToyTrainExperiment.from_config(
        config,
        uuid="20260724_135000-CBRN-deadbeef",
    )

    monkeypatch.setattr(experiment, "_init_wandb", lambda: None)
    monkeypatch.setattr(
        train_module.wandb, "log", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(train_module.wandb, "finish", lambda: None)

    callback_values: list[float] = []
    experiment.callbacks = {
        "epoch": [
            lambda **_: callback_values.append(random.random())
        ]
    }
    random.seed(321)
    expected_callback = random.random()
    expected_after_epoch = random.random()
    random.seed(321)

    experiment.run()

    assert callback_values == [expected_callback]
    assert (experiment.path / "checkpoints" / "epoch_0000.pt").is_file()
    assert (experiment.path / "checkpoints" / "last.pt").is_file()
    random.seed(999)
    resumed = _ToyTrainExperiment(str(experiment.path))
    resumed.load("last")
    assert random.random() == expected_after_epoch

def test_run_closes_owned_resources_when_wandb_init_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    experiment = _ToyTrainExperiment.from_config(_toy_config(tmp_path))
    close_calls: list[str] = []
    database = SimpleNamespace(close=lambda: close_calls.append("database"))
    shared_dataset = SimpleNamespace(backend=SimpleNamespace(db=database))
    experiment.dataset_bundle = TrainingDatasetBundle(
        train={"source": shared_dataset},
        val={"source": shared_dataset},
        modalities={"source": "xray"},
        _databases=(database,),
    )

    def fail_wandb_init() -> None:
        """Raise the simulated logger initialization failure."""

        raise RuntimeError("wandb unavailable")

    monkeypatch.setattr(experiment, "_init_wandb", fail_wandb_init)
    monkeypatch.setattr(
        train_module.wandb, "finish", lambda: close_calls.append("wandb")
    )

    with pytest.raises(RuntimeError, match="wandb unavailable"):
        experiment.run()
    experiment.close()

    assert close_calls == ["wandb", "database"]


def test_failed_checkpoint_does_not_advance_properties(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _ToyTrainExperiment.from_config(
        _toy_config(tmp_path), uuid="20260724_140000-FAIL-deadbeef"
    )
    experiment._epoch = 7

    def fail_save(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        experiment.checkpoint("last")


def test_wandb_identity_is_persisted_and_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _ToyTrainExperiment.from_config(
        _toy_config(tmp_path), uuid="20260724_150000-WANB-deadbeef"
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(train_module, "_generate_wandb_id", lambda: "stable-id")
    monkeypatch.setattr(
        train_module.wandb, "init", lambda **kwargs: calls.append(kwargs)
    )

    experiment._init_wandb()
    first_id = experiment.properties["wandb.id"]
    experiment._init_wandb()
    second_id = experiment.properties["wandb.id"]

    assert first_id == second_id == "stable-id"
    assert [call["id"] for call in calls] == ["stable-id", "stable-id"]


def test_packaged_base_matches_released_recipe() -> None:
    """The packaged base recipe must reproduce the released FleXray training run."""
    import yaml

    from fxr.launch.cli import load_training_base
    from fxr.models import UNet
    from fxr.protocols import load_protocol_by_name, resolve_run_output_label_names

    base = load_training_base("base")
    protocol = load_protocol_by_name("all_structures_flexray_v4")
    expected_model_labels = tuple(
        label
        for label in protocol.labels
        if label not in {"lumbar_spine", "thoracolumbar_spine"}
    )
    assert resolve_run_output_label_names(base) == expected_model_labels
    assert len(expected_model_labels) == 61

    model_cfg = {k: v for k, v in base["model"].items() if k not in {"_class", "compile_cfg"}}
    assert model_cfg["convs_per_block"] == 3
    assert all(model_cfg[flag] for flag in (
        "norm_after_activation", "upsample_align_corners", "residual_shortcut_norm"
    ))
    bundle = Path(__file__).resolve().parents[1] / "exports"
    released = sorted(bundle.glob("flexray-base-*/config.yml"))
    if released:
        released_cfg = yaml.safe_load(released[-1].read_text())["model"]
        released_cfg.pop("_class")
        expected = UNet(**released_cfg).state_dict()
        actual = UNet(**model_cfg, out_channels=released_cfg["out_channels"]).state_dict()
        assert {k: tuple(v.shape) for k, v in actual.items()} == {
            k: tuple(v.shape) for k, v in expected.items()
        }

    attenuation = base["drr_model"]["default"]["attenuation_cfg"]
    assert attenuation == {
        "prob": 0.5,
        "range": [0.1, 4.0],
        "scope": "per_label",
        "distribution": {"type": "lognormal", "mode": 1.0, "sigma": 1.0},
    }
    moose = base["drr_model"]["datasets"]["MOOSE"]
    assert moose["intrinsics_cfg"]["delx"] == [1.25, 2.25]
    assert moose["intrinsics_cfg"]["projection_mode"] == {"cone": 0.9, "orthographic": 0.1}
    assert moose["sample_params_cfg"]["xyz_range"]["y"] == [500, 900]
    assert base["loss_func"]["dataset_losses"]["FluXray"] == "standard_seg"
    assert base["augmentation"] == {"presets": {"CT": "CT_base", "Xray": "Xray_base"}}
    weighting = {"scheme": "inverse_label_frequency", "tau": 0.5, "class_aggregation": "max"}
    for name in ("MOOSE", "ElbowCT", "RSNAFrac", "PedsCT"):
        assert base["data"]["CT"][name]["sample_weighting"] == weighting
    assert "sample_weighting" not in base["data"]["CT"]["HANSeg"]
    eval_sets = base["callbacks"]["epoch"]["eval_sets"]
    assert len(eval_sets["data"]["Xray"]) == 12
    assert "JRST" not in eval_sets["data"]["Xray"]
    assert eval_sets["min_ground_truth_area_fraction"] == 0.001
