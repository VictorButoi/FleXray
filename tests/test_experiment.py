"""End-to-end tests for the X-ray segmentation training experiment."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

import wandb

from fxr.augmentation import Standardize, build_segmentation_augmentation_pipeline
from fxr.config import Config
from fxr.experiment import (
    EvalSetMetricLogger,
    FleXrayTrainExperiment,
    WandbSamplePredictionLogger,
    resolve_batch_inputs,
)
from fxr.experiment.batch_inputs import BatchInputs
from fxr.experiment.drr_forward import DrrForwardPipeline
from fxr.models.camera import CTDRRRenderResult
from fxr.protocols import load_protocol_by_name, resolve_run_output_label_names

_HEIGHT = 32
_WIDTH = 32


class _SyntheticXray(Dataset):
    """In-memory X-ray dataset emitting HipRay-native integer labels."""

    def __init__(self, size: int, seed: int = 0) -> None:
        self.size = size
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(size, 1, _HEIGHT, _WIDTH, generator=generator)
        self.labels = torch.randint(0, 3, (size, _HEIGHT, _WIDTH), generator=generator)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        return {
            "image": self.images[index],
            "label": self.labels[index],
            "dataset_name": "HipRay",
            "modality": "xray",
        }


class _SyntheticSegExperiment(FleXrayTrainExperiment):
    """Segmentation experiment backed by synthetic in-memory X-ray data."""

    def build_data(self, load_data: bool) -> None:
        if not load_data:
            return
        self.train_datasets = {"HipRay": _SyntheticXray(size=4, seed=1)}
        self.val_datasets = {"HipRay": _SyntheticXray(size=2, seed=2)}
        self.modalities = {"HipRay": "xray"}
        self.build_dataloader()


def _config(root) -> dict:
    return {
        "experiment": {"seed": 0},
        "train": {"epochs": 2, "eval_freq": 1},
        "protocol": {"name": "all_structures_flexray_v4"},
        # Synthetic 32x32 images: the Xray_base aspect crop needs >= 224 px.
        "augmentation": {"presets": {"Xray": "base_light"}},
        "model": {"_class": "fxr.models.UNet", "in_channels": 1, "filters": [8, 16]},
        "loss_func": {"_class": "SoftDiceLoss", "from_logits": True, "mode": "onehot"},
        "optim": {"_class": "torch.optim.AdamW", "lr": 1e-3},
        "dataloader": {"batch_size": 2, "num_workers": 0},
        "data": {"Xray": {"HipRay": {}}},
        "log": {
            "root": str(root),
            "wandb": {"mode": "disabled"},
            "model_weights": {"save_freq": 0},
        },
    }


class _TrainOnlySyntheticSegExperiment(FleXrayTrainExperiment):
    """Synthetic segmentation experiment with no validation datasets."""

    def build_data(self, load_data: bool) -> None:
        """Build only a synthetic training dataset.

        Args:
            load_data: Whether to construct the synthetic training source.

        Returns:
            ``None``.
        """

        if not load_data:
            return
        self.train_datasets = {"HipRay": _SyntheticXray(size=2, seed=1)}
        self.val_datasets = {}
        self.modalities = {"HipRay": "xray"}
        self._channel_source_names = {}
        self.build_dataloader()


def test_train_only_experiment_runs_when_core_validation_is_disabled(tmp_path):
    config = _config(tmp_path)
    config["train"].update({"epochs": 1, "eval_freq": 0})
    experiment = _TrainOnlySyntheticSegExperiment.from_config(config)

    assert experiment.val_datasets == {}
    assert experiment.val_dl is None

    experiment.run()

    assert (experiment.path / "checkpoints" / "last.pt").is_file()


def test_train_runs_and_checkpoints(tmp_path):
    experiment = _SyntheticSegExperiment.from_config(_config(tmp_path))
    labels = resolve_run_output_label_names(_config(tmp_path))

    # out_channels is derived from the protocol's default model label space.
    assert experiment.model.output_conv.out_channels == len(labels) == 61
    experiment.run()

    assert (experiment.path / "checkpoints" / "last.pt").exists()
    assert (experiment.path / "config.yml").exists()
    assert experiment._epoch == 1


def test_step_loss_is_finite(tmp_path):
    experiment = _SyntheticSegExperiment.from_config(_config(tmp_path))
    batch = next(iter(experiment.train_dl))

    outputs = experiment.run_step(batch, phase="train", backward=True)
    metrics = experiment.compute_metrics(outputs)

    assert torch.isfinite(outputs["loss"]).all()
    assert metrics["loss"] == pytest.approx(metrics["loss"])  # not NaN
    assert 0.0 <= metrics["dice"] <= 1.0


def test_skipped_scaled_step_does_not_advance_training_state(tmp_path):
    experiment = _SyntheticSegExperiment.from_config(_config(tmp_path))
    batch = next(iter(experiment.train_dl))
    parameters_before = {
        name: value.detach().clone()
        for name, value in experiment.model.state_dict().items()
    }

    class _SkippingScaler:
        """Minimal scaler double that reports an overflow-skipped update."""

        def __init__(self):
            """Start above the post-overflow scale."""
            self.current_scale = 2.0

        def get_scale(self):
            """Return the current scale."""
            return self.current_scale

        def scale(self, loss):
            """Return an unmodified loss for backward."""
            return loss

        def step(self, optimizer):
            """Skip the optimizer update."""

        def update(self):
            """Lower the scale to signal an overflow."""
            self.current_scale = 1.0

    experiment.grad_scaler = _SkippingScaler()
    experiment.run_step(batch, phase="train", backward=True)

    assert experiment._global_step == 0
    assert experiment.ema.num_updates == 0
    assert all(
        torch.equal(value, parameters_before[name])
        for name, value in experiment.model.state_dict().items()
    )


def test_run_phase_uses_sample_weighting_for_uneven_batches(tmp_path):
    experiment = _SyntheticSegExperiment.from_config(_config(tmp_path))
    experiment.val_dl = ["large", "small"]

    def run_step(batch, *, phase: str, backward: bool) -> dict:
        if batch == "large":
            return {
                "y_pred": torch.zeros(3, 1, 1, 1),
                "dataset_name": "HipRay",
                "dice_value": 1.0,
                "loss_value": 2.0,
            }
        return {
            "y_pred": torch.zeros(1, 1, 1, 1),
            "dataset_name": "HipRay",
            "dice_value": 0.0,
            "loss_value": 6.0,
        }

    def compute_metrics(outputs: dict) -> dict[str, float]:
        return {"dice": outputs["dice_value"], "loss": outputs["loss_value"]}

    experiment.run_step = run_step
    experiment.compute_metrics = compute_metrics

    metrics = experiment.run_phase("val", 0)

    assert metrics["dice"] == pytest.approx(0.75)
    assert metrics["loss"] == pytest.approx(3.0)
    assert metrics["HipRay/dice"] == pytest.approx(0.75)
    assert metrics["HipRay/loss"] == pytest.approx(3.0)


def test_checkpoint_resume_round_trip(tmp_path):
    experiment = _SyntheticSegExperiment.from_config(_config(tmp_path))
    experiment.run()
    trained_state = {k: v.clone() for k, v in experiment.model.state_dict().items()}

    resumed = _SyntheticSegExperiment(str(experiment.path))
    # Freshly-built weights differ before loading.
    assert not _states_match(resumed.model.state_dict(), trained_state)

    resumed.load("last")
    assert _states_match(resumed.model.state_dict(), trained_state)
    assert resumed._epoch == 1


def test_wandb_sample_logger_produces_overlays(tmp_path):
    experiment = _SyntheticSegExperiment.from_config(_config(tmp_path))
    logger = WandbSamplePredictionLogger(experiment, every=1, max_samples=2)

    wandb.init(mode="disabled")
    try:
        # Inspect the overlay payload directly rather than the no-op wandb.log.
        outputs = logger._predict_one_batch("HipRay", experiment.val_datasets["HipRay"])
        overlays = logger._overlay_images(outputs)
        logger(epoch=0)  # full pass must not raise
    finally:
        wandb.finish()

    assert len(overlays) == 2
    assert all(isinstance(image, wandb.Image) for image in overlays)


def test_wandb_overlays_preserve_overlapping_foreground_channels(monkeypatch):
    captured: list[dict] = []

    def fake_image(image, *, masks):
        captured.append({"image": image, "masks": masks})
        return captured[-1]

    monkeypatch.setattr("fxr.experiment.callbacks.wandb.Image", fake_image)
    experiment = SimpleNamespace(
        model_label_names=("background", "bone_a", "bone_b")
    )
    logger = WandbSamplePredictionLogger(experiment, max_samples=1)
    outputs = {
        "x": torch.zeros(1, 1, 2, 2),
        "y_pred": torch.tensor(
            [
                [
                    [[-10.0, -10.0], [-10.0, -10.0]],
                    [[10.0, -10.0], [-10.0, -10.0]],
                    [[10.0, -10.0], [-10.0, -10.0]],
                ]
            ]
        ),
        "y_true": torch.tensor(
            [
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[1.0, 0.0], [0.0, 0.0]],
                ]
            ]
        ),
    }

    overlays = logger._overlay_images(outputs)

    assert overlays == captured
    assert set(captured[0]["masks"]) == {
        "prediction/bone_a",
        "ground_truth/bone_a",
        "prediction/bone_b",
        "ground_truth/bone_b",
    }
    assert captured[0]["masks"]["prediction/bone_a"]["mask_data"][0, 0] == 1
    assert captured[0]["masks"]["prediction/bone_b"]["mask_data"][0, 0] == 1
    assert captured[0]["masks"]["ground_truth/bone_a"]["mask_data"][0, 0] == 1
    assert captured[0]["masks"]["ground_truth/bone_b"]["mask_data"][0, 0] == 1


class _VariableMetadataXray(_SyntheticXray):
    """Synthetic X-ray dataset with non-stackable metadata."""

    def __getitem__(self, index: int) -> dict:
        """Return one sample with variable-length metadata."""
        sample = super().__getitem__(index)
        sample["metadata"] = {"variable": list(range(index + 1))}
        return sample


class _NamedDenseXray(_SyntheticXray):
    """Dense-map X-rays with legacy ordered native-label metadata."""

    def __init__(self, size: int, seed: int = 0) -> None:
        """Build samples and expose names ordered by native dense id."""
        super().__init__(size=size, seed=seed)
        self.backend = SimpleNamespace(
            label_names=("background", "femurs", "hips")
        )


class _NamedDenseSegExperiment(FleXrayTrainExperiment):
    """Synthetic experiment reproducing a dense schema with ordered native names."""

    def build_data(self, load_data: bool) -> None:
        """Build dense named train and validation datasets."""
        if not load_data:
            return
        self.train_datasets = {"HipRay": _NamedDenseXray(size=2, seed=1)}
        self.val_datasets = {"HipRay": _NamedDenseXray(size=2, seed=2)}
        self.modalities = {"HipRay": "xray"}
        self._channel_source_names = self._collect_channel_source_names()
        self.build_dataloader()


def test_named_dense_xray_metadata_does_not_select_channel_projection(tmp_path):
    experiment = _NamedDenseSegExperiment.from_config(_config(tmp_path))
    batch = next(iter(experiment.train_dl))

    outputs = experiment.run_step(batch, phase="val", backward=False)

    assert outputs["y_true"].ndim == 4
    assert outputs["y_true"].shape[1] == len(
        resolve_run_output_label_names(_config(tmp_path))
    )


class _VariableMetadataSegExperiment(FleXrayTrainExperiment):
    """Segmentation experiment whose validation samples carry variable metadata."""

    def build_data(self, load_data: bool) -> None:
        """Build synthetic train and validation datasets."""
        if not load_data:
            return
        self.train_datasets = {"HipRay": _VariableMetadataXray(size=2, seed=1)}
        self.val_datasets = {"HipRay": _VariableMetadataXray(size=2, seed=2)}
        self.build_dataloader()


def test_wandb_sample_logger_handles_variable_metadata(tmp_path):
    experiment = _VariableMetadataSegExperiment.from_config(_config(tmp_path))
    logger = WandbSamplePredictionLogger(
        experiment, every=1, max_samples=1, batch_size=2
    )

    outputs = logger._predict_one_batch("HipRay", experiment.val_datasets["HipRay"])

    assert outputs["dataset_name"] == "HipRay"
    assert outputs["y_pred"].shape[0] == 2


def test_evalset_metric_logger_handles_variable_metadata(tmp_path):
    experiment = _VariableMetadataSegExperiment.from_config(_config(tmp_path))
    logger = EvalSetMetricLogger(
        experiment, data={"Xray": {"HipRay": {}}}, every=1, batch_size=2
    )

    dice = logger._evaluate_dataset("HipRay", experiment.val_datasets["HipRay"])

    assert 0.0 <= dice <= 1.0


def test_evalset_metric_logger_bypasses_training_loss_routing(tmp_path):
    experiment = _SyntheticSegExperiment.from_config(_config(tmp_path))

    def reject_loss_routing(*args, **kwargs):
        raise AssertionError("eval-only metrics must not invoke the training loss")

    experiment.loss_func.forward = reject_loss_routing
    logger = EvalSetMetricLogger(
        experiment,
        data={"Xray": {"MetricsOnly": {}}},
        every=1,
        batch_size=2,
    )

    dice = logger._evaluate_dataset("HipRay", experiment.val_datasets["HipRay"])

    assert 0.0 <= dice <= 1.0


def test_wandb_sample_logger_bypasses_training_loss_and_step_callbacks(tmp_path):
    experiment = _SyntheticSegExperiment.from_config(_config(tmp_path))

    def reject_loss_routing(*args, **kwargs):
        raise AssertionError("sample logging must not invoke the training loss")

    experiment.loss_func.forward = reject_loss_routing
    experiment.callbacks = {
        "step": [
            lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("sample logging must not dispatch step callbacks")
            )
        ]
    }
    logger = WandbSamplePredictionLogger(experiment, batch_size=2)

    outputs = logger._predict_one_batch(
        "HipRay", experiment.val_datasets["HipRay"]
    )

    assert outputs["dataset_name"] == "HipRay"
    assert "loss" not in outputs


def test_multilabel_metrics_preserve_overlapping_foreground_channels(tmp_path):
    experiment = _SyntheticSegExperiment.from_config(_config(tmp_path))
    outputs = {
        "loss": torch.tensor(0.0),
        "y_pred": torch.tensor([[[[-10.0]], [[10.0]], [[10.0]]]]),
        "y_true": torch.tensor([[[[0.0]], [[1.0]], [[1.0]]]]),
    }

    metrics = experiment.compute_metrics(outputs)

    assert metrics["dice"] == pytest.approx(1.0)


def test_evalset_metric_logger_evaluates_two_xray_datasets(tmp_path, monkeypatch):
    experiment = _SyntheticSegExperiment.from_config(_config(tmp_path))
    built: list[tuple[dict, str, str]] = []

    def fake_build_named_datasets(data_cfg, split: str, modality: str) -> dict:
        built.append((data_cfg, split, modality))
        return {
            "HipRay": _SyntheticXray(size=3, seed=5),
            "JRST": _SyntheticXray(size=5, seed=6),
        }

    monkeypatch.setattr(
        "fxr.experiment.callbacks.build_named_datasets", fake_build_named_datasets
    )
    logger = EvalSetMetricLogger(
        experiment,
        data={"Xray": {"HipRay": {}, "JRST": {}}},
        every=2,
        batch_size=2,
    )

    wandb.init(mode="disabled")
    try:
        logger(epoch=1)
        assert logger.datasets is None
        dice = logger._evaluate_dataset("HipRay", _SyntheticXray(size=3, seed=7))
        logger(epoch=0)
    finally:
        wandb.finish()

    assert 0.0 <= dice <= 1.0
    assert built == [({"Xray": {"HipRay": {}, "JRST": {}}}, "val", "Xray")]
    assert set(logger.datasets or {}) == {"HipRay", "JRST"}


def test_experiment_close_releases_lazy_evalset_readers_once(tmp_path, monkeypatch):
    experiment = _SyntheticSegExperiment.from_config(_config(tmp_path))
    close_calls = {"HipRay": 0, "JRST": 0}

    def fake_build_named_datasets(data_cfg, split: str, modality: str) -> dict:
        """Return closeable callback-local dataset stand-ins.

        Args:
            data_cfg: Callback dataset config.
            split: Requested eval split.
            modality: Requested runtime modality.

        Returns:
            Mapping of closeable dataset stand-ins.
        """

        del data_cfg, split, modality
        datasets = {}
        for name in close_calls:
            def close(dataset_name=name):
                close_calls[dataset_name] += 1

            datasets[name] = SimpleNamespace(close=close)
        return datasets

    monkeypatch.setattr(
        "fxr.experiment.callbacks.build_named_datasets", fake_build_named_datasets
    )
    logger = EvalSetMetricLogger(
        experiment,
        data={"Xray": {"HipRay": {}, "JRST": {}}},
    )
    assert set(logger._datasets()) == {"HipRay", "JRST"}
    experiment.callbacks = {"epoch": [logger]}

    experiment.close()
    experiment.close()

    assert close_calls == {"HipRay": 1, "JRST": 1}
    assert logger.datasets == {}


def test_evalset_metric_logger_empty_xray_config_noops(tmp_path):
    experiment = _SyntheticSegExperiment.from_config(_config(tmp_path))
    logger = EvalSetMetricLogger(experiment, data={"Xray": {}}, every=1)

    wandb.init(mode="disabled")
    try:
        logger(epoch=0)
    finally:
        wandb.finish()

    assert logger.datasets == {}


def test_evalset_metric_logger_rejects_ct_config():
    with pytest.raises(NotImplementedError, match="CT eval-set metrics are deferred"):
        EvalSetMetricLogger(object(), data={"CT": {"ToyCT": {}}})


def test_callbacks_are_built_from_config(tmp_path):
    config = _config(tmp_path)
    config["callbacks"] = {
        "epoch": {
            "samples": {
                "_class": "fxr.experiment.WandbSamplePredictionLogger",
                "every": 1,
            },
            "eval_sets": {
                "_class": "fxr.experiment.EvalSetMetricLogger",
                "every": 2,
                "data": {"Xray": {}},
            },
        }
    }
    experiment = _SyntheticSegExperiment.from_config(config)

    assert len(experiment.callbacks["epoch"]) == 2
    assert isinstance(experiment.callbacks["epoch"][0], WandbSamplePredictionLogger)
    assert isinstance(experiment.callbacks["epoch"][1], EvalSetMetricLogger)


def test_ct_batch_inputs_resolve_affine_and_centroids():
    batch = {
        "image": torch.zeros(1, 1, 6, 6, 6),
        "label": torch.zeros(1, 1, 6, 6, 6),
        "modality": ["ct"],
        "metadata": {
            "affine": torch.eye(4).unsqueeze(0),
            "fg_centroids_ijk": torch.zeros(1, 2, 3),
        },
    }
    inputs = resolve_batch_inputs(batch, modality="ct")

    assert inputs.modality == "ct"
    assert inputs.affine.shape == (4, 4)
    assert inputs.fg_centroids_ijk.shape == (2, 3)


def test_ct_batch_requires_affine_metadata():
    with pytest.raises(KeyError, match="metadata.affine"):
        resolve_batch_inputs(
            {"image": torch.zeros(1, 1, 4, 4, 4), "label": torch.zeros(1, 1, 4, 4, 4)},
            modality="ct",
        )


_CT_HEIGHT = 8
_CT_DEPTH = 12


class _SyntheticCT(Dataset):
    """In-memory CT dataset emitting model-id labels and an affine."""

    def __init__(self, size: int, seed: int = 0) -> None:
        self.size = size
        generator = torch.Generator().manual_seed(seed)
        shape = (size, 1, _CT_DEPTH, _CT_HEIGHT, _CT_HEIGHT)
        self.images = torch.rand(shape, generator=generator) * 400 - 150
        self.labels = torch.randint(0, 4, shape, generator=generator)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        return {
            "image": self.images[index],
            "label": self.labels[index],
            "dataset_name": "RSNAFrac",
            "modality": "ct",
            "metadata": {
                "affine": torch.eye(4),
                "spacing": torch.ones(3),
                "optional_note": None,
                "preprocessing": {"steps": ["window", index]},
            },
        }


class _SyntheticCTExperiment(FleXrayTrainExperiment):
    """Segmentation experiment backed by synthetic in-memory CT volumes."""

    def build_augmentations(self) -> None:
        """Build no-op augmentation for tiny synthetic CT render tests.

        Returns:
            ``None``.
        """
        self.normalizer = Standardize()
        self.aug_pipelines = {"ct": build_segmentation_augmentation_pipeline({})}

    def build_data(self, load_data: bool) -> None:
        if not load_data:
            return
        self.train_datasets = {"RSNAFrac": _SyntheticCT(size=2, seed=1)}
        self.val_datasets = {"RSNAFrac": _SyntheticCT(size=1, seed=2)}
        self.modalities = {"RSNAFrac": "ct"}
        self.drr_pipeline = DrrForwardPipeline.from_config(
            self.config.to_dict(), device=self.device
        )
        self.build_dataloader()


def _ct_config(root) -> dict:
    config = _config(root)
    config["dataloader"] = {"batch_size": 2, "num_workers": 0}
    config["data"] = {"CT": {"RSNAFrac": {}}}
    config["drr_model"] = {
        "default": {
            "preset": "frontal",
            "num_views": 2,
            "intrinsics_cfg": {
                "height": _CT_HEIGHT,
                "width": _CT_HEIGHT,
                "sdd": 1000.0,
                "delx": 2.0,
            },
            "isocenter_cfg": {"sample_scheme": "volume_center"},
            "seg_cfg": {"soft_labels": False, "threshold": 0.0},
        },
        "datasets": {"RSNAFrac": {}},
    }
    return config


def test_ct_training_step_renders_and_is_finite(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    experiment = _SyntheticCTExperiment.from_config(_ct_config(tmp_path))
    batch = next(iter(experiment.train_dl))

    outputs = experiment.run_step(batch, phase="train", backward=True)
    metrics = experiment.compute_metrics(outputs)

    # One CT volume is rendered into num_views DRRs forming the model batch.
    assert outputs["x"].shape == (2, 1, _CT_HEIGHT, _CT_HEIGHT)
    assert outputs["y_pred"].shape[0] == 2
    assert torch.isfinite(outputs["loss"]).all()
    assert 0.0 <= metrics["dice"] <= 1.0


def test_wandb_sample_logger_uses_ct_safe_collation(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    experiment = _SyntheticCTExperiment.from_config(_ct_config(tmp_path))
    logger = WandbSamplePredictionLogger(experiment, max_samples=1)

    outputs = logger._predict_one_batch(
        "RSNAFrac", experiment.val_datasets["RSNAFrac"]
    )

    assert outputs["dataset_name"] == "RSNAFrac"
    assert outputs["x"].shape == (2, 1, _CT_HEIGHT, _CT_HEIGHT)


def test_ct_model_inputs_pass_native_projection_metadata(tmp_path):
    protocol = load_protocol_by_name("all_structures_flexray_v4")
    experiment = object.__new__(FleXrayTrainExperiment)
    experiment.config = Config(_ct_config(tmp_path))
    experiment.model_label_names = tuple(protocol.labels)
    experiment._projection_cache = {}
    captured = {}

    class FakePipeline:
        """DRR pipeline test double that records render keyword arguments."""

        def render(self, **kwargs):
            """Record render inputs and return empty rendered tensors."""
            captured.update(kwargs)
            return CTDRRRenderResult(
                images=torch.zeros(2, 1, _CT_HEIGHT, _CT_HEIGHT),
                labels=torch.zeros(2, len(protocol.labels), _CT_HEIGHT, _CT_HEIGHT),
            )

    experiment.drr_pipeline = FakePipeline()
    label = torch.zeros(1, 1, 2, 2, 2)
    label[..., 0, 0, 0] = 1
    label[..., 1, 1, 1] = 15
    inputs = BatchInputs(
        modality="ct",
        image=torch.zeros(1, 1, 2, 2, 2),
        label=label,
        affine=torch.eye(4),
        fg_centroids_ijk=None,
    )

    experiment._model_inputs(inputs, dataset_name="RSNAFrac")

    collapse_map = captured["foreground_collapse_map"]
    assert collapse_map[0] == protocol.label_to_id["vertebra_c1"]
    assert collapse_map[14] == protocol.label_to_id["skull"]
    assert captured["attenuated_label_ids"] == tuple(range(1, 16))
    assert captured["label"].shape == (1, 2, 2, 2)
    assert int(captured["label"].max()) == 15


def test_ct_model_inputs_scale_rendered_images_per_view(tmp_path):
    protocol = load_protocol_by_name("all_structures_flexray_v4")
    experiment = object.__new__(FleXrayTrainExperiment)
    experiment.config = Config(_ct_config(tmp_path))
    experiment.model_label_names = tuple(protocol.labels)
    experiment._projection_cache = {}

    first_view = torch.linspace(2.0, 8.0, _CT_HEIGHT * _CT_HEIGHT).reshape(
        1, _CT_HEIGHT, _CT_HEIGHT
    )
    second_view = torch.full((1, _CT_HEIGHT, _CT_HEIGHT), 5.0)

    class FakePipeline:
        """DRR pipeline test double returning unnormalized rendered images."""

        def render(self, **kwargs):
            """Return rendered images whose raw range is outside [0, 1]."""
            return CTDRRRenderResult(
                images=torch.stack([first_view, second_view]),
                labels=torch.zeros(2, len(protocol.labels), _CT_HEIGHT, _CT_HEIGHT),
            )

    experiment.drr_pipeline = FakePipeline()
    inputs = BatchInputs(
        modality="ct",
        image=torch.zeros(1, 1, 2, 2, 2),
        label=torch.zeros(1, 1, 2, 2, 2),
        affine=torch.eye(4),
        fg_centroids_ijk=None,
    )

    image, _ = experiment._model_inputs(inputs, dataset_name="RSNAFrac")

    assert torch.isfinite(image).all()
    assert image[0].amin() == pytest.approx(0.0)
    assert image[0].amax() == pytest.approx(1.0)
    assert torch.count_nonzero(image[1]) == 0


def test_ct_scaled_rendered_images_survive_intensity_augmentations(tmp_path):
    protocol = load_protocol_by_name("all_structures_flexray_v4")
    experiment = object.__new__(FleXrayTrainExperiment)
    experiment.config = Config(_ct_config(tmp_path))
    experiment.model_label_names = tuple(protocol.labels)
    experiment._projection_cache = {}
    experiment.normalizer = Standardize()
    experiment.modalities = {"RSNAFrac": "ct"}
    experiment.aug_pipelines = {"ct": build_segmentation_augmentation_pipeline(
        {
            "RandomInvert": {"module": "kornia.augmentation", "p": 1.0},
            "RandomGamma": {
                "module": "kornia.augmentation",
                "gamma": [0.7, 0.7],
                "gain": [1.0, 1.0],
                "p": 1.0,
            },
        }
    )}

    raw_image = torch.linspace(0.0, 7.0, _CT_HEIGHT * _CT_HEIGHT).reshape(
        1, 1, _CT_HEIGHT, _CT_HEIGHT
    )

    class FakePipeline:
        """DRR pipeline test double returning high-range rendered images."""

        def render(self, **kwargs):
            """Return a finite CT render whose raw range exceeds [0, 1]."""
            return CTDRRRenderResult(
                images=raw_image,
                labels=torch.zeros(1, len(protocol.labels), _CT_HEIGHT, _CT_HEIGHT),
            )

    experiment.drr_pipeline = FakePipeline()
    inputs = BatchInputs(
        modality="ct",
        image=torch.zeros(1, 1, 2, 2, 2),
        label=torch.zeros(1, 1, 2, 2, 2),
        affine=torch.eye(4),
        fg_centroids_ijk=None,
    )

    image, label = experiment._model_inputs(inputs, dataset_name="RSNAFrac")
    augmented_image, _ = experiment._apply_augmentation(
        image, label, "train", "RSNAFrac"
    )
    normalized = experiment.normalizer(augmented_image)

    assert torch.isfinite(augmented_image).all()
    assert torch.isfinite(normalized).all()


def test_ct_model_inputs_reject_unknown_native_id_before_render(tmp_path):
    protocol = load_protocol_by_name("all_structures_flexray_v4")
    experiment = object.__new__(FleXrayTrainExperiment)
    experiment.config = Config(_ct_config(tmp_path))
    experiment.model_label_names = tuple(protocol.labels)
    experiment._projection_cache = {}

    class FailingPipeline:
        """DRR pipeline test double that fails if rendering is reached."""

        def render(self, **kwargs):
            """Raise because validation should happen before rendering."""
            raise AssertionError("render should not be called")

    experiment.drr_pipeline = FailingPipeline()
    inputs = BatchInputs(
        modality="ct",
        image=torch.zeros(1, 1, 2, 2, 2),
        label=torch.full((1, 1, 2, 2, 2), 99),
        affine=torch.eye(4),
        fg_centroids_ijk=None,
    )

    with pytest.raises(ValueError, match="unknown native label id"):
        experiment._model_inputs(inputs, dataset_name="RSNAFrac")


class _TwoSourceSegExperiment(FleXrayTrainExperiment):
    """Segmentation experiment backed by two synthetic X-ray sources."""

    def build_data(self, load_data: bool) -> None:
        if not load_data:
            return
        self.train_datasets = {
            "HipRay": _SyntheticXray(size=4, seed=1),
            "JRST": _SyntheticXray(size=4, seed=2),
        }
        self.val_datasets = {
            "HipRay": _SyntheticXray(size=2, seed=3),
            "JRST": _SyntheticXray(size=2, seed=4),
        }
        self.build_dataloader()


def test_run_phase_reports_overall_and_per_dataset_metrics(tmp_path):
    config = _config(tmp_path)
    config["data"]["Xray"] = {"HipRay": {}, "JRST": {}}
    experiment = _TwoSourceSegExperiment.from_config(config)

    metrics = experiment.run_phase("val", 0)

    expected = {
        "dice",
        "loss",
        "HipRay/dice",
        "HipRay/loss",
        "JRST/dice",
        "JRST/loss",
    }
    assert expected.issubset(metrics)
    assert all(torch.isfinite(torch.tensor(metrics[name])) for name in expected)
    assert 0.0 <= metrics["dice"] <= 1.0
    assert 0.0 <= metrics["HipRay/dice"] <= 1.0
    assert 0.0 <= metrics["JRST/dice"] <= 1.0


def test_zero_proportion_drops_dataset_from_train_and_val(tmp_path):
    config = _config(tmp_path)
    config["data"]["Xray"] = {"HipRay": {}, "JRST": {}}
    config["dataloader"]["proportions"] = {"HipRay": 0.75, "JRST": 0}
    experiment = _TwoSourceSegExperiment.from_config(config)

    assert set(experiment.train_dl.loaders) == {"HipRay"}
    assert set(experiment.val_dl.loaders) == {"HipRay"}


def test_build_data_prunes_zero_proportion_before_dataset_construction(monkeypatch):
    config = {
        "data": {
            "CT": {"RSNAFrac": {"version": 3.0}, "MOOSE": {"version": "?"}},
            "Xray": {"HipRay": {}},
        },
        "train": {"eval_freq": 0},
        "dataloader": {
            "batch_size": 1,
            "proportions": {"RSNAFrac": 1.0, "MOOSE": 0, "HipRay": 0},
        },
        "drr_model": {
            "default": {},
            "datasets": {"RSNAFrac": {}, "MOOSE": {}},
        },
    }
    captured = {}

    def fake_build_multimodal_datasets(data_cfg, *, include_validation):
        captured["data"] = data_cfg
        captured["include_validation"] = include_validation
        bundle = type(
            "Bundle",
            (),
            {"train": {}, "val": {}, "modalities": {}},
        )()
        captured["bundle"] = bundle
        return bundle

    def fake_drr_from_config(config_dict, *, device):
        captured["drr_datasets"] = dict(config_dict["drr_model"]["datasets"])
        return None

    monkeypatch.setattr(
        "fxr.experiment.segmentation.build_multimodal_datasets",
        fake_build_multimodal_datasets,
    )
    monkeypatch.setattr(
        "fxr.experiment.segmentation.DrrForwardPipeline.from_config",
        fake_drr_from_config,
    )

    experiment = object.__new__(FleXrayTrainExperiment)
    experiment.config = Config(config)
    experiment.device = torch.device("cpu")
    experiment.build_dataloader = lambda: captured.setdefault("loader_built", True)

    FleXrayTrainExperiment.build_data(experiment, load_data=True)

    assert captured["data"] == {"CT": {"RSNAFrac": {"version": 3.0}}, "Xray": {}}
    assert captured["include_validation"] is False
    assert captured["drr_datasets"] == {"RSNAFrac": {}}
    assert captured["loader_built"] is True
    assert experiment.dataset_bundle is captured["bundle"]


def test_loader_modality_overrides_batch_sizes(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    xray_config = _config(tmp_path / "xray")
    xray_config["data"]["Xray"] = {"HipRay": {}, "JRST": {}}
    xray_config["dataloader"] = {
        "batch_size": 2,
        "num_workers": 0,
        "Xray": {"batch_size": 3},
    }
    xray_experiment = _TwoSourceSegExperiment.from_config(xray_config)

    assert xray_experiment.train_dl.loaders["HipRay"].batch_size == 3

    ct_config = _ct_config(tmp_path / "ct")
    ct_config["dataloader"] = {
        "batch_size": 4,
        "num_workers": 0,
        "CT": {"batch_size": 1},
    }
    ct_experiment = _SyntheticCTExperiment.from_config(ct_config)

    assert ct_experiment.train_dl.loaders["RSNAFrac"].batch_size == 1


def test_ct_loader_rejects_explicit_batch_size_above_one(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = _ct_config(tmp_path)
    config["dataloader"] = {
        "batch_size": 4,
        "num_workers": 0,
        "CT": {"batch_size": 2},
    }

    with pytest.raises(ValueError, match="dataloader.CT.batch_size"):
        _SyntheticCTExperiment.from_config(config)


def test_negative_proportion_is_rejected():
    from fxr.experiment.segmentation import _excluded_datasets

    assert _excluded_datasets({"A": 2, "B": 0, "C": 1}) == {"B"}
    with pytest.raises(ValueError):
        _excluded_datasets({"A": -1})


_GEN_SOURCE_NAMES = ("background", "skull", "femurs")


class _SyntheticChannelMask(Dataset):
    """In-memory X-ray dataset emitting channel-first FluXray-style masks."""

    def __init__(self, size: int, seed: int = 0) -> None:
        self.size = size
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(size, 1, _HEIGHT, _WIDTH, generator=generator)
        shape = (size, len(_GEN_SOURCE_NAMES), _HEIGHT, _WIDTH)
        self.labels = torch.randint(0, 2, shape, generator=generator).float()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        return {
            "image": self.images[index],
            "label": self.labels[index],
            "dataset_name": "FluXray",
            "modality": "xray",
        }


class _SyntheticChannelMaskExperiment(FleXrayTrainExperiment):
    """Segmentation experiment backed by synthetic X-ray channel masks."""

    def build_data(self, load_data: bool) -> None:
        if not load_data:
            return
        self.train_datasets = {"FluXray": _SyntheticChannelMask(size=4, seed=1)}
        self.val_datasets = {"FluXray": _SyntheticChannelMask(size=2, seed=2)}
        self.modalities = {"FluXray": "xray"}
        self._channel_source_names = {"FluXray": _GEN_SOURCE_NAMES}
        self.drr_pipeline = None
        self.build_dataloader()


def _channel_mask_config(root) -> dict:
    config = _config(root)
    config["data"] = {"Xray": {"FluXray": {"version": "1.0"}}}
    return config


def test_wandb_sample_logger_handles_channel_mask_sources(tmp_path):
    experiment = _SyntheticChannelMaskExperiment.from_config(_channel_mask_config(tmp_path))
    logger = WandbSamplePredictionLogger(
        experiment, every=1, max_samples=1, batch_size=2
    )
    labels = resolve_run_output_label_names(_channel_mask_config(tmp_path))

    outputs = logger._predict_one_batch(
        "FluXray", experiment.val_datasets["FluXray"]
    )

    assert outputs["dataset_name"] == "FluXray"
    assert outputs["y_true"].shape[1] == len(labels) == 61


def test_channel_mask_step_loss_is_finite(tmp_path):
    experiment = _SyntheticChannelMaskExperiment.from_config(_channel_mask_config(tmp_path))
    labels = resolve_run_output_label_names(_channel_mask_config(tmp_path))
    batch = next(iter(experiment.train_dl))

    outputs = experiment.run_step(batch, phase="train", backward=True)
    metrics = experiment.compute_metrics(outputs)

    # The stored channel mask is projected onto the configured model space.
    assert outputs["y_true"].shape[1] == len(labels) == 61
    assert outputs["y_pred"].shape[1] == len(labels)
    assert torch.isfinite(outputs["loss"]).all()
    assert 0.0 <= metrics["dice"] <= 1.0


def test_channel_mask_model_inputs_project_named_channels(tmp_path):
    protocol = load_protocol_by_name("all_structures_flexray_v4")
    experiment = object.__new__(FleXrayTrainExperiment)
    experiment.model_label_names = tuple(protocol.labels)
    experiment._channel_source_names = {"FluXray": _GEN_SOURCE_NAMES}

    label = torch.zeros(1, len(_GEN_SOURCE_NAMES), 2, 2)
    label[0, 1, 0, 0] = 1.0  # skull
    label[0, 2, 1, 1] = 1.0  # femurs
    inputs = BatchInputs(
        modality="xray",
        image=torch.zeros(1, 1, 2, 2),
        label=label,
        affine=None,
        fg_centroids_ijk=None,
    )

    image, projected = experiment._model_inputs(inputs, dataset_name="FluXray")

    # Channels land in their named model positions, not their source order.
    assert projected.shape == (1, len(protocol.labels), 2, 2)
    assert projected[0, protocol.label_to_id["skull"], 0, 0] == 1.0
    assert projected[0, protocol.label_to_id["femurs"], 1, 1] == 1.0
    assert torch.equal(image, inputs.image)


def _states_match(left: dict, right: dict) -> bool:
    if left.keys() != right.keys():
        return False
    return all(torch.allclose(left[key], right[key]) for key in left)


def test_resolve_supervise_empty_label_ids_maps_packaged_specs_to_model_channels() -> None:
    from fxr.experiment.protocol_resolve import resolve_supervise_empty_label_ids
    from fxr.protocols import resolve_run_output_label_names

    config = {"protocol": {"name": "all_structures_flexray_v4"}}
    labels = resolve_run_output_label_names(config)

    resolved = resolve_supervise_empty_label_ids(config, ["ElbowCT", "MOOSE", "NotPackaged"])

    expected = tuple(labels.index(name) for name in ("femurs", "patellae", "tibiae", "fibulae"))
    assert resolved == {"ElbowCT": expected}

    subset = {
        "protocol": {
            "name": "all_structures_flexray_v4",
            "model_labels": {"names": ["background", "humeri", "femurs", "tibiae"]},
        }
    }
    assert resolve_supervise_empty_label_ids(subset, ["ElbowCT"]) == {"ElbowCT": (2, 3)}


def test_routed_loss_receives_dataset_supervise_empty_ids(tmp_path) -> None:
    from fxr.protocols import load_protocol_by_name

    config = _config(tmp_path)
    config["data"] = {"Xray": {"FootBones": {}, "HipRay": {}}}
    config["dataloader"]["proportions"] = {"FootBones": 1.0, "HipRay": 0.0}
    config["loss_func"] = {
        "_class": "DatasetRoutedLoss",
        "losses": {
            "partial": {
                "_class": "PixelCELoss",
                "from_logits": True,
                "mode": "binary",
                "ignore_empty_labels": True,
            }
        },
        "dataset_losses": {"FootBones": "partial"},
    }
    experiment = _TrainOnlySyntheticSegExperiment.from_config(config, load_data=False)
    try:
        labels = load_protocol_by_name("all_structures_flexray_v4").labels
        expected = tuple(
            labels.index(name)
            for name in ("phalanges", "metacarpals", "carpals", "ulnae", "radii")
        )
        assert experiment.loss_func.dataset_supervise_empty_label_ids == {
            "FootBones": expected
        }
    finally:
        experiment.close()


def test_experiment_builds_per_modality_pipelines_and_snapshots_presets(tmp_path) -> None:
    from fxr.augmentation import load_named_augmentation_preset

    config = _config(tmp_path)
    config["augmentation"] = {"presets": {"Xray": "base_light"}}
    experiment = _TrainOnlySyntheticSegExperiment.from_config(config, load_data=False)
    try:
        assert set(experiment.aug_pipelines) == {"ct", "xray"}
        snapshot_dir = experiment.path / "augmentations"
        assert load_named_augmentation_preset(
            "xray", config_root=experiment.path
        ) == load_named_augmentation_preset("base_light")
        assert load_named_augmentation_preset(
            "ct", config_root=experiment.path
        ) == load_named_augmentation_preset("CT_base")
        assert sorted(p.name for p in snapshot_dir.iterdir()) == ["ct.yml", "xray.yml"]
    finally:
        experiment.close()


def test_ct_model_inputs_filter_centroids_to_supervised_labels() -> None:
    from fxr.datasets import compile_training_label_remap_by_name
    from fxr.experiment.label_projection import TrainingLabelProjection
    from fxr.experiment.segmentation import supervised_foreground_centroids

    remap = compile_training_label_remap_by_name("all_structures_flexray_v4", "PedsCT")
    projection = TrainingLabelProjection.from_label_remap(remap)
    centroids = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    inputs = BatchInputs(
        modality="ct",
        image=torch.zeros(1, 1, 2, 2, 2),
        label=torch.zeros(1, 1, 2, 2, 2, dtype=torch.long),
        affine=torch.eye(4),
        fg_centroids_ijk=centroids,
        fg_centroid_label_ids=torch.tensor([1, 13, 16]),  # adrenal (dropped), kidney, liver
    )

    kept = supervised_foreground_centroids(inputs, projection)

    assert torch.equal(kept, centroids[1:])
    untagged = BatchInputs("ct", inputs.image, inputs.label, inputs.affine, centroids)
    assert supervised_foreground_centroids(untagged, projection) is centroids


def test_ct_train_loader_uses_weighted_sampler_when_configured(tmp_path) -> None:
    from torch.utils.data import WeightedRandomSampler

    from fxr.protocols import load_protocol_by_name

    class _Weighted(torch.utils.data.Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return index

        def sample_weights(self, label_lut):
            assert len(label_lut) == 16  # RSNAFrac native ids 0..15
            return torch.tensor([1.0, 2.0, 0.0])

    experiment = object.__new__(FleXrayTrainExperiment)
    experiment.config = Config(_ct_config(tmp_path))
    experiment.model_label_names = tuple(load_protocol_by_name("all_structures_flexray_v4").labels)
    experiment._projection_cache = {}

    sampler = experiment._weighted_sampler(_Weighted(), "RSNAFrac", seed=3)
    assert isinstance(sampler, WeightedRandomSampler)
    assert sampler.num_samples == 3 and sampler.replacement
    assert set(iter(sampler)) <= {0, 1}
    assert experiment._weighted_sampler(torch.utils.data.TensorDataset(torch.zeros(2)), "RSNAFrac", seed=3) is None


def test_channel_projection_recomputes_background_after_dropped_native_channels(tmp_path) -> None:
    from fxr.protocols import load_protocol_by_name

    labels = load_protocol_by_name("all_structures_flexray_v4").labels
    experiment = object.__new__(FleXrayTrainExperiment)
    experiment.model_label_names = tuple(labels)
    experiment._channel_source_names_for = lambda name: ("background", "hips", "background")
    label = torch.zeros(1, 3, 1, 2)
    label[0, 1, 0, 0] = 1.0  # hips
    label[0, 2, 0, 0] = 1.0  # dropped native structure overlapping the hip
    label[0, 0, 0, 1] = 1.0

    projected = experiment._project_channel_label(label, "Toy")

    hips = labels.index("hips")
    assert projected[0, hips].tolist() == [[1.0, 0.0]]
    assert projected[0, 0].tolist() == [[0.0, 1.0]]


def test_evalset_area_filter_ignores_small_labels_and_skips_unscoreable_images() -> None:
    from fxr.experiment.callbacks import _area_filtered_dice

    y_true = torch.zeros(3, 3, 4, 4)
    y_true[0, 1, :2, :] = 1.0  # 50% area: scored
    y_true[0, 2, 0, 0] = 1.0  # 6% area: below threshold
    y_true[1, 2, :, :] = 1.0  # 100% area: scored
    y_true[2, 1, 0, 0] = 1.0  # only a tiny label: image skipped
    y_true[:, 0] = y_true[:, 1:].amax(dim=1) <= 0
    logits = torch.full((3, 3, 4, 4), -10.0)
    logits[:, 1, :2, :] = 10.0  # perfect hip prediction
    logits[1, 2] = 10.0  # perfect femur prediction on image 1

    dice_sum, count = _area_filtered_dice(logits, y_true, min_fraction=0.25)

    assert count == 2.0
    assert dice_sum == pytest.approx(2.0)


def test_evalset_area_filter_zero_matches_legacy_dice(tmp_path) -> None:
    experiment = _SyntheticSegExperiment.from_config(_config(tmp_path))
    try:
        legacy = EvalSetMetricLogger(experiment, data={"Xray": {"HipRay": {}}}, batch_size=2)
        filtered = EvalSetMetricLogger(
            experiment,
            data={"Xray": {"HipRay": {}}},
            batch_size=2,
            min_ground_truth_area_fraction=0.0,
        )
        dataset = experiment.val_datasets["HipRay"]
        assert legacy._evaluate_dataset("HipRay", dataset) == filtered._evaluate_dataset(
            "HipRay", dataset
        )
        with pytest.raises(ValueError, match="min_ground_truth_area_fraction"):
            EvalSetMetricLogger(
                experiment, data={"Xray": {}}, min_ground_truth_area_fraction=1.5
            )
    finally:
        experiment.close()
