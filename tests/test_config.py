"""Tests for the dependency-free experiment config layer (``fxr.config``)."""

from __future__ import annotations

import functools

import pytest

from fxr.config import (
    Config,
    absolute_import,
    check_missing,
    config_digest,
    eval_config,
    generate_tuid,
    merge_configs,
    prune_zero_proportion_datasets,
    zero_proportion_dataset_names,
)
from fxr.config.training import requires_training_source_validation


def test_config_dotted_access_and_subconfig():
    cfg = Config({"train": {"epochs": 10, "opt": {"lr": 0.1}}})

    assert cfg["train.epochs"] == 10
    assert cfg["train.opt.lr"] == 0.1
    # Sub-mappings are themselves Config objects.
    sub = cfg["train"]
    assert isinstance(sub, Config)
    assert sub["opt.lr"] == 0.1


def test_config_get_default_and_contains():
    cfg = Config({"train": {"epochs": 10}})

    assert "train.epochs" in cfg
    assert "train.missing" not in cfg
    assert cfg.get("train.missing", 7) == 7
    # Explicit None default must be distinguishable from absence handling.
    assert cfg.get("train.missing") is None


@pytest.mark.parametrize("every", [True, 0.5, "not-an-integer"])
def test_malformed_sample_callback_interval_conservatively_requires_validation(
    every,
):
    """Malformed sample intervals never bypass validation-split readiness."""

    config = {
        "train": {"eval_freq": 0},
        "callbacks": {
            "epoch": {
                "samples": {
                    "_class": "fxr.experiment.WandbSamplePredictionLogger",
                    "every": every,
                }
            }
        },
    }

    assert requires_training_source_validation(config) is True


def test_config_to_dict_is_deep_copy():
    source = {"a": {"b": 1}}
    cfg = Config(source)

    dumped = cfg.to_dict()
    dumped["a"]["b"] = 99
    # Mutating the dump must not affect the config.
    assert cfg["a.b"] == 1


def test_eval_config_instantiates_class():
    # Entries other than ``_class`` are passed as keyword arguments.
    cfg = {"_class": "builtins.dict", "x": 1, "y": 2}

    obj = eval_config(cfg)

    assert obj == {"x": 1, "y": 2}


def test_eval_config_recurses_nested_objects():
    cfg = {
        "_class": "builtins.dict",
        "inner": {"_class": "builtins.dict", "x": 1},
        "items_list": [{"_class": "builtins.dict", "y": 2}],
    }

    obj = eval_config(cfg)

    assert obj == {"inner": {"x": 1}, "items_list": [{"y": 2}]}


def test_eval_config_fn_returns_partial():
    cfg = {"_fn": "builtins.dict"}

    built = eval_config(cfg)

    assert isinstance(built, functools.partial)
    assert built([("k", 2)]) == {"k": 2}


def test_eval_config_rejects_class_and_fn():
    with pytest.raises(ValueError):
        eval_config({"_class": "collections.Counter", "_fn": "collections.Counter"})


def test_absolute_import_resolves_and_rejects():
    assert absolute_import("collections.OrderedDict").__name__ == "OrderedDict"
    with pytest.raises(ValueError):
        absolute_import("nodots")
    with pytest.raises(ImportError):
        absolute_import("collections.DefinitelyNotHere")


def test_config_digest_is_order_independent_and_sensitive():
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    c = {"x": 1, "y": 3}

    assert config_digest(a) == config_digest(b)
    assert config_digest(a) != config_digest(c)


def test_merge_configs_deep_merges_without_mutating_base():
    base = {"train": {"epochs": 10, "lr": 0.1}, "data": {"name": "A"}}
    override = {"train": {"lr": 0.5}, "extra": True}

    merged = merge_configs(base, override)

    assert merged == {
        "train": {"epochs": 10, "lr": 0.5},
        "data": {"name": "A"},
        "extra": True,
    }
    # Base is untouched.
    assert base["train"]["lr"] == 0.1


def test_check_missing_flags_placeholder():
    with pytest.raises(ValueError, match="dataloader.batch_size"):
        check_missing({"dataloader": {"batch_size": "?"}})
    # A fully specified config passes silently.
    check_missing({"dataloader": {"batch_size": 4}})


def test_prune_zero_proportion_datasets_removes_inactive_config_leaves():
    config = {
        "data": {
            "CT": {"ActiveCT": {"version": 1}, "InactiveCT": {"version": "?"}},
            "Xray": {"InactiveXray": {"version": "?"}, "OmittedXray": {"version": "?"}},
        },
        "dataloader": {
            "proportions": {
                "ActiveCT": 0.5,
                "InactiveCT": 0,
                "InactiveXray": "0.0",
            }
        },
        "drr_model": {"datasets": {"ActiveCT": {}, "InactiveCT": {}}},
        "loss_func": {
            "_class": "DatasetRoutedLoss",
            "losses": {"standard": {}},
            "dataset_losses": {
                "ActiveCT": "standard",
                "InactiveCT": "standard",
                "InactiveXray": "standard",
                "OmittedXray": "standard",
            },
        },
        "callbacks": {
            "epoch": {
                "eval_sets": {
                    "_class": "fxr.experiment.EvalSetMetricLogger",
                    "data": {"Xray": {"InactiveXray": {}, "EvalOnlyXray": {}}},
                }
            }
        },
    }

    pruned = prune_zero_proportion_datasets(config)

    assert zero_proportion_dataset_names(config) == {"InactiveCT", "InactiveXray"}
    assert pruned["data"] == {"CT": {"ActiveCT": {"version": 1}}, "Xray": {}}
    assert pruned["drr_model"]["datasets"] == {"ActiveCT": {}}
    assert pruned["loss_func"]["dataset_losses"] == {"ActiveCT": "standard"}
    # Callback eval-set data is exempt from pruning: eval coverage is
    # independent of the train mix.
    assert pruned["callbacks"]["epoch"]["eval_sets"]["data"] == {
        "Xray": {"InactiveXray": {}, "EvalOnlyXray": {}}
    }
    check_missing(pruned)


def test_zero_proportion_dataset_names_accepts_fractional_weights():
    inactive = zero_proportion_dataset_names(
        {"dataloader": {"proportions": {"A": 0.5, "B": "0.0"}}}
    )

    assert inactive == {"B"}


def test_zero_proportion_dataset_names_rejects_bad_weights():
    with pytest.raises(ValueError, match=">= 0"):
        zero_proportion_dataset_names({"dataloader": {"proportions": {"A": -1}}})
    with pytest.raises(ValueError, match="finite non-negative"):
        zero_proportion_dataset_names({"dataloader": {"proportions": {"A": "bad"}}})


def test_generate_tuid_shape():
    timestamp, nonce = generate_tuid(nonce_length=4)

    assert len(timestamp) == len("YYYYmmdd_HHMMSS")
    assert len(nonce) == 4
    assert nonce.isalnum()
