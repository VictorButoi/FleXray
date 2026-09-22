from __future__ import annotations

import os
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import pytest
import yaml

import fxr.protocols as public_protocols
from fxr.protocols import (
    DatasetSpec,
    EvalContract,
    ModelLabelSpace,
    ProtocolSpec,
    compile_eval_contract_by_name,
    compile_training_lut_by_name,
    list_dataset_spec_names,
    list_protocol_names,
    load_dataset_spec_by_name,
    load_eval_contract_by_model_name,
    load_model_label_space_by_name,
    load_protocol_by_name,
)

PROTOCOL_NAME = "all_structures_flexray_v4"


def _write_config(
    root: Path,
    subdir: str,
    name: str,
    text: str,
    *,
    suffix: str = ".yml",
) -> Path:
    path = root / subdir / f"{name}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _toy_protocol_text(name: str = "toy") -> str:
    return f"""\
protocol_name: {name}
labels:
  - background
  - hip
"""


def _toy_dataset_text(name: str = "ToySet") -> str:
    return f"""\
dataset_name: {name}
skip_subjects: null
stored_labels:
  0: background
  1: hip
"""


def _toy_eval_contract_text(model_name: str = "toy") -> str:
    return f"""\
model_name: {model_name}
identity_defaults: dataset_eval_labels
eval_sets:
  ToySet: {{}}
"""


def _packaged_base_xray_eval_set_names() -> set[str]:
    text = files("fxr.configs.training").joinpath("base.yml").read_text(
        encoding="utf-8",
    )
    raw = yaml.safe_load(text)
    return set(raw["callbacks"]["epoch"]["eval_sets"]["data"]["Xray"])

def _normalized_foreground_labels(
    spec: DatasetSpec,
    protocol: ProtocolSpec,
) -> tuple[str, ...]:
    """Derive normalized foreground labels from one packaged dataset spec."""

    aliases = spec.protocol_label_aliases.get(protocol.protocol_name, {})
    dropped = set(spec.protocol_drop_labels.get(protocol.protocol_name, ()))
    normalized: list[str] = []
    for _, source_name in sorted(spec.stored_labels.items()):
        target_name = aliases.get(source_name, source_name)
        if source_name in dropped or target_name == "background":
            continue
        assert target_name in protocol.label_to_id
        if target_name not in normalized:
            normalized.append(target_name)
    return tuple(normalized)


def test_public_protocol_api_exposes_usable_core_types_and_loaders() -> None:
    assert set(public_protocols.__all__) == {
        "DatasetSpec",
        "EvalContract",
        "ModelLabelSpace",
        "ProtocolSpec",
        "compile_eval_contract_by_name",
        "compile_training_lut",
        "compile_training_lut_by_name",
        "describe_dataset",
        "describe_protocol",
        "explain_lut",
        "explain_mapping",
        "list_datasets",
        "list_dataset_spec_names",
        "list_protocol_names",
        "list_protocols",
        "load_dataset_spec",
        "load_dataset_spec_by_name",
        "load_eval_contract_by_model_name",
        "load_model_label_space_by_name",
        "load_protocol",
        "load_protocol_by_name",
        "resolve_run_attenuated_label_ids",
        "resolve_run_output_label_names",
        "resolve_run_protocol_spec",
        "resolve_run_render_label_collapse_index",
        "resolve_run_render_label_names",
    }
    exports = (
        ProtocolSpec,
        DatasetSpec,
        ModelLabelSpace,
        EvalContract,
        compile_eval_contract_by_name,
        compile_training_lut_by_name,
        list_dataset_spec_names,
        list_protocol_names,
        load_dataset_spec_by_name,
        load_eval_contract_by_model_name,
        load_model_label_space_by_name,
        load_protocol_by_name,
    )

    for export in exports:
        assert getattr(public_protocols, export.__name__) is export


def test_list_named_configs_match_packaged_directories() -> None:
    protocol_stems = {
        path.name.rsplit(".", 1)[0]
        for path in files("fxr.configs").joinpath("protocols").iterdir()
        if path.name.endswith((".yml", ".yaml"))
    }
    dataset_stems = {
        path.name.rsplit(".", 1)[0]
        for path in files("fxr.configs").joinpath("datasets").iterdir()
        if path.name.endswith((".yml", ".yaml"))
    }

    protocol_names = list_protocol_names()
    dataset_names = list_dataset_spec_names()
    assert protocol_names == tuple(sorted(protocol_stems))
    assert dataset_names == tuple(sorted(dataset_stems))
    assert PROTOCOL_NAME in protocol_names
    for name in dataset_names:
        assert load_dataset_spec_by_name(name).dataset_name == name


def test_list_named_configs_honor_custom_config_root(tmp_path: Path) -> None:
    _write_config(tmp_path, "protocols", "toy", _toy_protocol_text())
    _write_config(tmp_path, "datasets", "ToySet", _toy_dataset_text())

    assert list_protocol_names(config_root=tmp_path) == ("toy",)
    assert list_dataset_spec_names(config_root=tmp_path) == ("ToySet",)

    _write_config(tmp_path, "protocols", "toy", _toy_protocol_text(), suffix=".yaml")
    with pytest.raises(ValueError, match="Ambiguous FleXray protocol"):
        list_protocol_names(config_root=tmp_path)


def test_strict_yaml_rejects_duplicate_keys(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "protocols",
        "dup",
        """\
protocol_name: dup
protocol_name: dup
labels:
  - background
  - hip
""",
    )

    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key"):
        load_protocol_by_name("dup", config_root=tmp_path)


@pytest.mark.parametrize(
    "payload,match",
    [
        (
            """\
protocol_name: toy
labels:
  - background
  - hip
unexpected: true
""",
            "unknown keys",
        ),
        (
            """\
labels:
  - background
  - hip
""",
            "protocol_name",
        ),
        (
            """\
protocol_name: toy
labels:
  - hip
  - background
""",
            "background.*channel 0",
        ),
    ],
)
def test_strict_protocol_validation_errors(
    tmp_path: Path,
    payload: str,
    match: str,
) -> None:
    _write_config(tmp_path, "protocols", "toy", payload)

    with pytest.raises((TypeError, ValueError), match=match):
        load_protocol_by_name("toy", config_root=tmp_path)


@pytest.mark.parametrize(
    "payload,removed_key",
    [
        (
            """\
dataset_name: ToySet
mode: native_remap
skip_subjects: null
stored_labels:
  0: background
  1: hip
""",
            "mode",
        ),
        (
            """\
dataset_name: ToySet
skip_subjects: null
training_visible_labels:
  - hip
stored_labels:
  0: background
  1: hip
""",
            "training_visible_labels",
        ),
        (
            """\
dataset_name: ToySet
skip_subjects: null
requires_overlay_priority: true
stored_labels:
  0: background
  1: hip
""",
            "requires_overlay_priority",
        ),
    ],
)
def test_dataset_specs_reject_removed_keys(
    tmp_path: Path,
    payload: str,
    removed_key: str,
) -> None:
    _write_config(tmp_path, "datasets", "ToySet", payload)

    with pytest.raises(ValueError, match=removed_key):
        load_dataset_spec_by_name("ToySet", config_root=tmp_path)


def test_named_registry_resolves_yaml_and_rejects_duplicate_suffixes(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path, "protocols", "toy", _toy_protocol_text(), suffix=".yaml")
    assert load_protocol_by_name("toy", config_root=tmp_path).protocol_name == "toy"

    _write_config(tmp_path, "protocols", "toy", _toy_protocol_text(), suffix=".yml")
    with pytest.raises(ValueError, match="Ambiguous FleXray protocol"):
        load_protocol_by_name("toy", config_root=tmp_path)


def test_named_registry_rejects_path_like_and_declared_name_mismatch(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path, "protocols", "toy", _toy_protocol_text("other"))

    with pytest.raises(ValueError, match="does not match requested name"):
        load_protocol_by_name("toy", config_root=tmp_path)

    with pytest.raises(ValueError, match="must be a config name"):
        load_protocol_by_name("../toy", config_root=tmp_path)


def test_packaged_dataset_specs_are_self_consistent_and_aliases_resolve() -> None:
    dataset_root = files("fxr.configs").joinpath("datasets")
    dataset_paths = sorted(
        (path for path in dataset_root.iterdir() if path.name.endswith(".yml")),
        key=lambda path: path.name,
    )

    assert dataset_paths
    for path in dataset_paths:
        config_name = Path(path.name).stem
        spec = load_dataset_spec_by_name(config_name)
        native_ids = tuple(spec.stored_labels)
        assert native_ids
        assert spec.dataset_name == config_name
        assert native_ids == tuple(sorted(set(native_ids)))
        assert all(native_id >= 0 for native_id in native_ids)
        assert spec.stored_labels[0] == "background"
        assert spec.label_to_native_id == {
            label_name: native_id
            for native_id, label_name in spec.stored_labels.items()
        }
        for alias in spec.dataset_aliases:
            assert load_dataset_spec_by_name(alias) == spec


def test_packaged_dataset_specs_match_training_and_eval_references() -> None:
    dataset_root = files("fxr.configs").joinpath("datasets")
    spec_names = {
        path.name.removesuffix(".yml")
        for path in dataset_root.iterdir()
        if path.name.endswith(".yml")
    }
    training_root = files("fxr.configs").joinpath("training")
    training_source_names: set[str] = set()
    for path in training_root.iterdir():
        if not path.name.endswith(".yml"):
            continue
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        for sources in raw.get("data", {}).values():
            training_source_names.update(sources)
    canonical_training_sources = {
        load_dataset_spec_by_name(name).dataset_name
        for name in training_source_names
    }
    contract_sources = set(
        load_eval_contract_by_model_name(PROTOCOL_NAME).eval_sets
    )

    assert spec_names
    assert spec_names == canonical_training_sources | contract_sources


def test_packaged_protocol_and_eval_contract_resources_load() -> None:
    config_root = files("fxr.configs")
    protocol_root = config_root.joinpath("protocols")
    contract_root = config_root.joinpath("eval_contracts")
    protocol_paths = sorted(
        (path for path in protocol_root.iterdir() if path.name.endswith(".yml")),
        key=lambda path: path.name,
    )
    contract_paths = sorted(
        (path for path in contract_root.iterdir() if path.name.endswith(".yml")),
        key=lambda path: path.name,
    )

    assert protocol_paths
    assert contract_paths
    for path in protocol_paths:
        name = Path(path.name).stem
        assert load_protocol_by_name(name).protocol_name == name
    for path in contract_paths:
        name = Path(path.name).stem
        assert load_eval_contract_by_model_name(name).model_name == name


def test_base_xray_eval_sets_have_eval_contract_coverage() -> None:
    contract = load_eval_contract_by_model_name(PROTOCOL_NAME)
    for dataset_name in sorted(_packaged_base_xray_eval_set_names()):
        mapping = compile_eval_contract_by_name(PROTOCOL_NAME, dataset_name)

        assert mapping.eval_set_name in contract.eval_sets
        assert mapping.eval_label_names


def test_base_xray_eval_sets_are_supported_contract_entries() -> None:
    contract = load_eval_contract_by_model_name(PROTOCOL_NAME)
    resolved_entries = {
        compile_eval_contract_by_name(PROTOCOL_NAME, dataset_name).eval_set_name
        for dataset_name in _packaged_base_xray_eval_set_names()
    }

    assert resolved_entries <= set(contract.eval_sets)


def test_flexray_v4_protocol_resource_and_colormap_are_consistent() -> None:
    protocol = load_protocol_by_name(PROTOCOL_NAME)

    assert protocol.labels[0] == "background"
    assert len(protocol.labels) == len(set(protocol.labels))
    assert protocol.label_to_id == {
        label_name: index for index, label_name in enumerate(protocol.labels)
    }

    colormap_path = files("fxr.configs").joinpath(
        "colormaps",
        f"{PROTOCOL_NAME}.yml",
    )
    colormap = yaml.safe_load(colormap_path.read_text(encoding="utf-8"))
    assert colormap["protocol_name"] == PROTOCOL_NAME
    assert tuple(colormap["labels"]) == protocol.labels
    assert colormap["labels"]["background"] is None


def test_packaged_model_label_space_omits_eval_only_spine_aggregates() -> None:
    protocol = load_protocol_by_name(PROTOCOL_NAME)
    model = load_model_label_space_by_name(PROTOCOL_NAME)

    assert model.model_name == PROTOCOL_NAME
    assert len(protocol.labels) == 63
    assert len(model.labels) == 61
    assert model.labels == tuple(
        label_name
        for label_name in protocol.labels
        if label_name not in {"lumbar_spine", "thoracolumbar_spine"}
    )


def test_training_lut_follows_packaged_alias_and_drop_rules() -> None:
    protocol = load_protocol_by_name(PROTOCOL_NAME)
    spec = load_dataset_spec_by_name("FluXray")
    lut = compile_training_lut_by_name(PROTOCOL_NAME, "FluXray")
    aliases = spec.protocol_label_aliases.get(PROTOCOL_NAME, {})
    dropped = set(spec.protocol_drop_labels.get(PROTOCOL_NAME, ()))

    assert lut.dataset_name == spec.dataset_name
    assert lut.protocol_name == PROTOCOL_NAME
    assert len(lut.label_lut) == max(spec.stored_labels) + 1
    assert all(protocol_id >= 0 for protocol_id in lut.label_lut)
    for native_id, source_name in spec.stored_labels.items():
        target_name = (
            "background"
            if source_name in dropped
            else aliases.get(source_name, source_name)
        )
        assert lut.label_lut[native_id] == protocol.label_to_id[target_name]
        assert lut.native_id_to_protocol_label[native_id] == target_name


def test_packaged_identity_eval_defaults_follow_dataset_normalization() -> None:
    protocol = load_protocol_by_name(PROTOCOL_NAME)
    model = load_model_label_space_by_name(PROTOCOL_NAME)
    contract = load_eval_contract_by_model_name(PROTOCOL_NAME)
    identity_eval_sets = [
        dataset_name
        for dataset_name, explicit_rules in contract.eval_sets.items()
        if not explicit_rules
    ]

    assert identity_eval_sets
    for dataset_name in identity_eval_sets:
        spec = load_dataset_spec_by_name(dataset_name)
        mapping = compile_eval_contract_by_name(PROTOCOL_NAME, dataset_name)
        expected_names = _normalized_foreground_labels(spec, protocol)
        assert mapping.eval_label_names == expected_names
        assert mapping.eval_label_to_model_ids == {
            label_name: (model.label_to_id[label_name],)
            for label_name in expected_names
        }


def test_eval_identity_defaults_apply_aliases_drops_and_dedupe(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path, "protocols", "toy", _toy_protocol_text())
    _write_config(
        tmp_path,
        "datasets",
        "ToySet",
        """\
dataset_name: ToySet
skip_subjects: null
stored_labels:
  0: background
  1: hip_left
  2: hip_right
  3: ignored_native_label
protocol_label_aliases:
  toy:
    hip_left: hip
    hip_right: hip
protocol_drop_labels:
  toy:
    - ignored_native_label
""",
    )
    _write_config(tmp_path, "eval_contracts", "toy", _toy_eval_contract_text())

    mapping = compile_eval_contract_by_name("toy", "ToySet", config_root=tmp_path)

    assert mapping.eval_label_names == ("hip",)
    assert mapping.eval_label_to_model_labels == {"hip": ("hip",)}


def test_unknown_stored_labels_error_without_alias_or_drop(tmp_path: Path) -> None:
    _write_config(tmp_path, "protocols", "toy", _toy_protocol_text())
    _write_config(
        tmp_path,
        "datasets",
        "ToySet",
        """\
dataset_name: ToySet
skip_subjects: null
stored_labels:
  0: background
  1: femur
""",
    )
    _write_config(tmp_path, "eval_contracts", "toy", _toy_eval_contract_text())

    with pytest.raises(ValueError, match="add it to protocol_label_aliases"):
        compile_training_lut_by_name("toy", "ToySet", config_root=tmp_path)

    with pytest.raises(ValueError, match="add it to protocol_label_aliases"):
        compile_eval_contract_by_name("toy", "ToySet", config_root=tmp_path)


def test_eval_contract_explicit_aggregate_candidates(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "protocols",
        "toy",
        """\
protocol_name: toy
labels:
  - background
  - vertebra_c1
  - vertebra_c2
""",
    )
    _write_config(
        tmp_path,
        "datasets",
        "SpineSet",
        """\
dataset_name: SpineSet
skip_subjects: null
stored_labels:
  0: background
  1: vertebra_c1
  2: vertebra_c2
""",
    )
    _write_config(
        tmp_path,
        "eval_contracts",
        "toy",
        """\
model_name: toy
identity_defaults: dataset_eval_labels
eval_sets:
  SpineSet:
    cervicothoracic_spine:
      candidates:
        - model_labels:
            - cervicothoracic_spine
        - model_labels:
            - vertebra_c1
            - vertebra_c2
""",
    )

    mapping = compile_eval_contract_by_name("toy", "SpineSet", config_root=tmp_path)

    assert mapping.eval_label_names == ("cervicothoracic_spine",)
    assert mapping.eval_label_to_model_labels["cervicothoracic_spine"] == (
        "vertebra_c1",
        "vertebra_c2",
    )


def test_name_based_compile_works_with_custom_config_root(tmp_path: Path) -> None:
    _write_config(tmp_path, "protocols", "toy", _toy_protocol_text())
    _write_config(tmp_path, "datasets", "ToySet", _toy_dataset_text())
    _write_config(
        tmp_path,
        "eval_contracts",
        "toy",
        _toy_eval_contract_text(),
    )

    lut = compile_training_lut_by_name("toy", "ToySet", config_root=tmp_path)
    mapping = compile_eval_contract_by_name("toy", "ToySet", config_root=tmp_path)

    assert lut.label_lut == (0, 1)
    assert mapping.eval_label_names == ("hip",)


def test_protocol_code_does_not_import_heavy_dependencies() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "fxr" / "protocols"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in source_root.glob("*.py")
    )
    for blocked in ("torch", "renderer", "trainer"):
        assert blocked not in source

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    script = """
import sys
import fxr.protocols
blocked_prefixes = ("torch",)
blocked = [
    name for name in sys.modules
    if name in blocked_prefixes or name.startswith(tuple(p + "." for p in blocked_prefixes))
]
print(blocked)
raise SystemExit(1 if blocked else 0)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_dataset_specs_load_supervise_empty_labels_and_reject_stored_names(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path, "protocols", "toy", _toy_protocol_text())
    _write_config(
        tmp_path,
        "datasets",
        "ToySet",
        _toy_dataset_text() + "supervise_empty_labels:\n  - femur\n",
    )
    assert load_dataset_spec_by_name("ToySet", config_root=tmp_path).supervise_empty_labels == (
        "femur",
    )
    assert load_dataset_spec_by_name("HipRay").supervise_empty_labels == ()

    _write_config(
        tmp_path,
        "datasets",
        "ToySet",
        _toy_dataset_text() + "supervise_empty_labels:\n  - hip\n",
    )
    with pytest.raises(ValueError, match="never stores"):
        load_dataset_spec_by_name("ToySet", config_root=tmp_path)


def test_deepfluoro_sacrum_halves_alias_to_protocol_sacrum() -> None:
    compiled = compile_training_lut_by_name("all_structures_flexray_v4", "DeepFluoro")

    assert compiled.native_id_to_protocol_label[4] == "sacrum"
    assert compiled.native_id_to_protocol_label[7] == "sacrum"
    assert "subject01_003" in load_dataset_spec_by_name("DeepFluoro").skip_subjects


def test_dataset_spec_rejects_known_negative_reached_through_alias(tmp_path):
    from fxr.protocols import load_dataset_spec

    path = tmp_path / "custom.yml"
    path.write_text(yaml.safe_dump({
        "dataset_name": "Custom", "skip_subjects": None,
        "stored_labels": {0: "background", 1: "left_hip"},
        "protocol_label_aliases": {"toy": {"left_hip": "hip"}},
        "supervise_empty_labels": ["hip"],
    }))
    with pytest.raises(ValueError, match="stored or aliased labels.*hip"):
        load_dataset_spec(path)


@pytest.mark.parametrize("modality", ["Xray", "CT"])
def test_known_negatives_load_from_configured_spec_path(tmp_path, modality):
    from fxr.experiment.protocol_resolve import resolve_supervise_empty_label_ids

    path = tmp_path / "custom.yml"
    spec = {
        "dataset_name": "Custom", "skip_subjects": None,
        "stored_labels": {0: "background", 1: "humeri"},
        "supervise_empty_labels": ["femurs", "tibiae"],
    }
    path.write_text(yaml.safe_dump(spec))
    config = {
        "protocol": {
            "name": PROTOCOL_NAME,
            "model_labels": {"names": ["background", "humeri", "femurs"]},
        },
        "data": {modality: {"Custom": {"dataset_spec": str(path)}}},
    }
    assert resolve_supervise_empty_label_ids(config, ["Custom"]) == {"Custom": (2,)}
    spec["supervise_empty_labels"] = ["outside_protocol"]
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(AssertionError, match="outside protocol"):
        resolve_supervise_empty_label_ids(config, ["Custom"])


def test_configured_spec_overrides_packaged_known_negatives(tmp_path):
    from fxr.experiment.protocol_resolve import resolve_supervise_empty_label_ids

    path = tmp_path / "elbow.yml"
    path.write_text(yaml.safe_dump({
        "dataset_name": "ElbowCT", "skip_subjects": None,
        "stored_labels": {0: "background", 1: "humeri"},
    }))
    config = {
        "protocol": {"name": PROTOCOL_NAME},
        "data": {"CT": {"ElbowCT": {"dataset_spec": str(path)}}},
    }
    assert resolve_supervise_empty_label_ids(config, ["ElbowCT"]) == {}
