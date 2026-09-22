"""Behavior tests for ``fxr-protocol`` and the shared introspection helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

import fxr.mcp.protocol_tools as mcp_tools
import fxr.protocols as protocols
from fxr.protocols import compile_training_lut_by_name, list_dataset_spec_names
from fxr.protocols.cli import main as protocol_main

_PROTOCOL = "all_structures_flexray_v4"
_CUSTOM_SPEC = """\
dataset_name: CustomHips
skip_subjects: null
stored_labels:
  0: background
  1: hip_left
  2: hip_right
  3: femurs
  4: implant
protocol_label_aliases:
  all_structures_flexray_v4:
    hip_left: hips
    hip_right: hips
protocol_drop_labels:
  all_structures_flexray_v4:
    - implant
"""


def test_list_prints_packaged_protocols_and_datasets(capsys) -> None:
    assert protocol_main(["list"]) == 0
    lines = capsys.readouterr().out.splitlines()

    assert lines[0] == "protocols (1):"
    assert lines[1] == f"  {_PROTOCOL}  63 labels"
    assert lines[2] == f"dataset specs ({len(list_dataset_spec_names())}):"
    assert "  MOOSE  aliases: FluXray" in lines
    assert len(lines) == 3 + len(list_dataset_spec_names())


def test_show_prints_channels_in_order(capsys) -> None:
    assert protocol_main(["show", _PROTOCOL]) == 0
    lines = capsys.readouterr().out.splitlines()

    assert lines[0] == f"{_PROTOCOL}: 63 labels"
    assert lines[1] == "    0  background"
    assert lines[-2] == "   62  heart"
    assert lines[-1] == (
        "note: channels are protocol positions; a model bundle's output channels "
        "follow its label_schema.json"
    )


def test_compile_by_name_matches_compiled_lut(capsys) -> None:
    assert protocol_main(["compile", "--dataset", "HipRay"]) == 0
    out = capsys.readouterr().out

    lut = list(compile_training_lut_by_name(_PROTOCOL, "HipRay").label_lut)
    assert out.splitlines()[0] == f"HipRay -> {_PROTOCOL}"
    assert out.rstrip().endswith(f"label_lut: {lut}")
    assert "identity" in out


def test_compile_from_spec_path_reports_alias_and_drop_rules(tmp_path: Path, capsys) -> None:
    spec = tmp_path / "CustomHips.yml"
    spec.write_text(_CUSTOM_SPEC, encoding="utf-8")

    assert protocol_main(["compile", "--dataset", str(spec)]) == 0
    lines = capsys.readouterr().out.splitlines()

    assert lines[0] == f"CustomHips -> {_PROTOCOL}"
    rules = {line.split()[1]: line.split()[2] for line in lines[2:7]}
    assert rules == {
        "background": "background",
        "hip_left": "alias",
        "hip_right": "alias",
        "femurs": "identity",
        "implant": "drop",
    }
    assert lines[-1] == "label_lut: [0, 54, 54, 11, 0]"


def test_unknown_or_invalid_names_exit_with_parser_error(capsys) -> None:
    for argv in (["show", "nope"], ["compile", "--dataset", "../MOOSE"]):
        with pytest.raises(SystemExit) as exc_info:
            protocol_main(argv)
        assert exc_info.value.code == 2
        assert "Traceback" not in capsys.readouterr().err


def test_explain_lut_rules_match_spec_and_mcp_reuses_introspect() -> None:
    explained = protocols.explain_mapping(_PROTOCOL, "MOOSE")
    by_label = {row["native_label"]: row for row in explained["rows"]}

    assert by_label["background"]["rule"] == "background"
    assert by_label["vertebra_l6"]["rule"] == "drop"
    assert by_label["vertebra_l6"]["protocol_label"] == "background"
    assert mcp_tools.explain_mapping is protocols.explain_mapping
