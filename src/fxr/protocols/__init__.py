"""Strict FleXray protocol loaders and compilers.

This module re-exports the public protocol-core schemas and named loader/compiler
helpers. Packaged YAML configs live in the top-level `fxr.configs` package.
"""

from .compiler import compile_training_lut
from .introspect import (
    describe_dataset,
    describe_protocol,
    explain_lut,
    explain_mapping,
    list_datasets,
    list_protocols,
)
from .io import load_dataset_spec, load_protocol
from .registry import (
    compile_eval_contract_by_name,
    compile_training_lut_by_name,
    list_dataset_spec_names,
    list_protocol_names,
    load_dataset_spec_by_name,
    load_eval_contract_by_model_name,
    load_model_label_space_by_name,
    load_protocol_by_name,
)
from .run_config import (
    resolve_run_attenuated_label_ids,
    resolve_run_output_label_names,
    resolve_run_protocol_spec,
    resolve_run_render_label_collapse_index,
    resolve_run_render_label_names,
)
from .schemas import DatasetSpec, EvalContract, ModelLabelSpace, ProtocolSpec

__all__ = [
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
    "list_protocols",
    "list_dataset_spec_names",
    "list_protocol_names",
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
]
