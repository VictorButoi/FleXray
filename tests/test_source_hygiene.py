"""Structural tests for the documented FleXray source conventions."""

from __future__ import annotations

import ast
import re
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "fxr"


def _python_sources() -> tuple[Path, ...]:
    """Return all FleXray Python sources in stable path order."""

    return tuple(sorted(_SOURCE_ROOT.rglob("*.py")))


_METHOD_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
_SECTION_HEADING = re.compile(r"^[A-Za-z][A-Za-z ]*:$")
_DOCUMENTED_NAME = re.compile(r"^\s*\*{0,2}([A-Za-z_]\w*)(?:\s*\([^\n]*\))?\s*:")


def _direct_methods(
    node: ast.ClassDef,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
    """Return methods defined directly on one class."""

    return tuple(child for child in node.body if isinstance(child, _METHOD_NODES))


def _method_parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...]:
    """Return every documented method parameter except ``self`` and ``cls``."""

    parameters = [
        argument.arg
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
        if argument.arg not in {"self", "cls"}
    ]
    if node.args.vararg is not None:
        parameters.append(node.args.vararg.arg)
    if node.args.kwarg is not None:
        parameters.append(node.args.kwarg.arg)
    return tuple(parameters)


def _doc_section(docstring: str, heading: str) -> str | None:
    """Return one Google-style docstring section without its heading."""

    lines = docstring.splitlines()
    try:
        start = next(
            index + 1
            for index, line in enumerate(lines)
            if line.strip() == f"{heading}:"
        )
    except StopIteration:
        return None

    end = len(lines)
    for index in range(start, len(lines)):
        if _SECTION_HEADING.fullmatch(lines[index].strip()):
            end = index
            break
    return "\n".join(lines[start:end])


def _documented_names(section: str) -> set[str]:
    """Return field names declared by a Google-style docstring section."""

    names: set[str] = set()
    for line in section.splitlines():
        match = _DOCUMENTED_NAME.match(line)
        if match is not None:
            names.add(match.group(1))
    return names


def _decorator_name(node: ast.expr) -> str | None:
    """Return the terminal name of a decorator expression."""

    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _collect_instance_assignment(target: ast.expr, names: set[str]) -> None:
    """Collect direct ``self.<name>`` assignments from one assignment target."""

    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        names.add(target.attr)
    elif isinstance(target, (ast.List, ast.Tuple)):
        for element in target.elts:
            _collect_instance_assignment(element, names)


def _represented_attributes(node: ast.ClassDef) -> tuple[str, ...]:
    """Return statically represented fields, instance attributes, and properties."""

    names: set[str] = set()
    for child in node.body:
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            names.add(child.target.id)
        if not isinstance(child, _METHOD_NODES):
            continue
        if any(
            _decorator_name(decorator) in {"property", "cached_property"}
            for decorator in child.decorator_list
        ):
            names.add(child.name)
        for descendant in ast.walk(child):
            if isinstance(descendant, ast.Assign):
                for target in descendant.targets:
                    _collect_instance_assignment(target, names)
            elif isinstance(descendant, (ast.AnnAssign, ast.AugAssign)):
                _collect_instance_assignment(descendant.target, names)
    return tuple(sorted(names))


def test_every_source_class_and_method_has_a_docstring() -> None:
    """Require documentation for every source class and direct class method."""

    missing: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if ast.get_docstring(node) is None:
                missing.append(
                    f"{path.relative_to(_SOURCE_ROOT)}:{node.lineno} {node.name}"
                )
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if ast.get_docstring(child) is None:
                        missing.append(
                            f"{path.relative_to(_SOURCE_ROOT)}:{child.lineno} "
                            f"{node.name}.{child.name}"
                        )

    assert not missing, "Missing class or method docstrings:\n" + "\n".join(missing)


def test_method_docstrings_document_arguments_and_returns() -> None:
    """Require complete argument and return contracts for every method."""

    violations: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative_path = path.relative_to(_SOURCE_ROOT)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for method in _direct_methods(node):
                docstring = ast.get_docstring(method)
                if docstring is None:
                    continue
                qualified_name = (
                    f"{relative_path}:{method.lineno} {node.name}.{method.name}"
                )
                parameters = _method_parameters(method)
                args_section = _doc_section(docstring, "Args")
                if parameters and args_section is None:
                    violations.append(f"{qualified_name} missing Args section")
                elif args_section is not None:
                    missing_parameters = sorted(
                        set(parameters) - _documented_names(args_section)
                    )
                    if missing_parameters:
                        violations.append(
                            f"{qualified_name} undocumented parameter(s): "
                            + ", ".join(missing_parameters)
                        )
                if _doc_section(docstring, "Returns") is None:
                    violations.append(f"{qualified_name} missing Returns section")

    assert not violations, "Incomplete method docstrings:\n" + "\n".join(violations)


def test_class_docstrings_document_represented_attributes() -> None:
    """Require every statically represented class attribute to be documented."""

    violations: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative_path = path.relative_to(_SOURCE_ROOT)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            represented = _represented_attributes(node)
            if not represented:
                continue
            docstring = ast.get_docstring(node)
            if docstring is None:
                continue
            qualified_name = f"{relative_path}:{node.lineno} {node.name}"
            attributes_section = _doc_section(docstring, "Attributes")
            if attributes_section is None:
                violations.append(f"{qualified_name} missing Attributes section")
                continue
            missing_attributes = sorted(
                set(represented) - _documented_names(attributes_section)
            )
            if missing_attributes:
                violations.append(
                    f"{qualified_name} undocumented attribute(s): "
                    + ", ".join(missing_attributes)
                )

    assert not violations, "Incomplete class docstrings:\n" + "\n".join(violations)


def test_package_initializers_are_reexport_only() -> None:
    """Allow only imports, a module docstring, and ``__all__`` declarations."""

    violations: list[str] = []
    for path in sorted(_SOURCE_ROOT.rglob("__init__.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            if (
                isinstance(node, ast.Assign)
                and all(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in node.targets
                )
                and isinstance(node.value, (ast.List, ast.Tuple))
                and all(
                    isinstance(element, ast.Constant) and isinstance(element.value, str)
                    for element in node.value.elts
                )
            ):
                continue
            violations.append(
                f"{path.relative_to(_SOURCE_ROOT)}:{node.lineno} "
                f"{type(node).__name__}"
            )

    assert not violations, "Substantive package initializer statements:\n" + "\n".join(
        violations
    )
