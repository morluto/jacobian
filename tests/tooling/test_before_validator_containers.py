"""Guard strict-JSON container handling in owner-local preflight validators."""

from __future__ import annotations

import ast
import re
from pathlib import Path


def _is_before_validator(decorator: ast.expr) -> bool:
    return (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Name)
        and decorator.func.id == "model_validator"
        and any(
            keyword.arg == "mode"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "before"
            for keyword in decorator.keywords
        )
    )


def _uses_canonical_container_projection(function: ast.FunctionDef) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "canonicalize_json_containers"
        for node in ast.walk(function)
    )


def test_math_before_validators_project_json_arrays_to_canonical_tuples() -> None:
    """Preflight validators cannot hand raw JSON arrays back to Pydantic.

    ``model_validator(mode="before")`` switches downstream tuple validation
    to Python semantics.  Each owner therefore projects strict-JSON arrays
    to its canonical tuple values before returning to Pydantic.
    """

    source_root = Path(__file__).parents[2] / "src" / "jacobian" / "math"
    missing = [
        f"{path.relative_to(source_root)}:{function.lineno}"
        for path in sorted(source_root.rglob("*.py"))
        for function in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(function, ast.FunctionDef)
        and any(_is_before_validator(item) for item in function.decorator_list)
        and not _uses_canonical_container_projection(function)
    ]

    assert not missing, (
        "mode='before' validators must call canonicalize_json_containers: "
        + ", ".join(missing)
    )


_CONTAINER_PARTS = ("tuple[", "list[", "sequence[", "frozenset[", "set[")
_MAPPING_PARTS = ("dict[", "mapping[", "defaultdict[")


def _module_aliases(root: Path) -> dict[str, set[str]]:
    """Every module-level ``Name = ...`` right-hand side, by name."""

    aliases: dict[str, set[str]] = {}
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and node.value is not None:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        aliases.setdefault(target.id, set()).add(
                            ast.unparse(node.value)
                        )
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
            ):
                aliases.setdefault(node.target.id, set()).add(ast.unparse(node.value))
    return aliases


def _enforced_container_field(source: str, aliases: dict[str, set[str]]) -> bool:
    """Whether a field annotation is a sequence container needing projection.

    Every tuple/list container must reach strict validation as a tuple, so a
    before-validator must project each one (or sit under a total projection).
    Mappings validate from JSON objects directly; arrays nested inside one
    belong to the nested validator. A bare alias resolving uniquely to a
    container counts. Anything else is left unenforced rather than guessed at.
    """

    low = source.lower()
    if not any(part in low for part in _CONTAINER_PARTS):
        bare = source.strip()
        if not re.fullmatch(r"[A-Za-z_]\w*", bare):
            return False
        resolved = aliases.get(bare, set())
        if len(resolved) != 1:
            return False
        return _enforced_container_field(next(iter(resolved)), aliases)
    return not any(part in low for part in _MAPPING_PARTS)


def _loop_variables(generators: list[ast.comprehension]) -> set[str]:
    names: set[str] = set()
    for generator in generators:
        for node in ast.walk(generator.target):
            if isinstance(node, ast.Name):
                names.add(node.id)
    return names


def _model_fields_drives_keys(node: ast.AST) -> bool:
    """Whether a comprehension projects exactly the model's declared fields.

    Total-driving shape: the keys come from ``model_fields`` membership, with
    no narrowing comparison on the loop variable. A filter such as
    ``if key == "old"`` narrows and does not count, and neither does a mere
    mention elsewhere.
    """

    if not isinstance(
        node, (ast.DictComp, ast.SetComp, ast.ListComp, ast.GeneratorExp)
    ):
        return False
    loop_variables = _loop_variables(node.generators)
    if not loop_variables:
        return False

    def mentions_model_fields(item: ast.AST) -> bool:
        return any(
            (isinstance(part, ast.Name) and part.id == "model_fields")
            or (isinstance(part, ast.Attribute) and part.attr == "model_fields")
            for part in ast.walk(item)
        )

    driven = any(
        mentions_model_fields(generator.iter)
        or any(mentions_model_fields(part) for part in generator.ifs)
        for generator in node.generators
    )
    if not driven:
        return False
    for generator in node.generators:
        for condition in generator.ifs:
            for part in ast.walk(condition):
                if not isinstance(part, ast.Compare):
                    continue
                left_names = {
                    item.id
                    for item in ast.walk(part.left)
                    if isinstance(item, ast.Name)
                }
                if not (left_names & loop_variables):
                    continue
                if any(isinstance(action, (ast.Eq, ast.NotEq)) for action in part.ops):
                    return False
                if any(
                    isinstance(action, (ast.In, ast.NotIn))
                    and not mentions_model_fields(part.comparators[0])
                    for action in part.ops
                ):
                    return False
    return True


def _total_projection_names(function: ast.FunctionDef, params: set[str]) -> set[str]:
    """Local names bound to a total projection of the validator input."""

    bound = set(params)
    for _ in range(4):
        grew = False
        for node in ast.walk(function):
            if isinstance(node, ast.Assign):
                targets: list[ast.expr] = list(node.targets)
                value: ast.expr | None = node.value
            elif isinstance(node, ast.AnnAssign):
                targets, value = [node.target], node.value
            else:
                continue
            if value is None:
                continue
            whole = isinstance(value, ast.Name) and value.id in bound
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                whole = value.func.id == "dict" and any(
                    isinstance(argument, ast.Name) and argument.id in bound
                    for argument in value.args
                )
            if isinstance(value, ast.Dict):
                whole = any(
                    key is None and isinstance(item, ast.Name) and item.id in bound
                    for key, item in zip(value.keys, value.values, strict=True)
                )
            driven = _model_fields_drives_keys(value)
            if whole or driven:
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in bound:
                        bound.add(target.id)
                        grew = True
        if not grew:
            break
    return bound


def _reachable_functions(
    validator: ast.FunctionDef, functions: dict[str, ast.FunctionDef]
) -> list[ast.FunctionDef]:
    """The validator plus the module-local helpers it calls, transitively."""

    group = [validator]
    seen = {id(validator)}
    frontier = [validator]
    while frontier:
        current = frontier.pop()
        for call in ast.walk(current):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                target = functions.get(call.func.id)
                if target is not None and id(target) not in seen:
                    seen.add(id(target))
                    group.append(target)
                    frontier.append(target)
    return group


def _validator_covers_fields(
    validator: ast.FunctionDef,
    functions: dict[str, ast.FunctionDef],
    fields: set[str],
) -> set[str]:
    """Fields neither under a total projection nor named in reach."""

    group = _reachable_functions(validator, functions)
    params = {argument.arg for argument in validator.args.args}
    total = False
    for function in group:
        bound = _total_projection_names(function, params)
        for call in ast.walk(function):
            if not (
                isinstance(call, ast.Call)
                and getattr(call.func, "id", None) == "canonicalize_json_containers"
            ):
                continue
            if any(
                isinstance(argument, ast.Name) and argument.id in bound
                for argument in call.args
            ):
                total = True
            if any(_model_fields_drives_keys(argument) for argument in call.args):
                total = True
    if total:
        return set()
    named = {
        part.value
        for function in group
        for part in ast.walk(function)
        if isinstance(part, ast.Constant) and isinstance(part.value, str)
    }
    return fields - named


def test_before_validators_cover_every_sequence_container_field() -> None:
    """A before-validator projection must reach every sequence container field.

    ``parse_operation_input`` validates strictly, so a declared array field
    left as a raw JSON array refuses the whole request with ``tuple_type`` —
    including the outer array of a nested-model container, whose child
    validators never run until the parent level validates. Whole-payload and
    ``cls.model_fields``-driven projections are total by construction; every
    other field must be named in the validator's reachable projection code.
    Mappings validate from JSON objects directly and stay exempt. A
    ``model_fields`` mention only counts when it drives the projected keys
    through a comprehension membership filter with no narrowing comparison.
    Naming is deliberately broad: a literal that only reads a field without
    converting it still counts, because restricting names to
    canonicalize-flow flags suite-pinned correct code whose conversion
    happens in hand-rolled tuple steps and leaf delegation. A
    read-without-convert gap still refuses loudly under strict dispatch, so
    omission — the realistic regression — is what this gate owns.
    """

    source_root = Path(__file__).parents[2] / "src" / "jacobian" / "math"
    aliases = _module_aliases(source_root)
    violations: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions: dict[str, ast.FunctionDef] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.setdefault(node.name, node)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            before = [
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef)
                and any(_is_before_validator(item) for item in item.decorator_list)
            ]
            if not before:
                continue
            fields = {
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
                and _enforced_container_field(ast.unparse(item.annotation), aliases)
            }
            if not fields:
                continue
            for validator in before:
                uncovered = _validator_covers_fields(validator, functions, fields)
                for name in sorted(uncovered):
                    violations.append(
                        f"{path.relative_to(source_root)}:{validator.lineno} "
                        f"{node.name}.{name} is never projected"
                    )

    assert not violations, (
        "before-validator projections must reach every sequence container "
        "field; project the payload, derive ownership from cls.model_fields, "
        "or name the field in reachable projection code: "
        + ", ".join(sorted(violations))
    )
