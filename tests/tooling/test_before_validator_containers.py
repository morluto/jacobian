"""Guard strict-JSON container handling in owner-local preflight validators."""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest


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
    missing: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Before-validators are module-level functions or class methods, so
        # scanning those two levels avoids walking every statement of every
        # module in the tree.
        for owner in tree.body:
            candidates = (
                owner.body
                if isinstance(owner, ast.ClassDef)
                else [owner]
                if isinstance(owner, ast.FunctionDef)
                else ()
            )
            for function in candidates:
                if (
                    isinstance(function, ast.FunctionDef)
                    and any(
                        _is_before_validator(item) for item in function.decorator_list
                    )
                    and not _uses_canonical_container_projection(function)
                ):
                    missing.append(f"{path.relative_to(source_root)}:{function.lineno}")
        del tree

    assert not missing, (
        "mode='before' validators must call canonicalize_json_containers: "
        + ", ".join(missing)
    )


def _declared_aliases(tree: ast.Module) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.TypeAlias):
            aliases[node.name.id] = {ast.unparse(node.value)}
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    aliases[target.id] = {ast.unparse(node.value)}
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            aliases[node.target.id] = {ast.unparse(node.value)}
    return aliases


def _expand_alias(
    source: str, aliases: dict[str, set[str]], seen: frozenset[str] = frozenset()
) -> str:
    class Expand(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            definitions = aliases.get(node.id, set())
            if node.id in seen or len(definitions) != 1:
                return node
            return ast.parse(
                _expand_alias(next(iter(definitions)), aliases, seen | {node.id}),
                mode="eval",
            ).body

    return ast.unparse(Expand().visit(ast.parse(source, mode="eval").body))


def _resolved_annotation(source: str, aliases: dict[str, set[str]]) -> ast.expr:
    """One expanded annotation grammar for field ownership and array depth."""

    class Normalize(ast.NodeTransformer):
        def __init__(self) -> None:
            self.quoted: set[str] = set()

        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            if not isinstance(node.value, str) or node.value in self.quoted:
                return node
            self.quoted.add(node.value)
            try:
                parsed = ast.parse(_expand_alias(node.value, aliases), mode="eval").body
            except SyntaxError:
                return node
            result = self.visit(parsed)
            assert isinstance(result, ast.AST)
            self.quoted.remove(node.value)
            return result

        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            self.generic_visit(node)
            name = getattr(node.value, "id", getattr(node.value, "attr", ""))
            if name.lower() in {"tuple", "list", "sequence", "set", "frozenset"}:
                node.value = ast.Name(id=name.lower(), ctx=ast.Load())
            elif name in {"Annotated", "Optional", "Union"}:
                node.value = ast.Name(id=name, ctx=ast.Load())
            return node

    result = Normalize().visit(
        ast.parse(_expand_alias(source, aliases), mode="eval").body
    )
    assert isinstance(result, ast.expr)
    return result


def _import_path(node: ast.ImportFrom, path: Path, root: Path) -> Path | None:
    if node.level:
        directory = path.parent
        for _ in range(node.level - 1):
            directory = directory.parent
        stem = directory.joinpath(*(node.module or "").split("."))
    else:
        module = (node.module or "").removeprefix("jacobian.math.")
        stem = root.joinpath(*module.split("."))
    candidate = stem.with_suffix(".py")
    if not candidate.is_file():
        candidate = stem / "__init__.py"
    return candidate if candidate.is_file() else None


def _module_functions(
    path: Path,
    root: Path,
    trees: dict[Path, ast.Module],
    cache: dict[Path, dict[str, ast.FunctionDef]],
) -> dict[str, ast.FunctionDef]:
    if path in cache:
        return cache[path]
    tree = trees[path] if path in trees else ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    cache[path] = functions
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and (source := _import_path(node, path, root)) is not None
        ):
            imported = _module_functions(source, root, trees, cache)
            for name in node.names:
                if name.name in imported:
                    functions.setdefault(name.asname or name.name, imported[name.name])
    return functions


def _module_classes(
    path: Path,
    root: Path,
    trees: dict[Path, ast.Module],
    cache: dict[Path, dict[str, ast.ClassDef]],
) -> dict[str, ast.ClassDef]:
    if path in cache:
        return cache[path]
    tree = trees[path] if path in trees else ast.parse(path.read_text(encoding="utf-8"))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    cache[path] = classes
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and (source := _import_path(node, path, root)) is not None
        ):
            imported = _module_classes(source, root, trees, cache)
            for name in node.names:
                if name.name in imported:
                    classes.setdefault(name.asname or name.name, imported[name.name])
    return classes


def _module_aliases(
    path: Path,
    root: Path,
    cache: dict[Path, dict[str, set[str]]],
    trees: dict[Path, ast.Module] | None = None,
) -> dict[str, set[str]]:
    """Resolve explicit imports in their source module before applying local names."""

    if path in cache:
        return cache[path]
    cache[path] = {}
    tree = (
        trees[path]
        if trees is not None and path in trees
        else ast.parse(path.read_text(encoding="utf-8"))
    )
    aliases: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        imported_path = _import_path(node, path, root)
        if imported_path is None:
            continue
        imported = _module_aliases(imported_path, root, cache, trees)
        for name in node.names:
            if name.name in imported:
                aliases[name.asname or name.name] = {
                    _expand_alias(value, imported, frozenset({name.name}))
                    for value in imported[name.name]
                }
    aliases.update(_declared_aliases(tree))
    cache[path] = aliases
    return aliases


def _leaf_container_field(
    source: str, classes: set[str], aliases: dict[str, set[str]]
) -> bool:
    """Recognize outer arrays, including arrays of models, through aliases.

    A bare nested model delegates to its owner; its parent's outer array does
    not. Unknown and cyclic aliases are left unenforced.
    """

    def visit(node: ast.expr, seen: frozenset[str]) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return visit(ast.parse(node.value, mode="eval").body, seen)
        if isinstance(node, ast.Name):
            definitions = aliases.get(node.id, set())
            if node.id in seen or len(definitions) != 1:
                return False
            return visit(
                ast.parse(next(iter(definitions)), mode="eval").body,
                seen | {node.id},
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return visit(node.left, seen) or visit(node.right, seen)
        if isinstance(node, ast.Subscript):
            name = getattr(node.value, "id", getattr(node.value, "attr", ""))
            if name.lower() in {"tuple", "list", "sequence", "frozenset", "set"}:
                return True
            parts = (
                node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            )
            if name == "Annotated":
                return visit(parts[0], seen)
            if name in {"Optional", "Union"}:
                return any(visit(part, seen) for part in parts)
        return False

    return visit(_resolved_annotation(source, aliases), frozenset())


_WHOLE = frozenset({"*"})


def _model_fields(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "cls"
        and node.attr == "model_fields"
    )


def _own_nodes(function: ast.FunctionDef) -> list[ast.AST]:
    """Nodes of one function body, excluding nested scopes.

    Returns, assignments, and calls inside an uncalled nested definition do
    not belong to the validator's own execution. Nested helpers are still
    analyzed when a reachable call invokes them; only their uninvoked bodies
    are out of scope here.
    """

    scoped: list[ast.AST] = []
    frontier: list[ast.AST] = [function]
    while frontier:
        node = frontier.pop()
        scoped.append(node)
        for child in ast.iter_child_nodes(node):
            if (
                isinstance(
                    child,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
                )
                and child is not function
            ):
                continue
            frontier.append(child)
    return scoped


def _write_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return "name:" + node.id
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return "item:" + node.value.id + ":" + repr(node.slice.value)
    return None


def _shadowed_assignments(nodes: list[ast.AST]) -> set[int]:
    """Assignments whose write cannot reach any return.

    An earlier write to a name or literal mapping key is dead only when a
    later write to that same location sits in the same straight-line statement list with no return, loop,
    branch, or exception boundary between them. Anything else keeps all
    writes: branches may each serve a different return, and loops may skip.
    """

    parents: dict[int, ast.AST] = {}
    for node in nodes:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                parents.setdefault(id(child), node)
    lists: dict[int, list[ast.stmt]] = {}
    for node in nodes:
        if not isinstance(node, ast.stmt):
            continue
        parent = parents.get(id(node))
        if parent is None:
            continue
        lists.setdefault(id(parent), []).append(node)
    dead: set[int] = set()
    for statements in lists.values():
        seen: dict[str, ast.stmt] = {}
        for statement in sorted(statements, key=lambda item: item.lineno):
            if isinstance(
                statement,
                (
                    ast.Return,
                    ast.If,
                    ast.For,
                    ast.While,
                    ast.Try,
                    ast.Match,
                    ast.With,
                    ast.AsyncFor,
                    ast.AsyncWith,
                ),
            ):
                seen.clear()
                continue
            reads = {
                _write_key(read)
                for read in ast.walk(statement)
                if isinstance(read, (ast.Name, ast.Subscript))
                and isinstance(read.ctx, ast.Load)
            }
            seen = {name: write for name, write in seen.items() if name not in reads}
            if isinstance(statement, ast.Assign):
                targets: list[ast.expr] = list(statement.targets)
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                targets = [statement.target]
            else:
                continue
            for target in targets:
                key = _write_key(target)
                if key is not None:
                    if key in seen:
                        dead.add(id(seen[key]))
                    seen[key] = statement
    return dead


def _target_values(
    target: ast.expr, value: ast.expr | None
) -> list[tuple[ast.expr, ast.expr | None]]:
    if isinstance(target, (ast.Tuple, ast.List)):
        if any(isinstance(child, ast.Starred) for child in target.elts):
            return [
                pair for child in target.elts for pair in _target_values(child, None)
            ]
        matched = isinstance(value, (ast.Tuple, ast.List)) and len(target.elts) == len(
            value.elts
        )
        pairs: list[tuple[ast.expr, ast.expr | None]] = []
        for index, child in enumerate(target.elts):
            item = (
                value.elts[index]
                if matched and isinstance(value, (ast.Tuple, ast.List))
                else ast.copy_location(
                    ast.Subscript(
                        value=value, slice=ast.Constant(index), ctx=ast.Load()
                    ),
                    value,
                )
                if value is not None
                else None
            )
            pairs.extend(_target_values(child, item))
        return pairs
    if isinstance(target, ast.Starred):
        return _target_values(target.value, None)
    return [(target, value)]


def _assignment_pairs(node: ast.AST) -> list[tuple[ast.expr, ast.expr | None]]:
    if isinstance(node, ast.Assign):
        return [
            pair
            for target in node.targets
            for pair in _target_values(target, node.value)
        ]
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return _target_values(node.target, node.value)
    return []


@dataclass
class _ModelShape:
    types: dict[str, frozenset[str]] = field(default_factory=dict)
    max_lengths: dict[str, str] = field(default_factory=dict)
    model_names: frozenset[str] = frozenset()


def _declared_max_lengths(owner: ast.ClassDef, prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for declaration in owner.body:
        if (
            isinstance(declaration, ast.AnnAssign)
            and isinstance(declaration.target, ast.Name)
            and isinstance(declaration.value, ast.Call)
            and getattr(declaration.value.func, "id", "") == "Field"
        ):
            for keyword in declaration.value.keywords:
                if keyword.arg == "max_length":
                    result[prefix + declaration.target.id] = ast.dump(keyword.value)
    return result


def _json_shapes(
    model: ast.ClassDef,
    classes: dict[str, ast.ClassDef],
    contexts: dict[ast.ClassDef, dict[str, set[str]]],
) -> _ModelShape:
    """Known JSON shapes of declared fields, including nested model fields."""

    shapes = _ModelShape(
        model_names=frozenset(
            name
            for name, owner in classes.items()
            if any(
                getattr(base, "id", "") in {"StrictModel", "BaseModel"}
                for base in owner.bases
            )
        )
    )

    def annotation(node: ast.expr, path: str, seen: frozenset[str]) -> frozenset[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return annotation(ast.parse(node.value, mode="eval").body, path, seen)
        if isinstance(node, ast.Subscript):
            name = getattr(node.value, "id", getattr(node.value, "attr", ""))
            parts = (
                node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            )
            if name == "Annotated":
                return annotation(parts[0], path, seen)
            if name.lower() in {"tuple", "list", "sequence", "set", "frozenset"}:
                elements = (
                    [
                        part
                        for part in parts
                        if not (
                            isinstance(part, ast.Constant) and part.value is Ellipsis
                        )
                    ]
                    if name.lower() == "tuple"
                    else parts[:1]
                )
                shapes.types[path + "[]"] = frozenset().union(
                    *(annotation(part, path + "[]", seen) for part in elements)
                )
                return frozenset({"list"})
            if name in {"Union", "Optional"}:
                result = frozenset().union(
                    *(annotation(part, path, seen) for part in parts)
                )
                return result | ({"NoneType"} if name == "Optional" else set())
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return annotation(node.left, path, seen) | annotation(
                node.right, path, seen
            )
        if isinstance(node, ast.Constant) and node.value is None:
            return frozenset({"NoneType"})
        name = getattr(node, "id", "")
        primitive = {
            "StrictInt": "int",
            "StrictBool": "bool",
            "StrictStr": "str",
            "StrictFloat": "float",
        }.get(name, name)
        if primitive in {"int", "bool", "str", "float"}:
            return frozenset({primitive})
        if name in classes and name not in seen:
            visit(classes[name], path + ".", seen | {name})
            return frozenset({"dict"})
        return frozenset()

    def visit(owner: ast.ClassDef, prefix: str, seen: frozenset[str]) -> None:
        aliases = contexts.get(owner, {})
        shapes.max_lengths.update(_declared_max_lengths(owner, prefix))
        for name, source in _inherited_fields(owner, classes).items():
            shapes.types[prefix + name] = annotation(
                _resolved_annotation(source, aliases), prefix + name, seen
            )

    visit(model, "", frozenset({model.name}))
    return shapes


class _Projection:
    """Bounded syntactic data flow; not a proof of all execution paths."""

    def __init__(
        self,
        function: ast.FunctionDef,
        functions: dict[str, ast.FunctionDef],
        arguments: dict[str, frozenset[str]],
        covered: set[str],
        stack: frozenset[str],
        required_fields: frozenset[str] = frozenset(),
        shapes: _ModelShape | None = None,
    ) -> None:
        scoped = _own_nodes(function)
        dead = _shadowed_assignments(scoped)
        self.nodes = [node for node in scoped if id(node) not in dead]
        self.write_bindings = [
            (node, target, value)
            for node in self.nodes
            if not isinstance(node, ast.AugAssign)
            for target, value in _assignment_pairs(node)
        ]
        self.loop_controls = [
            node for node in self.nodes if isinstance(node, (ast.Break, ast.Continue))
        ]
        self.unmodeled_bindings: set[str] = set()
        for node in self.nodes:
            targets: list[ast.AST] = []
            if isinstance(node, (ast.For, ast.AsyncFor, ast.NamedExpr)):
                targets.append(node.target)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                targets.extend(
                    item.optional_vars
                    for item in node.items
                    if item.optional_vars is not None
                )
            elif isinstance(node, (ast.MatchAs, ast.MatchStar, ast.ExceptHandler)):
                if node.name is not None:
                    self.unmodeled_bindings.add(node.name)
            elif isinstance(node, ast.MatchMapping) and node.rest is not None:
                self.unmodeled_bindings.add(node.rest)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self.unmodeled_bindings.update(
                    alias.asname or alias.name.split(".")[0] for alias in node.names
                )
            self.unmodeled_bindings.update(
                part.id
                for target in targets
                for part in ast.walk(target)
                if isinstance(part, ast.Name)
            )
        self.local_names = (
            self.unmodeled_bindings
            | {
                node.arg
                for node in ast.walk(function.args)
                if isinstance(node, ast.arg)
            }
            | {
                part.id
                for _, target, _ in self.write_bindings
                for part in ast.walk(target)
                if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store)
            }
        )
        self.shadowed_names = self.local_names | set(functions)
        self.functions = functions
        self.arguments = arguments
        self.covered = covered
        self.stack = stack
        self.required_fields = required_fields
        self.shapes = shapes or _ModelShape()
        self.live_cache: set[ast.AST] | None = None
        self.origin_cache: dict[ast.expr, frozenset[str]] = {}
        self.projection_cache: dict[tuple[ast.expr, str], bool] = {}
        self.path_cache: dict[tuple[ast.AST, str], bool] = {}
        self.effect_cache: dict[tuple[ast.Call, str], bool] = {}
        self.canonical_arguments: set[str] = set()
        self.checking_guards = False
        self.return_filter: ast.Return | None = None
        self.parents = {
            child: parent for parent in scoped for child in ast.iter_child_nodes(parent)
        }

    def keys(self, node: ast.expr) -> frozenset[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return frozenset({node.value})
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return frozenset().union(*(self.keys(item) for item in node.elts))
        if isinstance(node, ast.Name):
            return frozenset().union(
                *(
                    self.keys(part.iter)
                    for part in self.nodes
                    if isinstance(part, ast.For)
                    and isinstance(part.target, ast.Name)
                    and part.target.id == node.id
                    and part.lineno < node.lineno <= (part.end_lineno or part.lineno)
                )
            )
        return frozenset()

    def name_binding(self, node: ast.Name) -> tuple[ast.AST | None, ast.expr | None]:
        bindings = [
            (statement, value)
            for statement, target, value in self.write_bindings
            if isinstance(target, ast.Name)
            and target.id == node.id
            and (
                getattr(statement, "end_lineno", 0),
                getattr(statement, "end_col_offset", 0),
            )
            < (node.lineno, node.col_offset)
        ]
        return (
            max(
                bindings,
                key=lambda item: (
                    getattr(item[0], "lineno", 0),
                    getattr(item[0], "col_offset", 0),
                ),
            )
            if bindings
            else (None, None)
        )

    def binding(self, node: ast.Name, seen: frozenset[str]) -> frozenset[str]:
        statement, value = self.name_binding(node)
        if statement is not None:
            return (
                self.value(value, seen | {node.id})
                if value is not None
                else frozenset()
            )
        return self.arguments.get(node.id, frozenset())

    def comprehension(self, node: ast.DictComp, seen: frozenset[str]) -> frozenset[str]:
        if len(node.generators) != 1:
            return frozenset()
        generator = node.generators[0]
        # {key: data[key] for key in cls.model_fields}, without subset filters.
        if (
            not generator.ifs
            and isinstance(generator.target, ast.Name)
            and ast.dump(node.key)
            == ast.dump(generator.target).replace("Store()", "Load()")
            and isinstance(node.value, ast.Subscript)
            and ast.dump(node.value.slice) == ast.dump(node.key)
            and self.value(node.value.value, seen) == _WHOLE
            and _model_fields(generator.iter)
        ):
            return _WHOLE
        # {key: item for key, item in data.items() if key in cls.model_fields}.
        if not (
            isinstance(generator.target, ast.Tuple)
            and len(generator.target.elts) == 2
            and ast.dump(node.key)
            == ast.dump(generator.target.elts[0], include_attributes=False).replace(
                "Store()", "Load()"
            )
            and ast.dump(node.value)
            == ast.dump(generator.target.elts[1], include_attributes=False).replace(
                "Store()", "Load()"
            )
            and isinstance(generator.iter, ast.Call)
            and isinstance(generator.iter.func, ast.Attribute)
            and generator.iter.func.attr == "items"
            and not generator.iter.args
            and self.value(generator.iter.func.value, seen) == _WHOLE
            and len(generator.ifs) == 1
        ):
            return frozenset()
        condition = generator.ifs[0]
        if (
            isinstance(condition, ast.Compare)
            and ast.dump(condition.left) == ast.dump(node.key)
            and len(condition.ops) == 1
            and isinstance(condition.ops[0], ast.In)
            and _model_fields(condition.comparators[0])
        ):
            return _WHOLE
        return frozenset()

    def call(self, node: ast.Call, seen: frozenset[str]) -> frozenset[str]:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and self.value(node.func.value, seen) == _WHOLE
            and node.args
        ):
            return self.keys(node.args[0])
        name = getattr(node.func, "id", "")
        if name in {"dict", "canonicalize_json_containers"} and node.args:
            return self.value(node.args[0], seen)
        if name == "cast" and len(node.args) == 2:
            return self.value(node.args[1], seen)
        if name == "tuple" and node.args:
            origin = self.value(node.args[0], seen)
            return frozenset() if origin == _WHOLE else origin
        if name in self.functions and name not in self.stack:
            helper = self.functions[name]
            arguments = {
                parameter.arg: self.value(argument, seen)
                for parameter, argument in zip(
                    (helper.args.posonlyargs + helper.args.args),
                    node.args,
                    strict=False,
                )
            }
            helper_covered: set[str] = set()
            projection = _Projection(
                helper,
                self.functions,
                arguments,
                helper_covered,
                self.stack | {name},
                self.required_fields,
                self.shapes,
            )
            result = projection.analyze()
            if helper_covered and not self.checking_guards:
                self.record(node, self.destinations(node, helper_covered))
            # Bounded materializers return existing JSON arrays unchanged before
            # entering their iterator path. Preserve that explicit fast path.
            if helper.body and isinstance(helper.body[0], ast.If):
                first = helper.body[0]
                if len(first.body) == 1 and isinstance(first.body[0], ast.Return):
                    returned = first.body[0].value
                    if isinstance(returned, ast.Name):
                        origin = arguments.get(returned.id, frozenset())
                        if (
                            len(origin) == 1
                            and projection.guard_value(first.test, next(iter(origin)))
                            is True
                        ):
                            return origin
            return result
        return frozenset()

    def value(
        self, node: ast.expr, seen: frozenset[str] = frozenset()
    ) -> frozenset[str]:
        if self.checking_guards and node in self.origin_cache:
            return self.origin_cache[node]
        result = self._value(node, seen)
        if self.checking_guards:
            self.origin_cache[node] = result
        return result

    def sequence_element(
        self, value: ast.expr, index: int, seen: frozenset[str]
    ) -> frozenset[str]:
        if isinstance(value, (ast.Tuple, ast.List)) and 0 <= index < len(value.elts):
            return self.value(value.elts[index], seen)
        if isinstance(value, ast.Name):
            _, bound = self.name_binding(value)
            return (
                self.sequence_element(bound, index, seen | {value.id})
                if bound is not None and value.id not in seen
                else frozenset()
            )
        if not isinstance(value, ast.Call):
            return frozenset()
        name = getattr(value.func, "id", "")
        if name not in self.functions or name in self.stack:
            return frozenset()
        helper = self.functions[name]
        projection = _Projection(
            helper,
            self.functions,
            {
                parameter.arg: self.value(argument, seen)
                for parameter, argument in zip(
                    (helper.args.posonlyargs + helper.args.args),
                    value.args,
                    strict=False,
                )
            },
            set(),
            self.stack | {name},
            self.required_fields,
            self.shapes,
        )
        returned = [
            projection.sequence_element(node.value, index, frozenset())
            for node in projection.nodes
            if isinstance(node, ast.Return) and node.value is not None
        ]
        return frozenset.intersection(*returned) if returned else frozenset()

    def _value(self, node: ast.expr, seen: frozenset[str]) -> frozenset[str]:
        if isinstance(node, ast.Name):
            return self.binding(node, seen)
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant) and isinstance(
                node.slice.value, int
            ):
                return self.sequence_element(node.value, node.slice.value, seen)
            if self.value(node.value, seen) == _WHOLE:
                return self.keys(node.slice)
        if isinstance(node, ast.Dict):
            result: set[str] = set()
            for key, item in zip(node.keys, node.values, strict=True):
                result.update(self.value(item, seen) if key is None else self.keys(key))
            return frozenset(result)
        if isinstance(node, ast.DictComp):
            return self.comprehension(node, seen)
        if isinstance(node, ast.Call):
            return self.call(node, seen)
        return frozenset()

    def returned_nodes(self) -> set[ast.AST]:
        """Backward slice from returned expressions and writes into their mappings."""

        if self.live_cache is not None:
            return self.live_cache
        live = {
            part
            for node in self.nodes
            if isinstance(node, ast.Return) and node.value is not None
            for part in ast.walk(node.value)
        }
        while True:
            previous = len(live)
            names = {part.id for part in live if isinstance(part, ast.Name)}
            for node in self.nodes:
                if (
                    not isinstance(node, (ast.Assign, ast.AnnAssign))
                    or node.value is None
                ):
                    continue
                for target, assigned in _assignment_pairs(node):
                    if (
                        isinstance(target, ast.Name)
                        and isinstance(assigned, ast.Name)
                        and assigned.id in names
                    ):
                        live.add(target)
                        live.add(node)
                    receiver = (
                        target.value if isinstance(target, ast.Subscript) else target
                    )
                    if isinstance(receiver, ast.Name) and receiver.id in names:
                        live.add(node)
                        live.update(ast.walk(node.value))
            if len(live) == previous:
                self.live_cache = live
                return live

    def shape_origin(
        self, node: ast.expr, seen: frozenset[int] = frozenset()
    ) -> str | None:
        if id(node) in seen:
            return None
        seen = seen | {id(node)}
        if isinstance(node, ast.Name):
            for part in sorted(
                self.nodes, key=lambda part: getattr(part, "lineno", 0), reverse=True
            ):
                if (
                    isinstance(part, (ast.Assign, ast.AnnAssign))
                    and part.value is not None
                ):
                    targets = (
                        part.targets if isinstance(part, ast.Assign) else [part.target]
                    )
                    if (
                        part.value.end_lineno or part.lineno,
                        part.value.end_col_offset or 0,
                    ) < (node.lineno, node.col_offset) and any(
                        isinstance(target, ast.Name) and target.id == node.id
                        for target in targets
                    ):
                        return self.shape_origin(part.value, seen)
                if (
                    isinstance(part, ast.For)
                    and isinstance(part.target, ast.Name)
                    and part.target.id == node.id
                    and part.lineno < node.lineno <= (part.end_lineno or part.lineno)
                ):
                    origin = self.shape_origin(part.iter, seen)
                    return None if origin is None else origin + "[]"
            argument_origin = self.arguments.get(node.id, frozenset())
            return next(iter(argument_origin)) if len(argument_origin) == 1 else None
        if isinstance(node, ast.Subscript):
            origin = self.shape_origin(node.value, seen)
            if origin is None:
                return None
            if isinstance(node.slice, ast.Constant) and isinstance(
                node.slice.value, str
            ):
                return ("" if origin == "*" else origin + ".") + node.slice.value
            return origin + "[]"
        if isinstance(node, ast.Call):
            return self.call_shape_origin(node, seen)
        return None

    def call_shape_origin(self, node: ast.Call, seen: frozenset[int]) -> str | None:
        name = getattr(node.func, "id", "")
        if name in {"canonicalize_json_containers", "dict", "cast"} and node.args:
            return self.shape_origin(node.args[-1], seen)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
        ):
            receiver, key = node.func.value, node.args[0]
        elif (
            name in self.functions
            and len(node.args) == 2
            and self.is_field_reader(self.functions[name])
        ):
            receiver, key = node.args
        else:
            return None
        origin = self.shape_origin(receiver, seen)
        if (
            origin is not None
            and isinstance(key, ast.Constant)
            and isinstance(key.value, str)
        ):
            return ("" if origin == "*" else origin + ".") + key.value
        return None

    @staticmethod
    def is_field_reader(helper: ast.FunctionDef) -> bool:
        if len(helper.args.posonlyargs + helper.args.args) != 2:
            return False
        receiver, key = (
            argument.arg for argument in (helper.args.posonlyargs + helper.args.args)
        )
        returns = [
            node.value for node in _own_nodes(helper) if isinstance(node, ast.Return)
        ]
        expected = {
            ast.dump(ast.parse(f"{receiver}.get({key})", mode="eval").body),
            ast.dump(ast.parse(f"getattr({receiver}, {key}, None)", mode="eval").body),
        }
        return bool(returns) and all(
            value is not None and ast.dump(value) in expected for value in returns
        )

    def declared_types(self, node: ast.expr, field: str) -> frozenset[str]:
        origin = self.shape_origin(node)
        if origin == "*":
            return frozenset({"dict"})
        if origin == field:
            return frozenset({"list"})
        declared = self.shapes.types.get(origin or "", frozenset())
        if declared:
            return declared
        traced = self.value(node)
        return (
            frozenset({"dict"})
            if traced == _WHOLE
            else frozenset({"list"})
            if traced == frozenset({field})
            else frozenset()
        )

    def guard_value(self, test: ast.expr, field: str) -> bool | None:
        """Evaluate only structural guards known for a present JSON array field."""

        guard_symbols = {
            "isinstance",
            "type",
            "len",
            "dict",
            "list",
            "tuple",
            "set",
            "frozenset",
            "str",
            "int",
            "float",
            "bool",
            "Mapping",
        }
        if any(
            isinstance(node, ast.Name)
            and node.id in guard_symbols
            and node.id in self.shadowed_names
            for node in ast.walk(test)
        ):
            return None
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            value = self.guard_value(test.operand, field)
            return None if value is None else not value
        if isinstance(test, ast.BoolOp):
            values = [self.guard_value(part, field) for part in test.values]
            if isinstance(test.op, ast.And):
                return (
                    False
                    if False in values
                    else True
                    if all(value is True for value in values)
                    else None
                )
            return (
                True
                if True in values
                else False
                if all(value is False for value in values)
                else None
            )
        if (
            isinstance(test, ast.Call)
            and getattr(test.func, "id", "") == "isinstance"
            and len(test.args) == 2
        ):
            types = (
                test.args[1].elts
                if isinstance(test.args[1], ast.Tuple)
                else [test.args[1]]
            )
            names = {getattr(item, "id", "") for item in types}
            if "Mapping" in names:
                names.add("dict")
            expected = self.declared_types(test.args[0], field)
            if expected:
                return (
                    True
                    if expected <= names
                    else False
                    if not expected & names
                    else None
                )
            origin = self.value(test.args[0])
            if origin == _WHOLE:
                return bool(names & {"dict", "Mapping"})
            if origin == frozenset({field}):
                return "list" in names
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            return self.comparison_guard(test, field)
        return None

    def comparison_guard(self, test: ast.Compare, field: str) -> bool | None:
        if (
            isinstance(test.left, ast.Call)
            and getattr(test.left.func, "id", "") == "len"
            and len(test.left.args) == 1
        ):
            length_origin = self.shape_origin(test.left.args[0])
            if self.shapes.max_lengths.get(length_origin or "") == ast.dump(
                test.comparators[0]
            ):
                if isinstance(test.ops[0], ast.LtE):
                    return True
                if isinstance(test.ops[0], ast.Gt):
                    return False
        if (
            isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
            and isinstance(test.ops[0], (ast.Is, ast.IsNot))
        ):
            expected = self.declared_types(test.left, field)
            if (expected and "NoneType" not in expected) or self.field_refusal_source(
                test.left, field
            ):
                return isinstance(test.ops[0], ast.IsNot)
        if (
            isinstance(test.ops[0], (ast.In, ast.NotIn))
            and self.keys(test.left) == frozenset({field})
            and self.value(test.comparators[0]) == _WHOLE
        ):
            return isinstance(test.ops[0], ast.In)
        if (
            isinstance(test.left, ast.Call)
            and getattr(test.left.func, "id", "") == "type"
            and len(test.left.args) == 1
            and isinstance(test.ops[0], (ast.Is, ast.IsNot, ast.In, ast.NotIn))
        ):
            declared = self.declared_types(test.left.args[0], field)
            other = test.comparators[0]
            candidates = other.elts if isinstance(other, ast.Tuple) else [other]
            names = {getattr(item, "id", "") for item in candidates}
            if declared:
                equal = (
                    True
                    if declared <= names
                    else False
                    if not declared & names
                    else None
                )
                if equal is not None:
                    return (
                        equal
                        if isinstance(test.ops[0], (ast.Is, ast.In))
                        else not equal
                    )
        return None

    def preserves_payload(self, node: ast.Call) -> bool:
        previous = self.checking_guards
        self.checking_guards = True
        try:
            return self.value(node) == _WHOLE
        finally:
            self.checking_guards = previous

    def destinations(
        self,
        node: ast.AST,
        fields: set[str] | frozenset[str],
        seen: frozenset[int] = frozenset(),
    ) -> set[str]:
        previous = self.checking_guards
        self.checking_guards = True
        try:
            return {
                field
                for field in self._destinations(node, fields, seen)
                if self.execution_guaranteed(node, field)
            }
        finally:
            self.checking_guards = previous

    def _destinations(
        self,
        node: ast.AST,
        fields: set[str] | frozenset[str],
        seen: frozenset[int] = frozenset(),
    ) -> set[str]:
        """Follow a projected expression through aliases to returned mapping keys."""

        if id(node) in seen:
            return set()
        parent = self.parents.get(node)
        if isinstance(parent, ast.Return):
            return (
                set(fields)
                if self.return_filter is None or parent is self.return_filter
                else set()
            )
        if isinstance(parent, (ast.Assign, ast.AnnAssign)):
            targets = (
                parent.targets if isinstance(parent, ast.Assign) else [parent.target]
            )
            result: set[str] = set()
            for target in targets:
                if (
                    isinstance(target, ast.Subscript)
                    and self.value(target.value) == _WHOLE
                ):
                    owned = self.keys(target.slice)
                    if self.return_filter is None:
                        result.update(owned)
                    elif isinstance(target.value, ast.Name):
                        result.update(
                            self.returned_mapping_reads(
                                target.value, owned, seen | {id(node)}
                            )
                        )
                elif isinstance(target, ast.Name):
                    for read in self.nodes:
                        if (
                            isinstance(read, ast.Name)
                            and isinstance(read.ctx, ast.Load)
                            and read.id == target.id
                            and (read.lineno, read.col_offset)
                            > (target.lineno, target.col_offset)
                        ):
                            result.update(
                                self.destinations(read, fields, seen | {id(node)})
                            )
            return result
        if isinstance(parent, ast.Dict):
            return self.dictionary_destinations(parent, node, fields, seen)
        if isinstance(parent, ast.IfExp):
            return self.destinations(parent, fields, seen | {id(node)})
        if isinstance(parent, ast.Call) and (
            getattr(parent.func, "id", "") in {"dict", "tuple", "cast"}
            or (
                getattr(parent.func, "id", "") in self.functions
                and self.preserves_payload(parent)
            )
        ):
            return self.destinations(parent, fields, seen | {id(node)})
        return set()

    def returned_mapping_reads(
        self, target: ast.Name, fields: frozenset[str], seen: frozenset[int]
    ) -> set[str]:
        result: set[str] = set()
        for read in self.nodes:
            if (
                isinstance(read, ast.Name)
                and isinstance(read.ctx, ast.Load)
                and read.id == target.id
                and (read.lineno, read.col_offset) > (target.lineno, target.col_offset)
            ):
                result.update(self.destinations(read, fields, seen))
        return result

    def reaches_return(self, node: ast.AST, returned: ast.Return, field: str) -> bool:
        previous = self.return_filter
        self.return_filter = returned
        try:
            expression = (
                node.value
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                and node.value is not None
                else node
            )
            return field in self.destinations(expression, {field})
        finally:
            self.return_filter = previous

    def dictionary_destinations(
        self,
        parent: ast.Dict,
        node: ast.AST,
        fields: set[str] | frozenset[str],
        seen: frozenset[int],
    ) -> set[str]:
        remaining = set(self.required_fields if "*" in fields else fields)
        found = False
        for key, value in zip(parent.keys, parent.values, strict=True):
            if value is node:
                found = True
                if key is not None:
                    remaining = set(self.keys(key))
            elif found:
                overridden = self.keys(key) if key is not None else remaining.copy()
                remaining.difference_update(
                    field
                    for field in overridden
                    if not self.projected_return(value, field)
                )
        return self.destinations(parent, remaining, seen | {id(node)})

    def field_refusal_source(
        self, value: ast.expr, field: str, seen: frozenset[int] = frozenset()
    ) -> bool:
        if id(value) in seen:
            return False
        seen = seen | {id(value)}
        origin = self.shape_origin(value)
        if origin == field or (
            origin is not None and origin.startswith((field + "[]", field + "."))
        ):
            return True
        if isinstance(value, ast.Name):
            _, bound = self.name_binding(value)
            return bound is not None and self.field_refusal_source(bound, field, seen)
        if isinstance(value, ast.Call):
            helper = self.functions.get(getattr(value.func, "id", ""))
            if (
                helper is None
                or helper.returns is None
                or "None" not in ast.unparse(helper.returns)
            ):
                return False
            return any(
                self.field_refusal_source(argument, field, seen)
                for argument in value.args
            )
        return False

    def refusal_guard(self, test: ast.expr, field: str) -> bool:
        """Only declared or field-derived shape refusals exempt a raw return."""

        return self.guard_value(test, field) is False

    def guaranteed_bindings(
        self, value: ast.expr, field: str, seen: frozenset[int] = frozenset()
    ) -> bool:
        """Do not infer projection from an assignment on an optional path."""

        if id(value) in seen:
            return False
        seen = seen | {id(value)}
        if isinstance(value, ast.Name) and isinstance(value.ctx, ast.Load):
            if value.id in self.unmodeled_bindings:
                return False
            statement, bound = self.name_binding(value)
            if statement is not None:
                return (
                    bound is not None
                    and self.execution_guaranteed(statement, field)
                    and self.guaranteed_bindings(bound, field, seen)
                )
        return all(
            self.guaranteed_bindings(child, field, seen)
            for child in ast.iter_child_nodes(value)
            if isinstance(child, ast.expr)
        )

    def sequence_changed(
        self, name: str, binding: ast.AST, use: ast.expr, index: int
    ) -> bool:
        """Stored sequence elements need an unchanged sequence up to their use."""

        aliases = {name}
        start = (
            getattr(binding, "end_lineno", 0),
            getattr(binding, "end_col_offset", 0),
        )
        stop = (use.lineno, use.col_offset)
        for node in sorted(
            self.nodes,
            key=lambda node: (
                getattr(node, "lineno", 0),
                getattr(node, "col_offset", 0),
            ),
        ):
            if (
                not start
                <= (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
                < stop
            ):
                continue
            if isinstance(node, (ast.For, ast.AsyncFor, ast.NamedExpr)) and (
                self.sequence_reference(node.target, aliases)
            ):
                return True
            if isinstance(node, (ast.With, ast.AsyncWith)) and any(
                item.optional_vars is not None
                and self.sequence_reference(item.optional_vars, aliases)
                for item in node.items
            ):
                return True
            if isinstance(node, ast.ExceptHandler) and node.name in aliases:
                return True
            if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)) and (
                self.sequence_reference(node, aliases)
            ):
                return True
            for target, assigned in _assignment_pairs(node):
                if (
                    assigned is not None
                    and self.sequence_reference(assigned, aliases)
                    and not isinstance(target, ast.Name)
                ):
                    return True
                if (
                    isinstance(target, ast.Name)
                    and assigned is not None
                    and self.sequence_reference(assigned, aliases)
                ):
                    aliases.add(target.id)
                if (
                    isinstance(node, ast.AugAssign)
                    and isinstance(target, ast.Name)
                    and target.id in aliases
                ):
                    return True
                if (
                    isinstance(target, ast.Subscript)
                    and self.sequence_reference(target.value, aliases)
                    and not (
                        isinstance(node, (ast.Assign, ast.AnnAssign))
                        and isinstance(target.value, ast.Name)
                        and isinstance(target.slice, ast.Constant)
                        and isinstance(target.slice.value, int)
                        and target.slice.value >= 0
                        and target.slice.value != index
                    )
                ):
                    return True
            if isinstance(node, ast.Delete) and any(
                self.sequence_reference(target, aliases) for target in node.targets
            ):
                return True
            if (
                isinstance(node, ast.Attribute)
                and self.sequence_reference(node.value, aliases)
                and node.attr not in {"count", "index", "copy"}
            ):
                return True
            if (
                isinstance(node, ast.Call)
                and (
                    getattr(node.func, "id", "")
                    not in {"len", "bool", "tuple", "list", "sorted", "iter"}
                    or getattr(node.func, "id", "") in self.functions
                )
                and any(
                    self.sequence_reference(argument, aliases)
                    for argument in [
                        *node.args,
                        *(keyword.value for keyword in node.keywords),
                    ]
                )
            ):
                return True
        return False

    @staticmethod
    def sequence_reference(value: ast.AST, aliases: set[str]) -> bool:
        return any(
            isinstance(part, ast.Name) and part.id in aliases
            for part in ast.walk(value)
        )

    def projected_sequence_element(
        self,
        value: ast.expr,
        index: int,
        field: str,
        seen: frozenset[int],
        use: ast.expr | None = None,
    ) -> bool:
        if id(value) in seen:
            return False
        seen = seen | {id(value)}
        if isinstance(value, (ast.Tuple, ast.List)):
            return (
                0 <= index < len(value.elts)
                and self.guaranteed_bindings(value.elts[index], field)
                and self.projected_return(value.elts[index], field, seen)
            )
        if isinstance(value, ast.Name):
            if value.id in self.unmodeled_bindings:
                return False
            statement, bound = self.name_binding(value)
            return (
                statement is not None
                and bound is not None
                and self.execution_guaranteed(statement, field)
                and not self.sequence_changed(value.id, statement, use or value, index)
                and self.projected_sequence_element(
                    bound, index, field, seen, use or value
                )
            )
        if not isinstance(value, ast.Call):
            return False
        name = getattr(value.func, "id", "")
        if (
            name not in self.functions
            or name in self.local_names
            or name in self.stack
            or any(keyword.arg is None for keyword in value.keywords)
        ):
            return False
        helper = self.functions[name]
        if any(
            node is not helper
            and isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
            )
            for node in ast.walk(helper)
        ):
            # Closure effects are outside this bounded helper analysis.
            return False
        arguments = {
            parameter.arg: argument
            for parameter, argument in zip(
                (helper.args.posonlyargs + helper.args.args), value.args, strict=False
            )
        }
        arguments.update(
            {
                keyword.arg: keyword.value
                for keyword in value.keywords
                if keyword.arg is not None
            }
        )
        projection = _Projection(
            helper,
            self.functions,
            {
                parameter: self.value(argument)
                for parameter, argument in arguments.items()
            },
            set(),
            self.stack | {name},
            self.required_fields,
            self.shapes,
        )
        projection.canonical_arguments = {
            parameter
            for parameter, argument in arguments.items()
            if self.guaranteed_bindings(argument, field)
            and self.projected_return(argument, field, seen)
        }
        if (
            projection.canonical_argument_escapes()
            or projection.mutated_after(helper, field)
            or projection.overwritten_after(helper, field)
        ):
            return False
        returned = [
            node
            for node in projection.nodes
            if isinstance(node, ast.Return)
            and not projection.impossible_path(node, field)
        ]
        return bool(returned) and all(
            node.value is not None
            and projection.projected_sequence_element(
                node.value, index, field, frozenset()
            )
            for node in returned
        )

    def canonical_argument_escapes(self) -> bool:
        """Only the returned sequence may hold a canonical helper argument."""

        returned_sequences: set[int] = set()
        for node in self.nodes:
            if not isinstance(node, ast.Return) or node.value is None:
                continue
            value = node.value
            seen: set[int] = set()
            while isinstance(value, ast.Name) and id(value) not in seen:
                seen.add(id(value))
                _, bound = self.name_binding(value)
                if bound is None:
                    break
                value = bound
            if isinstance(value, (ast.Tuple, ast.List)):
                returned_sequences.add(id(value))
        aliases = set(self.canonical_arguments)
        for _, target, assigned in sorted(
            self.write_bindings,
            key=lambda binding: getattr(binding[0], "lineno", 0),
        ):
            if assigned is None or isinstance(
                assigned, (ast.Tuple, ast.List, ast.Dict, ast.Set)
            ):
                continue
            if isinstance(target, ast.Attribute) and self.sequence_reference(
                assigned, aliases
            ):
                # Storing a canonical value in an attribute hands it to code
                # this projection cannot see writing back through, so the
                # returned sequence can no longer be proven untouched.
                return True
            if isinstance(target, ast.Name) and self.sequence_reference(
                assigned, aliases
            ):
                aliases.add(target.id)
        return any(
            isinstance(node, (ast.Tuple, ast.List, ast.Dict, ast.Set))
            and id(node) not in returned_sequences
            and self.sequence_reference(node, aliases)
            for node in self.nodes
        )

    def projected_subscript(
        self, value: ast.Subscript, field: str, seen: frozenset[int]
    ) -> bool:
        if isinstance(value.slice, ast.Constant) and isinstance(value.slice.value, int):
            return self.projected_sequence_element(
                value.value, value.slice.value, field, seen
            )
        writes = [
            node
            for node in self.nodes
            if isinstance(node, ast.Assign) and node.lineno < value.lineno
        ]
        for write in sorted(writes, key=lambda item: item.lineno, reverse=True):
            if any(_write_key(target) == _write_key(value) for target in write.targets):
                return self.projected_return(write.value, field, seen | {id(value)})
        return self.projected_return(value.value, field, seen | {id(value)})

    def normalizes_outer_array(self, name: str) -> bool:
        helper = self.functions.get(name)
        if helper is None:
            return False
        returns = [
            node.value for node in _own_nodes(helper) if isinstance(node, ast.Return)
        ]
        if len(returns) != 1 or not isinstance(returns[0], ast.IfExp):
            return False
        expression = returns[0]
        test = expression.test
        return (
            isinstance(expression.orelse, ast.Name)
            and isinstance(expression.body, ast.Call)
            and getattr(expression.body.func, "id", "") == "tuple"
            and len(expression.body.args) == 1
            and ast.dump(expression.body.args[0]) == ast.dump(expression.orelse)
            and isinstance(test, ast.Call)
            and getattr(test.func, "id", "") == "isinstance"
            and len(test.args) == 2
            and ast.dump(test.args[0]) == ast.dump(expression.orelse)
            and getattr(test.args[1], "id", "") == "list"
        )

    def helper_projects(self, value: ast.Call, field: str) -> bool:
        name = getattr(value.func, "id", "")
        if name not in self.functions or name in self.stack:
            return False
        helper = self.functions[name]
        if not any(
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "canonicalize_json_containers"
            for node in _own_nodes(helper)
        ):
            return False
        covered: set[str] = set()
        projection = _Projection(
            helper,
            self.functions,
            {
                parameter.arg: self.value(argument)
                for parameter, argument in zip(
                    (helper.args.posonlyargs + helper.args.args),
                    value.args,
                    strict=False,
                )
            },
            covered,
            self.stack | {name},
            self.required_fields,
            self.shapes,
        )
        projection.analyze()
        return field in covered

    def array_depth(self, field: str) -> int:
        depth = 1
        path = field + "[]"
        while "list" in self.shapes.types.get(path, frozenset()):
            depth += 1
            path += "[]"
        return depth

    def tuple_depth(
        self, value: ast.expr, field: str, seen: frozenset[int] = frozenset()
    ) -> int:
        if id(value) in seen:
            return 0
        seen = seen | {id(value)}
        if isinstance(value, ast.IfExp):
            return min(
                self.tuple_depth(value.body, field, seen),
                self.tuple_depth(value.orelse, field, seen),
            )
        if isinstance(value, ast.Tuple):
            return (
                100
                if not value.elts
                else 1 + min(self.tuple_depth(item, field, seen) for item in value.elts)
            )
        if isinstance(value, ast.Name):
            return self.name_tuple_depth(value, field, seen)
        if isinstance(value, ast.Subscript):
            return max(0, self.tuple_depth(value.value, field, seen) - 1)
        if isinstance(value, ast.Call):
            name = getattr(value.func, "id", "")
            if name == "canonicalize_json_containers":
                return 100
            if name in {"dict", "cast"} and value.args:
                return self.tuple_depth(value.args[-1], field, seen)
            if isinstance(value.func, ast.Attribute) and value.func.attr == "get":
                return max(0, self.tuple_depth(value.func.value, field, seen) - 1)
            if name == "tuple" and value.args:
                return 1 + self.element_depth(value.args[0], field, seen)
            if value.args and self.normalizes_outer_array(name):
                return max(1, self.tuple_depth(value.args[0], field, seen))
        return 0

    def name_tuple_depth(
        self, value: ast.Name, field: str, seen: frozenset[int]
    ) -> int:
        ancestor: ast.AST = value
        while (parent := self.parents.get(ancestor)) is not None:
            if isinstance(parent, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
                for generator in parent.generators:
                    if (
                        isinstance(generator.target, ast.Name)
                        and generator.target.id == value.id
                    ):
                        return max(0, self.tuple_depth(generator.iter, field, seen) - 1)
            ancestor = parent
        for part in sorted(
            self.nodes, key=lambda part: getattr(part, "lineno", 0), reverse=True
        ):
            if (
                isinstance(part, (ast.Assign, ast.AnnAssign))
                and part.value is not None
                and part.lineno < value.lineno
                and any(
                    isinstance(target, ast.Name) and target.id == value.id
                    for target in (
                        part.targets if isinstance(part, ast.Assign) else [part.target]
                    )
                )
            ):
                return (
                    self.tuple_depth(part.value, field, seen)
                    if self.execution_guaranteed(part, field)
                    else 0
                )
            if (
                isinstance(part, ast.For)
                and isinstance(part.target, ast.Name)
                and part.target.id == value.id
                and part.lineno < value.lineno <= (part.end_lineno or part.lineno)
            ):
                return max(0, self.tuple_depth(part.iter, field, seen) - 1)
        return 100 if value.id in self.canonical_arguments else 0

    def element_depth(self, value: ast.expr, field: str, seen: frozenset[int]) -> int:
        if isinstance(value, (ast.GeneratorExp, ast.SetComp, ast.ListComp)):
            return self.tuple_depth(value.elt, field, seen)
        if (
            isinstance(value, ast.Call)
            and getattr(value.func, "id", "") in {"sorted", "set", "list"}
            and value.args
        ):
            return self.element_depth(value.args[0], field, seen)
        if isinstance(value, ast.Name):
            appended = [
                call.args[0]
                for call in self.nodes
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "append"
                and getattr(call.func.value, "id", "") == value.id
                and len(call.args) == 1
                and call.lineno < value.lineno
            ]
            writes = [
                part
                for part in self.nodes
                if isinstance(part, (ast.Assign, ast.AnnAssign))
                and part.lineno < value.lineno
                and any(
                    isinstance(target, ast.Name) and target.id == value.id
                    for target in (
                        part.targets if isinstance(part, ast.Assign) else [part.target]
                    )
                )
            ]
            seed = max(writes, key=lambda part: part.lineno).value if writes else None
            unknown_mutations = any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and getattr(call.func.value, "id", "") == value.id
                and call.func.attr != "append"
                for call in self.nodes
            )
            if (
                appended
                and isinstance(seed, ast.List)
                and not seed.elts
                and not unknown_mutations
            ):
                return min(self.tuple_depth(item, field, seen) for item in appended)
        return max(0, self.tuple_depth(value, field, seen) - 1)

    def unshadowed_builtin(self, name: str) -> bool:
        """Whether a builtin-spelled name is still the builtin here.

        A module-level ``def dict(...)`` replaces the builtin at runtime, so
        trusting it by spelling alone certifies a payload that the replacement
        may have written back as raw JSON.
        """

        return bool(name) and name not in self.shadowed_names

    def projected_call(self, value: ast.Call, field: str, seen: frozenset[int]) -> bool:
        name = getattr(value.func, "id", "")
        if (
            isinstance(value.func, ast.Attribute)
            and value.func.attr == "model_validate"
            and getattr(value.func.value, "id", "") in self.shapes.model_names
        ):
            return True
        return (
            name == "canonicalize_json_containers"
            or (
                name == "tuple"
                and self.unshadowed_builtin(name)
                and self.tuple_depth(value, field) >= self.array_depth(field)
            )
            or (self.array_depth(field) == 1 and self.normalizes_outer_array(name))
            or self.helper_projects(value, field)
            or (
                name in {"dict", "cast"}
                and self.unshadowed_builtin(name)
                and bool(value.args)
                and self.projected_return(value.args[-1], field, seen | {id(value)})
            )
        )

    def projected_return(
        self, value: ast.expr, field: str, seen: frozenset[int] = frozenset()
    ) -> bool:
        if id(value) in seen:
            return False
        key = (value, field)
        if key not in self.projection_cache:
            self.projection_cache[key] = self._projected_return(value, field, seen)
        return self.projection_cache[key]

    def _projected_return(
        self, value: ast.expr, field: str, seen: frozenset[int]
    ) -> bool:
        if isinstance(value, ast.Tuple):
            return self.tuple_depth(value, field) >= self.array_depth(field)
        if isinstance(value, ast.Attribute):
            return self.projected_return(value.value, field, seen | {id(value)})
        if isinstance(value, ast.Subscript):
            return self.projected_subscript(value, field, seen)
        if isinstance(value, ast.Call):
            return self.projected_call(value, field, seen)
        if isinstance(value, ast.Dict):
            for key, item in reversed(list(zip(value.keys, value.values, strict=True))):
                if key is not None and field in self.keys(key):
                    return self.projected_return(item, field, seen | {id(value)})
                if key is None and self.projected_return(
                    item, field, seen | {id(value)}
                ):
                    return True
        if isinstance(value, ast.Name):
            writes = [
                node
                for node in self.nodes
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                and node.value is not None
                and node.lineno < value.lineno
            ]
            for write in sorted(writes, key=lambda item: item.lineno, reverse=True):
                if write.value is None:
                    continue
                for target in (
                    write.targets if isinstance(write, ast.Assign) else [write.target]
                ):
                    if isinstance(target, ast.Name) and target.id == value.id:
                        return self.projected_return(
                            write.value, field, seen | {id(value)}
                        )
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == value.id
                        and field in self.keys(target.slice)
                    ):
                        return self.projected_return(
                            write.value, field, seen | {id(value)}
                        )
        return isinstance(value, ast.Name) and value.id in self.canonical_arguments

    def block_exits(self, statements: list[ast.stmt], field: str) -> bool:
        for statement in statements:
            if isinstance(statement, (ast.Return, ast.Raise)):
                return True
            if (
                isinstance(statement, ast.Try)
                and self.block_exits(statement.body, field)
                and all(
                    self.block_exits(handler.body, field)
                    for handler in statement.handlers
                )
            ):
                return True
            if isinstance(statement, ast.If):
                known = self.guard_value(statement.test, field)
                branches = (
                    [statement.body]
                    if known is True
                    else [statement.orelse]
                    if known is False
                    else [statement.body, statement.orelse]
                )
                if all(self.block_exits(branch, field) for branch in branches):
                    return True
        return False

    def impossible_path(self, node: ast.AST, field: str) -> bool:
        key = (node, field)
        if key not in self.path_cache:
            self.path_cache[key] = self._impossible_path(node, field)
        return self.path_cache[key]

    def _impossible_path(self, node: ast.AST, field: str) -> bool:
        child = node
        while (parent := self.parents.get(child)) is not None:
            if isinstance(parent, ast.If):
                known = self.guard_value(parent.test, field)
                if (child in parent.body and known is False) or (
                    child in parent.orelse and known is True
                ):
                    return True
            for _, siblings in ast.iter_fields(parent):
                if isinstance(siblings, list) and child in siblings:
                    earlier = siblings[: siblings.index(child)]
                    if all(
                        isinstance(item, ast.stmt) for item in earlier
                    ) and self.block_exits(earlier, field):
                        return True
            child = parent
        return False

    def raw_return(self, node: ast.AST, field: str) -> bool:
        for returned in self.nodes:
            if not isinstance(returned, ast.Return) or self.impossible_path(
                returned, field
            ):
                continue
            if returned.lineno >= getattr(node, "lineno", 0) and self.reaches_return(
                node, returned, field
            ):
                continue
            if returned.value is not None and self.projected_return(
                returned.value, field
            ):
                continue
            child: ast.AST = returned
            invalid_shape = False
            while (parent := self.parents.get(child)) is not None:
                if isinstance(parent, ast.If):
                    in_body = child in parent.body
                    known = self.guard_value(parent.test, field)
                    if known is (not in_body) or (
                        known is None
                        and in_body
                        and self.refusal_guard(parent.test, field)
                    ):
                        invalid_shape = True
                if (
                    isinstance(parent, ast.ExceptHandler)
                    and getattr(parent.type, "id", "") == "TypeError"
                ):
                    region = self.parents.get(parent)
                    if (
                        isinstance(region, ast.Try)
                        and len(region.body) == 1
                        and isinstance(region.body[0], ast.Assign)
                    ):
                        call = region.body[0].value
                        if (
                            isinstance(call, ast.Call)
                            and getattr(call.func, "id", "") == "iter"
                            and call.args
                            and self.value(call.args[0]) == frozenset({field})
                        ):
                            invalid_shape = True
                child = parent
            if not invalid_shape:
                return True
        return False

    def loop_can_skip(self, loop: ast.For, node: ast.AST, field: str) -> bool:
        for control in self.loop_controls:
            if (
                control.lineno,
                control.col_offset,
            ) >= (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)):
                continue
            owner = self.parents.get(control)
            while owner is not None and not isinstance(
                owner, (ast.For, ast.While, ast.AsyncFor)
            ):
                owner = self.parents.get(owner)
            if owner is loop and not self.impossible_path(control, field):
                return True
        return False

    def execution_guaranteed(self, node: ast.AST, field: str) -> bool:
        """Reject unknown branches, skippable loops and exception regions."""

        child = node
        while (parent := self.parents.get(child)) is not None:
            if isinstance(parent, (ast.If, ast.IfExp)):
                body = parent.body if isinstance(parent.body, list) else [parent.body]
                other = (
                    parent.orelse
                    if isinstance(parent.orelse, list)
                    else [parent.orelse]
                )
                required = True if child in body else False if child in other else None
                if (
                    required is not None
                    and self.guard_value(parent.test, field) is not required
                ):
                    return False
            elif isinstance(parent, ast.For):
                if child in parent.body and not (
                    isinstance(parent.iter, (ast.Tuple, ast.List))
                    and any(
                        not isinstance(element, ast.Starred)
                        for element in parent.iter.elts
                    )
                ):
                    return False
                if child in parent.orelse or self.loop_can_skip(parent, node, field):
                    return False
            elif isinstance(parent, (ast.Try, ast.TryStar)):
                if not (
                    child in parent.body
                    and not parent.finalbody
                    and all(
                        len(handler.body) == 1
                        and isinstance(handler.body[0], ast.Raise)
                        for handler in parent.handlers
                    )
                ):
                    return False
            elif isinstance(
                parent,
                (
                    ast.While,
                    ast.AsyncFor,
                    ast.ExceptHandler,
                    ast.Match,
                    ast.With,
                    ast.AsyncWith,
                ),
            ):
                return False
            child = parent
        return True

    def helper_mutates(self, call: ast.Call, field: str) -> bool:
        arguments = [*call.args, *(keyword.value for keyword in call.keywords)]
        if not any(self.value(argument) == _WHOLE for argument in arguments):
            return False
        name = getattr(call.func, "id", "")
        if name in {
            "canonicalize_json_containers",
            "dict",
            "tuple",
            "cast",
            "isinstance",
            "type",
            "len",
            "bool",
            "set",
            "frozenset",
            "list",
            "sorted",
            "iter",
            "all",
            "any",
            "str",
            "repr",
            "min",
            "max",
        }:
            return False
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "model_validate"
            and getattr(call.func.value, "id", "") in self.shapes.model_names
        ):
            return False
        if name not in self.functions or name in self.stack:
            return True
        if any(keyword.arg is None for keyword in call.keywords):
            return True
        helper = self.functions[name]
        bound = {
            parameter.arg: self.value(argument)
            for parameter, argument in zip(
                (helper.args.posonlyargs + helper.args.args), call.args, strict=False
            )
        }
        bound.update(
            {
                keyword.arg: self.value(keyword.value)
                for keyword in call.keywords
                if keyword.arg is not None
            }
        )
        projection = _Projection(
            helper,
            self.functions,
            bound,
            set(),
            self.stack | {name},
            self.required_fields,
            self.shapes,
        )
        projection.canonical_arguments = {
            name for name, origin in bound.items() if origin == _WHOLE
        }
        return projection.mutated_after(helper, field) or projection.overwritten_after(
            helper, field
        )

    def mapping_update_mutates(self, value: ast.expr, field: str) -> bool:
        if not isinstance(value, ast.Dict):
            return True
        return any(
            (key is None or not self.keys(key) or field in self.keys(key))
            and not self.projected_return(item, field)
            for key, item in zip(value.keys, value.values, strict=True)
        )

    def augmented_mutates(self, write: ast.AugAssign, field: str) -> bool:
        target, assigned = _assignment_pairs(write)[0]
        assert assigned is not None
        if isinstance(target, ast.Name) and self.value(target) == _WHOLE:
            return not isinstance(write.op, ast.BitOr) or self.mapping_update_mutates(
                assigned, field
            )
        if isinstance(target, ast.Subscript) and self.value(target.value) == _WHOLE:
            return (
                not self.keys(target.slice) or field in self.keys(target.slice)
            ) and not self.projected_return(assigned, field)
        return False

    def mutates_field(self, call: ast.Call, field: str) -> bool:
        key = (call, field)
        if key not in self.effect_cache:
            self.effect_cache[key] = self._mutates_field(call, field)
        return self.effect_cache[key]

    def callable_expression(
        self, value: ast.expr, seen: frozenset[str] = frozenset()
    ) -> ast.expr:
        if not isinstance(value, ast.Name) or value.id in seen:
            return value
        _, bound = self.name_binding(value)
        return (
            self.callable_expression(bound, seen | {value.id})
            if bound is not None
            else value
        )

    def _mutates_field(self, call: ast.Call, field: str) -> bool:
        resolved = self.callable_expression(call.func)
        if resolved is not call.func:
            call = ast.Call(func=resolved, args=call.args, keywords=call.keywords)
        if (
            not isinstance(call.func, ast.Attribute)
            or self.value(call.func.value) != _WHOLE
        ):
            return self.helper_mutates(call, field)
        method = call.func.attr
        if method in {"get", "keys", "values", "items", "copy"}:
            return False
        if method == "update":
            if any(
                keyword.arg is None
                or (
                    keyword.arg == field
                    and not self.projected_return(keyword.value, field)
                )
                for keyword in call.keywords
            ):
                return True
            return any(
                self.mapping_update_mutates(argument, field) for argument in call.args
            )
        if method in {"__setitem__", "setdefault"} and len(call.args) == 2:
            return (
                not self.keys(call.args[0]) or field in self.keys(call.args[0])
            ) and not self.projected_return(call.args[1], field)
        return True

    def mutated_after(self, node: ast.AST, field: str) -> bool:
        location = (
            getattr(node, "lineno", 0)
            if isinstance(node, ast.FunctionDef)
            else (getattr(node, "end_lineno", None) or getattr(node, "lineno", 0)),
            getattr(node, "col_offset", 0)
            if isinstance(node, ast.FunctionDef)
            else (
                getattr(node, "end_col_offset", None) or getattr(node, "col_offset", 0)
            ),
        )
        return any(
            isinstance(write, (ast.Call, ast.AugAssign))
            and (write.lineno, write.col_offset) > location
            and not self.impossible_path(write, field)
            and (
                self.mutates_field(write, field)
                if isinstance(write, ast.Call)
                else self.augmented_mutates(write, field)
            )
            for write in self.nodes
        )

    def overwritten_after(self, node: ast.AST, field: str) -> bool:
        """A possibly executed later raw write invalidates earlier projection."""

        location = (
            getattr(node, "lineno", 0)
            if isinstance(node, ast.FunctionDef)
            else (getattr(node, "end_lineno", None) or getattr(node, "lineno", 0)),
            getattr(node, "col_offset", 0)
            if isinstance(node, ast.FunctionDef)
            else (
                getattr(node, "end_col_offset", None) or getattr(node, "col_offset", 0)
            ),
        )
        for write in self.nodes:
            if (
                not isinstance(write, (ast.Assign, ast.AnnAssign))
                or write.value is None
                or (
                    not isinstance(node, ast.FunctionDef)
                    and write not in self.returned_nodes()
                )
            ):
                continue
            if (write.lineno, write.col_offset) <= location:
                continue
            for target, assigned in _assignment_pairs(write):
                affected = (
                    isinstance(target, ast.Subscript)
                    and self.value(target.value) == _WHOLE
                    and (
                        not self.keys(target.slice) or field in self.keys(target.slice)
                    )
                ) or (
                    isinstance(target, ast.Name)
                    and self.value(target) == _WHOLE
                    and any(
                        isinstance(read, ast.Name)
                        and isinstance(read.ctx, ast.Load)
                        and read.id == target.id
                        and read.lineno > write.lineno
                        and read in self.returned_nodes()
                        for read in self.nodes
                    )
                )
                if affected and (
                    assigned is None or not self.projected_return(assigned, field)
                ):
                    return True
        return False

    def record(self, node: ast.AST, fields: set[str] | frozenset[str]) -> None:
        """Unknown conditional execution cannot prove projection coverage."""

        if self.checking_guards:
            return
        self.checking_guards = True
        try:
            for field in self.required_fields if "*" in fields else fields:
                if (
                    not self.raw_return(node, field)
                    and self.execution_guaranteed(node, field)
                    and not self.overwritten_after(node, field)
                    and not self.mutated_after(node, field)
                ):
                    self.covered.add(field)
        finally:
            self.checking_guards = False

    def analyze(self) -> frozenset[str]:
        live = self.returned_nodes()
        for node in self.nodes:
            if node not in live:
                continue
            if isinstance(node, ast.Call):
                if (
                    getattr(node.func, "id", "") == "canonicalize_json_containers"
                    and node.args
                ):
                    self.record(node, self.destinations(node, self.value(node.args[0])))
                elif getattr(node.func, "id", "") in self.functions:
                    self.value(node)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and self.value(target.value) == _WHOLE
                    ):
                        self.record(
                            node,
                            {
                                field
                                for field in self.keys(target.slice)
                                if self.projected_return(node.value, field)
                            },
                        )
        returns = [
            self.value(node.value)
            for node in self.nodes
            if isinstance(node, ast.Return) and node.value is not None
        ]
        return frozenset.intersection(*returns) if returns else frozenset()


def _uncovered_leaf_fields(
    validator: ast.FunctionDef,
    functions: dict[str, ast.FunctionDef],
    leaf_fields: set[str],
    shapes: _ModelShape | None = None,
) -> set[str]:
    """Trace selections into canonicalization, not admission-only mentions."""

    covered: set[str] = set()
    _Projection(
        validator,
        functions,
        {
            argument.arg: _WHOLE
            for argument in validator.args.args
            if argument.arg != "cls"
        },
        covered,
        frozenset({validator.name}),
        frozenset(leaf_fields),
        shapes,
    ).analyze()
    return set() if "*" in covered else leaf_fields - covered


def _inherited_fields(
    model: ast.ClassDef,
    classes: dict[str, ast.ClassDef],
    seen: frozenset[str] = frozenset(),
    contexts: dict[ast.ClassDef, dict[str, set[str]]] | None = None,
    scopes: dict[ast.ClassDef, dict[str, ast.ClassDef]] | None = None,
) -> dict[str, str]:
    """Collect known base declarations, preserving local annotation overrides."""

    if model.name in seen:
        return {}
    fields: dict[str, str] = {}
    for base in reversed(model.bases):
        parent = (scopes.get(model, classes) if scopes else classes).get(
            getattr(base, "id", "")
        )
        if parent is not None:
            fields.update(
                _inherited_fields(
                    parent, classes, seen | {model.name}, contexts, scopes
                )
            )
    fields.update(
        {
            item.target.id: (
                ast.unparse(item.annotation)
                if contexts is None
                else "tuple[object, ...]"
                if _leaf_container_field(
                    ast.unparse(item.annotation), set(), contexts.get(model, {})
                )
                else "object"
            )
            for item in model.body
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
        }
    )
    return fields


def _inherited_validators(
    model: ast.ClassDef,
    classes: dict[str, ast.ClassDef],
    seen: frozenset[str] = frozenset(),
    scopes: dict[ast.ClassDef, dict[str, ast.ClassDef]] | None = None,
) -> list[ast.FunctionDef]:
    if model.name in seen:
        return []
    methods: dict[str, ast.FunctionDef] = {}
    for base in reversed(model.bases):
        parent = (scopes.get(model, classes) if scopes else classes).get(
            getattr(base, "id", "")
        )
        if parent is not None:
            methods.update(
                {
                    method.name: method
                    for method in _inherited_validators(
                        parent, classes, seen | {model.name}, scopes
                    )
                }
            )
    for method in model.body:
        if isinstance(method, ast.FunctionDef):
            methods.pop(method.name, None)
            if any(_is_before_validator(item) for item in method.decorator_list):
                methods[method.name] = method
    return list(methods.values())


def _table_module_key(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _table_import_target(
    module: str, level: int, name: str | None, modules: set[str]
) -> str | None:
    """Resolve an import to a scanned module key, mirroring _import_path."""

    parts = module.split("/")[:-1]
    if level:
        if level - 1 > len(parts):
            return None
        base = "/".join(parts[: len(parts) - (level - 1)] if level > 1 else parts)
        remainder = f"{base}/{name}" if name else base
    else:
        remainder = (name or "").removeprefix("jacobian.math.").replace(".", "/")
    candidate = f"{remainder}.py"
    if candidate in modules:
        return candidate
    candidate = f"{remainder}/__init__.py"
    return candidate if candidate in modules else None


@dataclass(frozen=True)
class _Tables:
    """String-only indexes; parsed trees are discarded per module."""

    texts: dict[str, str]
    classes: dict[tuple[str, str], _ClassRow]
    aliases: dict[tuple[str, str], str]
    imports: dict[tuple[str, str], tuple[str, str]]
    validator_sources: dict[tuple[str, str, str], str]


@dataclass(frozen=True)
class _ClassRow:
    bases: tuple[str, ...]
    fields: dict[str, str]
    validators: dict[str, int]
    methods: dict[str, dict[str, object]]
    lineno: int
    max_lengths: dict[str, str]


def _base_names(node: ast.ClassDef) -> list[str]:
    """Direct base names, ignoring subscripts and attribute roots."""

    names: list[str] = []
    for base in node.bases:
        current = base
        while isinstance(current, ast.Subscript):
            current = current.value
        name = getattr(current, "id", getattr(current, "attr", ""))
        if name:
            names.append(name)
    return names


def _declared_fields(node: ast.ClassDef) -> dict[str, str]:
    return {
        item.target.id: ast.unparse(item.annotation)
        for item in node.body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    }


def _declared_max_length_fields(node: ast.ClassDef) -> dict[str, str]:
    return {
        item.target.id: ast.dump(keyword.value)
        for item in node.body
        if isinstance(item, ast.AnnAssign)
        and isinstance(item.target, ast.Name)
        and isinstance(item.value, ast.Call)
        and getattr(item.value.func, "id", "") == "Field"
        for keyword in item.value.keywords
        if keyword.arg == "max_length"
    }


def _declared_methods(node: ast.ClassDef) -> dict[str, dict[str, object]]:
    return {
        item.name: {
            "lineno": item.lineno,
            "before": any(
                _is_before_validator(decorator) for decorator in item.decorator_list
            ),
        }
        for item in node.body
        if isinstance(item, ast.FunctionDef)
    }


def _declared_validator_sources(
    text: str, node: ast.ClassDef, module: str, into: dict[tuple[str, str, str], str]
) -> None:
    """Retain before-validator bodies as source so they need no re-parse."""

    for item in node.body:
        if isinstance(item, ast.FunctionDef) and any(
            _is_before_validator(decorator) for decorator in item.decorator_list
        ):
            segment = ast.get_source_segment(text, item)
            if segment is not None:
                into[(module, node.name, item.name)] = segment


def _module_alias_declarations(node: ast.AST) -> dict[str, str]:
    """Alias sources declared by one top-level statement."""

    if isinstance(node, ast.TypeAlias):
        return {node.name.id: ast.unparse(node.value)}
    if isinstance(node, ast.Assign) and node.value is not None:
        source = ast.unparse(node.value)
        return {
            target.id: source for target in node.targets if isinstance(target, ast.Name)
        }
    if (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    ):
        return {node.target.id: ast.unparse(node.value)}
    return {}


def _build_tables(root: Path) -> _Tables:
    """String-only indexes; parsed trees are discarded per module.

    Retaining every parsed module costs two orders of magnitude more than the
    source and crashes memory-constrained CI workers on the tree-wide gate.
    Everything below is strings and line numbers; validator and helper bodies
    are re-parsed transiently per module and discarded afterwards.
    """

    modules = sorted(
        _table_module_key(path, root) for path in sorted(root.rglob("*.py"))
    )
    module_set = set(modules)
    texts: dict[str, str] = {}
    classes: dict[tuple[str, str], _ClassRow] = {}
    aliases: dict[tuple[str, str], str] = {}
    imports: dict[tuple[str, str], tuple[str, str]] = {}
    validator_sources: dict[tuple[str, str, str], str] = {}
    for module in modules:
        text = (root / module).read_text(encoding="utf-8")
        texts[module] = text
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = _declared_methods(node)
                classes[(module, node.name)] = _ClassRow(
                    bases=tuple(_base_names(node)),
                    fields=_declared_fields(node),
                    validators={
                        name: cast(int, info["lineno"])
                        for name, info in methods.items()
                        if info["before"]
                    },
                    methods=methods,
                    lineno=node.lineno,
                    max_lengths=_declared_max_length_fields(node),
                )
                _declared_validator_sources(text, node, module, validator_sources)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    target = _table_import_target(
                        module, node.level, node.module, module_set
                    )
                    if target is not None:
                        imports[(module, alias.asname or alias.name)] = (
                            target,
                            alias.name,
                        )
            for declared, source in _module_alias_declarations(node).items():
                aliases[(module, declared)] = source
        del tree
    return _Tables(
        texts=texts,
        classes=classes,
        aliases=aliases,
        imports=imports,
        validator_sources=validator_sources,
    )


def _table_scope(
    module: str,
    tables: _Tables,
    unique: dict[str, tuple[str, str]],
    cache: dict[str, dict[str, tuple[str, str]]],
) -> dict[str, tuple[str, str]]:
    """Name -> ``(module, name)`` visible in ``module``.

    Globally unique names rank lowest, then explicit imports, then local
    definitions, mirroring the previous ``unique_classes`` + module scope.
    """

    if module in cache:
        return cache[module]
    classes = tables.classes
    imports = tables.imports
    scope: dict[str, tuple[str, str]] = dict(unique)
    for (owner, alias), (target, original) in imports.items():
        if owner != module:
            continue
        if (target, original) in classes:
            scope.setdefault(alias, (target, original))
    for key in classes:
        if key[0] == module:
            scope[key[1]] = key
    cache[module] = scope
    return scope


def _table_aliases(
    module: str,
    tables: _Tables,
    cache: dict[str, dict[str, set[str]]],
) -> dict[str, set[str]]:
    """Resolved alias sources for ``module``, imports before local names."""

    if module in cache:
        return cache[module]
    cache[module] = {}
    imports = tables.imports
    declared = tables.aliases
    aliases: dict[str, set[str]] = {}
    for (owner, alias), (target, original) in imports.items():
        if owner != module:
            continue
        imported = _table_aliases(target, tables, cache)
        if original in imported:
            aliases[alias] = {
                _expand_alias(value, imported, frozenset({original}))
                for value in imported[original]
            }
    for (owner, name), source in declared.items():
        if owner == module:
            aliases[name] = {source}
    cache[module] = aliases
    return aliases


def _table_inherited_fields(
    key: tuple[str, str],
    tables: _Tables,
    scopes: dict[str, dict[str, tuple[str, str]]],
    aliases: dict[str, dict[str, set[str]]],
    unique: dict[str, tuple[str, str]],
    scope_cache: dict[str, dict[str, tuple[str, str]]],
    raw: bool = False,
    seen: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Declared fields along the MRO, subclass declarations winning."""

    module, name = key
    if name in seen:
        return {}
    row = tables.classes.get(key)
    if row is None:
        return {}
    scope = _table_scope(module, tables, unique, scope_cache)
    fields: dict[str, str] = {}
    for base in row.bases:
        parent = scope.get(base)
        if parent is not None:
            fields.update(
                _table_inherited_fields(
                    parent,
                    tables,
                    scopes,
                    aliases,
                    unique,
                    scope_cache,
                    raw,
                    seen | {name},
                )
            )
    if raw:
        fields.update(row.fields)
        return fields
    local = _table_aliases(module, tables, aliases)
    for declared, source in row.fields.items():
        fields[declared] = (
            "tuple[object, ...]"
            if _leaf_container_field(source, set(), local)
            else "object"
        )
    return fields


def _table_inherited_validators(
    key: tuple[str, str],
    tables: _Tables,
    unique: dict[str, tuple[str, str]],
    scope_cache: dict[str, dict[str, tuple[str, str]]],
    seen: frozenset[str] = frozenset(),
) -> list[tuple[str, str, str]]:
    """``(module, class, method)`` for before-validators Pydantic inherits."""

    module, name = key
    if name in seen:
        return []
    row = tables.classes.get(key)
    if row is None:
        return []
    scope = _table_scope(module, tables, unique, scope_cache)
    methods: dict[str, tuple[str, str, str]] = {}
    for base in reversed(row.bases):
        parent = scope.get(base)
        if parent is None:
            continue
        for entry in _table_inherited_validators(
            parent, tables, unique, scope_cache, seen | {name}
        ):
            methods[entry[2]] = entry
    for method, info in row.methods.items():
        methods.pop(method, None)
        if info["before"]:
            methods[method] = (module, name, method)
    return list(methods.values())


def _table_json_shapes(
    key: tuple[str, str],
    tables: _Tables,
    unique: dict[str, tuple[str, str]],
    scope_cache: dict[str, dict[str, tuple[str, str]]],
    alias_cache: dict[str, dict[str, set[str]]],
) -> _ModelShape:
    """Known JSON shapes of declared fields, including nested model fields."""

    classes = tables.classes
    scope = _table_scope(key[0], tables, unique, scope_cache)
    shapes = _ModelShape(
        model_names=frozenset(
            name
            for (module, name), row in classes.items()
            if any(base in {"StrictModel", "BaseModel"} for base in row.bases)
        )
    )

    def annotation(node: ast.expr, path: str, seen: frozenset[str]) -> frozenset[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return annotation(ast.parse(node.value, mode="eval").body, path, seen)
        if isinstance(node, ast.Subscript):
            name = getattr(node.value, "id", getattr(node.value, "attr", ""))
            parts = (
                node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            )
            if name == "Annotated":
                return annotation(parts[0], path, seen)
            if name.lower() in {"tuple", "list", "sequence", "set", "frozenset"}:
                elements = (
                    [
                        part
                        for part in parts
                        if not (
                            isinstance(part, ast.Constant) and part.value is Ellipsis
                        )
                    ]
                    if name.lower() == "tuple"
                    else parts[:1]
                )
                shapes.types[path + "[]"] = frozenset().union(
                    *(annotation(part, path + "[]", seen) for part in elements)
                )
                return frozenset({"list"})
            if name in {"Union", "Optional"}:
                result = frozenset().union(
                    *(annotation(part, path, seen) for part in parts)
                )
                return result | ({"NoneType"} if name == "Optional" else set())
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return annotation(node.left, path, seen) | annotation(
                node.right, path, seen
            )
        if isinstance(node, ast.Constant) and node.value is None:
            return frozenset({"NoneType"})
        name = getattr(node, "id", "")
        primitive = {
            "StrictInt": "int",
            "StrictBool": "bool",
            "StrictStr": "str",
            "StrictFloat": "float",
        }.get(name, name)
        if primitive in {"int", "bool", "str", "float"}:
            return frozenset({primitive})
        owner = scope.get(name)
        if owner is not None and name not in seen:
            visit(owner, path + ".", seen | {name})
            return frozenset({"dict"})
        return frozenset()

    def visit(owner: tuple[str, str], prefix: str, seen: frozenset[str]) -> None:
        row = classes.get(owner)
        if row is None:
            return
        local = _table_aliases(owner[0], tables, alias_cache)
        for declared, source in row.max_lengths.items():
            shapes.max_lengths[prefix + declared] = source
        for declared, source in _table_inherited_fields(
            owner, tables, {}, {}, unique, scope_cache, raw=True
        ).items():
            shapes.types[prefix + declared] = annotation(
                _resolved_annotation(source, local), prefix + declared, seen
            )

    visit(key, "", frozenset({key[1]}))
    return shapes


def _table_functions(
    module: str,
    tables: _Tables,
    texts: dict[str, str],
    cache: dict[str, dict[str, ast.FunctionDef]],
) -> dict[str, ast.FunctionDef]:
    """Callable names visible in ``module``; bodies parsed transiently.

    Local definitions at any depth win over imports, matching the previous
    ``_module_functions`` plus ``ast.walk`` fill-in. The cache holds only the
    working set for the module under analysis and is dropped with it.
    """

    if module in cache:
        return cache[module]
    tree = ast.parse(texts[module])
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    imports = tables.imports
    # Imported top-level functions are a partial scope, never a complete cache entry.
    import_scopes: dict[str, dict[str, ast.FunctionDef]] = {}
    for (owner, alias), (target, original) in imports.items():
        if owner != module:
            continue
        if target not in import_scopes:
            imported_tree = ast.parse(texts[target])
            import_scopes[target] = {
                node.name: node
                for node in imported_tree.body
                if isinstance(node, ast.FunctionDef)
            }
        imported = import_scopes[target]
        if original in imported:
            functions.setdefault(alias, imported[original])
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.setdefault(node.name, node)
    cache[module] = functions
    return functions


def _table_validator(
    entry: tuple[str, str, str],
    tables: _Tables,
    texts: dict[str, str],
) -> ast.FunctionDef:
    """Parse one before-validator from its retained source segment."""

    source = tables.validator_sources[entry]
    tree = ast.parse(textwrap.dedent(source))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    raise AssertionError("validator source did not parse to a function")


# Whole-tree static analysis: cost scales with src/jacobian/math, so it cannot
# live under the uniform 30s lane timeout and stay reliable as the tree grows.
@pytest.mark.timeout(300)
def test_before_validators_cover_every_leaf_container_field() -> None:
    """Every declared outer array needs projection, including arrays of models.

    Bare nested models retain their own admission/canonicalization boundary.
    Unsupported annotations are left unenforced; textual field mentions never
    establish projection coverage.
    """

    source_root = Path(__file__).parents[2] / "src" / "jacobian" / "math"
    violations: list[str] = []
    tables = _build_tables(source_root)
    texts = tables.texts
    classes = tables.classes
    counts: dict[str, int] = {}
    first: dict[str, tuple[str, str]] = {}
    for key in classes:
        counts[key[1]] = counts.get(key[1], 0) + 1
        first.setdefault(key[1], key)
    unique = {name: key for name, key in first.items() if counts[name] == 1}
    scope_cache: dict[str, dict[str, tuple[str, str]]] = {}
    alias_cache: dict[str, dict[str, set[str]]] = {}
    for module in sorted({key[0] for key in classes}):
        function_cache: dict[str, dict[str, ast.FunctionDef]] = {}
        for key in classes:
            if key[0] != module:
                continue
            before = _table_inherited_validators(key, tables, unique, scope_cache)
            if not before:
                continue
            fields = _table_inherited_fields(key, tables, {}, {}, unique, scope_cache)
            leaf_fields = {
                name
                for name, annotation in fields.items()
                if _leaf_container_field(annotation, set(), {})
            }
            if not leaf_fields:
                continue
            shapes = _table_json_shapes(key, tables, unique, scope_cache, alias_cache)
            for entry in before:
                validator = _table_validator(entry, tables, texts)
                # Inherited methods retain their defining module globals.
                functions = _table_functions(entry[0], tables, texts, function_cache)
                uncovered = _uncovered_leaf_fields(
                    validator, functions, leaf_fields, shapes
                )
                for name in sorted(uncovered):
                    violations.append(
                        f"{module}:{classes[entry[:2]].validators[entry[2]]} "
                        f"{key[1]}.{name} is never projected"
                    )

    assert not violations, (
        "before-validator projections must reach every leaf container field; "
        "project the payload, derive ownership from cls.model_fields, or name "
        "the field in reachable projection code: " + ", ".join(sorted(violations))
    )


def _fixture_uncovered(
    source: str, fields: set[str], shapes: _ModelShape | None = None
) -> set[str]:
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    return _uncovered_leaf_fields(functions["validate"], functions, fields, shapes)


def test_nested_model_outer_containers_need_projection() -> None:
    for annotation in (
        "tuple[Child, ...]",
        "list[Child]",
        "tuple[Child, ...] | None",
        "Children",
    ):
        assert _leaf_container_field(
            annotation, {"Child"}, {"Children": {"tuple[Child, ...]"}}
        )
    assert not _leaf_container_field("Child", {"Child"}, {})
    assert not _leaf_container_field("Cycle", set(), {"Cycle": {"Cycle"}})
    assert _fixture_uncovered(
        """
def validate(cls, data):
    data['old'] = canonicalize_json_containers(data['old'])
    return data
""",
        {"old", "children"},
    ) == {"children"}


def test_model_fields_mentions_do_not_establish_total_projection() -> None:
    for statement in (
        '"model_fields"',
        "len(cls.model_fields)",
        'selected = {k: data[k] for k in cls.model_fields if k == "old"}',
    ):
        assert _fixture_uncovered(
            f"""
def validate(cls, data):
    {statement}
    return canonicalize_json_containers({{"old": data["old"]}})
""",
            {"old", "provenance"},
        ) == {"provenance"}
    assert _fixture_uncovered(
        """
def validate(cls, data):
    selected = {k: data[k] for k in cls.model_fields if k == "old"}
    return canonicalize_json_containers(selected)
""",
        {"provenance"},
    ) == {"provenance"}
    assert not _fixture_uncovered(
        """
def validate(cls, data):
    selected = {k: data[k] for k in cls.model_fields}
    return canonicalize_json_containers(selected)
""",
        {"old", "provenance"},
    )


def test_admission_mentions_do_not_count_as_projection() -> None:
    assert _fixture_uncovered(
        """
def admit(data):
    return len(data['provenance']) < 10

def validate(cls, data):
    assert admit(data)
    assert len(data['provenance']) < 10
    data['old'] = canonicalize_json_containers(data['old'])
    return data
""",
        {"old", "provenance"},
    ) == {"provenance"}
    assert not _fixture_uncovered(
        """
def project(payload):
    selected = payload.get('provenance')
    payload['provenance'] = canonicalize_json_containers(selected)
    return payload

def validate(cls, data):
    return project(data)
""",
        {"provenance"},
    )
    assert not _fixture_uncovered(
        """
def validate(cls, data):
    normalized = dict(data)
    for key in ('old', 'provenance'):
        normalized[key] = canonicalize_json_containers(normalized.get(key))
    return normalized
""",
        {"old", "provenance"},
    )


def test_model_fields_items_projection_requires_exact_membership_filter() -> None:
    for condition, expected in (
        ("key in cls.model_fields", set()),
        ('key in cls.model_fields and key == "old"', {"provenance"}),
        ('key == "old"', {"provenance"}),
    ):
        assert (
            _fixture_uncovered(
                f"""
def validate(cls, data):
    owned = {{key: item for key, item in data.items() if {condition}}}
    return canonicalize_json_containers(owned)
""",
                {"provenance"},
            )
            == expected
        )


def test_unprojected_model_array_fails_strict_json_at_parent() -> None:
    from typing import Any

    import pytest
    from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

    class Child(BaseModel):
        model_config = ConfigDict(strict=True)
        value: int

    class Parent(BaseModel):
        model_config = ConfigDict(strict=True)
        children: tuple[Child, ...]

        @model_validator(mode="before")
        @classmethod
        def partial(cls, data: Any) -> Any:
            # Admission sees the field but never canonicalizes its outer array.
            assert len(data["children"]) == 1
            return data

    Parent.model_rebuild(_types_namespace={"Child": Child})
    with pytest.raises(ValidationError) as error:
        Parent.model_validate_json('{"children":[{"value":1}]}')
    assert [(item["loc"], item["type"]) for item in error.value.errors()] == [
        (("children",), "tuple_type")
    ]
    assert Parent.model_validate({"children": ({"value": 1},)}).children[0].value == 1


def test_discarded_projection_does_not_cover_returned_payload() -> None:
    for statement in (
        'canonicalize_json_containers(data["provenance"])',
        'unused = canonicalize_json_containers(data["provenance"])',
        "unused = canonicalize_json_containers(data)",
        'copy = dict(data); copy["provenance"] = canonicalize_json_containers(data["provenance"])',
    ):
        assert _fixture_uncovered(
            f"""
def validate(cls, data):
    {statement}
    return data
""",
            {"provenance"},
        ) == {"provenance"}
    assert not _fixture_uncovered(
        """
def validate(cls, data):
    projected = canonicalize_json_containers(data['provenance'])
    data['provenance'] = projected
    return data
""",
        {"provenance"},
    )


def test_inherited_fields_and_local_overrides() -> None:
    tree = ast.parse("""
class Base:
    labels: tuple[str, ...]
    replaced: tuple[int, ...]
class Derived(Base):
    replaced: int
    own: tuple[int, ...]
""")
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    fields = _inherited_fields(classes["Derived"], classes)
    assert fields == {
        "labels": "tuple[str, ...]",
        "replaced": "int",
        "own": "tuple[int, ...]",
    }
    enforced = {
        name
        for name, annotation in fields.items()
        if _leaf_container_field(annotation, set(classes), {})
    }
    assert _fixture_uncovered(
        """
def validate(cls, data):
    data['own'] = canonicalize_json_containers(data['own'])
    return data
""",
        enforced,
    ) == {"labels"}


def test_uncalled_nested_definitions_do_not_establish_coverage() -> None:
    assert _fixture_uncovered(
        """
def validate(cls, data):
    def unused():
        return canonicalize_json_containers(data)
    return data
""",
        {"provenance"},
    ) == {"provenance"}


def test_overwritten_definitions_do_not_establish_coverage() -> None:
    assert _fixture_uncovered(
        """
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized = data
    return normalized
""",
        {"provenance"},
    ) == {"provenance"}


def test_conditional_projection_does_not_cover_raw_return_path() -> None:
    assert _fixture_uncovered(
        """
def validate(cls, data):
    if data.get('enabled'):
        data['provenance'] = canonicalize_json_containers(data['provenance'])
    return data
""",
        {"provenance"},
    ) == {"provenance"}


def test_projection_guard_must_establish_array_execution() -> None:
    for condition in (
        "data.get('enabled')",
        "not data.get('enabled')",
        "isinstance(data['provenance'], list) and data.get('enabled')",
    ):
        assert _fixture_uncovered(
            f"""
def project(data):
    return canonicalize_json_containers(data)
def validate(cls, data):
    if {condition}:
        return project(data)
    return data
""",
            {"provenance"},
        ) == {"provenance"}
    assert not _fixture_uncovered(
        """
def validate(cls, data):
    if isinstance(data, dict):
        axis = data.get('provenance')
        if isinstance(axis, (list, tuple)):
            data['provenance'] = canonicalize_json_containers(axis)
    return data
""",
        {"provenance"},
    )


def test_materializer_fast_path_and_used_definitions_remain_visible() -> None:
    assert not _fixture_uncovered(
        """
def materialize(value):
    if isinstance(value, (list, tuple)):
        return value
    return tuple(value)
def validate(cls, data):
    axis = data['provenance']
    axis = materialize(axis)
    if isinstance(axis, (list, tuple)):
        data['provenance'] = canonicalize_json_containers(axis)
    return data
""",
        {"provenance"},
    )
    assert not _fixture_uncovered(
        """
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    retained = normalized
    normalized = data
    return retained
""",
        {"provenance"},
    )


def test_projected_sources_are_not_credited_to_other_destination_fields() -> None:
    for statement in (
        'data["old"] = canonicalize_json_containers(data["provenance"])',
        'projected = canonicalize_json_containers(data["provenance"]); data["old"] = projected',
    ):
        assert _fixture_uncovered(
            f"""
def validate(cls, data):
    {statement}
    return data
""",
            {"provenance"},
        ) == {"provenance"}


def test_skippable_expression_and_loop_projections_do_not_cover_fields() -> None:
    for body in (
        'return canonicalize_json_containers(data) if data.get("enabled") else data',
        'for item in data.get("items", []):\n        data["provenance"] = canonicalize_json_containers(data["provenance"])\n    return data',
        'while data.get("enabled"):\n        data = canonicalize_json_containers(data)\n        break\n    return data',
    ):
        assert _fixture_uncovered(
            f"def validate(cls, data):\n    {body}\n", {"provenance"}
        ) == {"provenance"}


def test_inherited_validator_covers_new_subclass_fields() -> None:
    tree = ast.parse("""
class Base:
    old: tuple[str, ...]
    @model_validator(mode='before')
    def validate(cls, data):
        data['old'] = canonicalize_json_containers(data['old'])
        return data
class Child(Base):
    provenance: tuple[str, ...]
class Replaced(Child):
    @model_validator(mode='before')
    def validate(cls, data):
        return canonicalize_json_containers(data)
""")
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    fields = set(_inherited_fields(classes["Child"], classes))
    inherited = _inherited_validators(classes["Child"], classes)
    assert len(inherited) == 1
    assert _uncovered_leaf_fields(inherited[0], {}, fields) == {"provenance"}
    replaced = _inherited_validators(classes["Replaced"], classes)
    assert len(replaced) == 1
    assert not _uncovered_leaf_fields(replaced[0], {}, fields)


def test_exception_handlers_must_not_return_raw_payload() -> None:
    for handler, expected in (("return data", {"provenance"}), ("raise", set())):
        assert (
            _fixture_uncovered(
                f"""
def validate(cls, data):
    try:
        return canonicalize_json_containers(data)
    except ValueError:
        {handler}
""",
                {"provenance"},
            )
            == expected
        )


def test_projection_destination_survives_only_payload_preserving_helpers() -> None:
    assert not _fixture_uncovered(
        """
def admit(data):
    if not isinstance(data, dict):
        return data
    return dict(data)
def validate(cls, data):
    return admit(canonicalize_json_containers(data))
""",
        {"provenance"},
    )
    assert _fixture_uncovered(
        """
def project(value):
    return canonicalize_json_containers(value)
def validate(cls, data):
    data['old'] = project(data['provenance'])
    return data
""",
        {"provenance"},
    ) == {"provenance"}


def test_unconditional_projection_with_conditional_destination_is_uncovered() -> None:
    assert _fixture_uncovered(
        """
def validate(cls, data):
    projected = canonicalize_json_containers(data['provenance'])
    if data.get('enabled'):
        data['provenance'] = projected
    return data
""",
        {"provenance"},
    ) == {"provenance"}


def test_pep695_array_aliases_are_enforced(tmp_path: Path) -> None:
    (tmp_path / "models.py").write_text(
        "type Trail = tuple[str, ...]\n", encoding="utf-8"
    )
    aliases = _module_aliases(tmp_path / "models.py", tmp_path, {})
    assert _leaf_container_field("Trail", set(), aliases)
    assert _fixture_uncovered(
        """
def validate(cls, data):
    data['old'] = canonicalize_json_containers(data['old'])
    return data
""",
        {"old", "audit"},
    ) == {"audit"}


def test_overwritten_mapping_projection_does_not_establish_coverage() -> None:
    assert _fixture_uncovered(
        """
def validate(cls, data):
    raw = data['provenance']
    data['provenance'] = canonicalize_json_containers(raw)
    data['provenance'] = raw
    return data
""",
        {"provenance"},
    ) == {"provenance"}


def test_earlier_raw_return_cannot_be_covered_by_later_projection() -> None:
    assert _fixture_uncovered(
        """
def validate(cls, data):
    if data.get('enabled'):
        return data
    return canonicalize_json_containers(data)
""",
        {"provenance"},
    ) == {"provenance"}


def test_aliases_resolve_in_declaring_and_importing_modules(tmp_path: Path) -> None:
    array_module = tmp_path / "arrays.py"
    scalar_module = tmp_path / "scalars.py"
    importer = tmp_path / "consumer.py"
    array_module.write_text(
        "type Element = str\ntype Trail = tuple[Element, ...]\n", encoding="utf-8"
    )
    scalar_module.write_text("type Trail = int\n", encoding="utf-8")
    importer.write_text(
        "from arrays import Trail as ImportedTrail\ntype Element = int\n",
        encoding="utf-8",
    )
    cache: dict[Path, dict[str, set[str]]] = {}
    assert _leaf_container_field(
        "Trail", set(), _module_aliases(array_module, tmp_path, cache)
    )
    assert not _leaf_container_field(
        "Trail", set(), _module_aliases(scalar_module, tmp_path, cache)
    )
    assert _module_aliases(importer, tmp_path, cache)["ImportedTrail"] == {
        "tuple[str, ...]"
    }


def test_projection_before_early_return_still_covers_each_path() -> None:
    assert not _fixture_uncovered(
        """
def validate(cls, data):
    data = canonicalize_json_containers(data)
    if data.get('enabled'):
        return data
    return data
""",
        {"provenance"},
    )
    assert not _fixture_uncovered(
        """
def validate(cls, data):
    if not isinstance(data, dict):
        return data
    if data.get('enabled'):
        return canonicalize_json_containers(data)
    return canonicalize_json_containers(data)
""",
        {"provenance"},
    )


def test_conditional_clobber_invalidates_projected_destination() -> None:
    assert _fixture_uncovered(
        """
def validate(cls, data):
    data['provenance'] = canonicalize_json_containers(data['provenance'])
    if data.get('enabled'):
        data['provenance'] = []
    return data
""",
        {"provenance"},
    ) == {"provenance"}


def test_later_canonical_mapping_writes_preserve_coverage() -> None:
    for replacement in (
        "tuple(data['provenance'])",
        "data['provenance']",
        "normalize(data['provenance'])",
    ):
        assert not _fixture_uncovered(
            f"""
def normalize(value):
    return tuple(value) if isinstance(value, list) else value
def validate(cls, data):
    data['provenance'] = canonicalize_json_containers(data['provenance'])
    if data.get('enabled'):
        data['provenance'] = {replacement}
    return data
""",
            {"provenance"},
        )


def test_imported_array_normalizer_resolves_by_module(tmp_path: Path) -> None:
    helper = tmp_path / "helper.py"
    owner = tmp_path / "owner.py"
    helper.write_text(
        "def normalize(value):\n    return tuple(value) if isinstance(value, list) else value\n",
        encoding="utf-8",
    )
    owner.write_text(
        """
from helper import normalize as prepare
def validate(cls, data):
    data['provenance'] = canonicalize_json_containers(data['provenance'])
    if data.get('enabled'):
        data['provenance'] = prepare(data['provenance'])
    return data
""",
        encoding="utf-8",
    )
    functions = _module_functions(owner, tmp_path, {}, {})
    assert not _uncovered_leaf_fields(functions["validate"], functions, {"provenance"})


def test_unrelated_shape_refusal_does_not_exempt_raw_return() -> None:
    assert _fixture_uncovered(
        """
def validate(cls, data):
    if not isinstance(data.get('enabled'), bool):
        return data
    return canonicalize_json_containers(data)
""",
        {"provenance"},
    ) == {"provenance"}


def test_later_dictionary_entries_override_projected_unpack() -> None:
    assert _fixture_uncovered(
        """
def validate(cls, data):
    return {**canonicalize_json_containers(data), 'provenance': data['provenance']}
""",
        {"provenance"},
    ) == {"provenance"}


def test_conditional_payload_replacement_invalidates_projection() -> None:
    assert _fixture_uncovered(
        """
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    if data.get('enabled'):
        normalized = data
    return normalized
""",
        {"provenance"},
    ) == {"provenance"}


def test_later_projected_unpack_replaces_raw_entries() -> None:
    assert not _fixture_uncovered(
        """
def validate(cls, data):
    return {'provenance': data['provenance'], **canonicalize_json_containers(data)}
""",
        {"provenance"},
    )


def test_declared_refusals_must_match_the_expected_type() -> None:
    source = """
def validate(cls, data):
    if not isinstance(data.get('enabled'), bool):
        return data
    return canonicalize_json_containers(data)
"""
    assert not _fixture_uncovered(
        source, {"provenance"}, _ModelShape(types={"enabled": frozenset({"bool"})})
    )
    for expected in ({"int"}, {"bool", "NoneType"}):
        assert _fixture_uncovered(
            source, {"provenance"}, _ModelShape(types={"enabled": frozenset(expected)})
        ) == {"provenance"}


def test_model_shape_index_includes_nested_arrays_and_length_bounds() -> None:
    tree = ast.parse("""
class Child(StrictModel):
    rows: tuple[tuple[int, ...], ...]
class Parent(StrictModel):
    child: Child
    values: tuple[str, ...] = Field(max_length=MAX_VALUES)
""")
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    shapes = _json_shapes(classes["Parent"], classes, {})
    assert shapes.types["child.rows"] == frozenset({"list"})
    assert shapes.types["child.rows[]"] == frozenset({"list"})
    assert shapes.types["child.rows[][]"] == frozenset({"int"})
    assert shapes.max_lengths["values"] == ast.dump(
        ast.Name(id="MAX_VALUES", ctx=ast.Load())
    )


def test_only_declared_length_bounds_establish_projection() -> None:
    source = """
def validate(cls, data):
    raw = data['provenance']
    if len(raw) <= MAX_VALUES:
        data['provenance'] = tuple(raw)
    return data
"""
    assert _fixture_uncovered(source, {"provenance"}) == {"provenance"}
    assert not _fixture_uncovered(
        source,
        {"provenance"},
        _ModelShape(
            max_lengths={
                "provenance": ast.dump(ast.Name(id="MAX_VALUES", ctx=ast.Load()))
            }
        ),
    )


def test_unknown_payload_replacement_remains_uncovered() -> None:
    assert _fixture_uncovered(
        """
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    if data.get('enabled'):
        normalized = unknown(normalized)
    return normalized
""",
        {"provenance"},
    ) == {"provenance"}


def test_shallow_tuple_does_not_cover_nested_arrays() -> None:
    assert _fixture_uncovered(
        """
def validate(cls, data):
    data['matrix'] = tuple(data['matrix'])
    return data
""",
        {"matrix"},
        _ModelShape(
            types={"matrix": frozenset({"list"}), "matrix[]": frozenset({"list"})}
        ),
    ) == {"matrix"}


def test_import_alias_preserves_inherited_validator(tmp_path: Path) -> None:
    base = tmp_path / "base.py"
    child = tmp_path / "child.py"
    base.write_text(
        """
class Base(StrictModel):
    old: tuple[int, ...]
    @model_validator(mode='before')
    def validate(cls, data):
        data['old'] = canonicalize_json_containers(data['old'])
        return data
""",
        encoding="utf-8",
    )
    child.write_text(
        """
from base import Base as ImportedBase
class Child(ImportedBase):
    added: tuple[int, ...]
""",
        encoding="utf-8",
    )
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8")) for path in (base, child)
    }
    classes = {
        node.name: node
        for tree in trees.values()
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    classes.update(_module_classes(child, tmp_path, trees, {}))
    validators = _inherited_validators(classes["Child"], classes)
    assert len(validators) == 1
    assert _uncovered_leaf_fields(validators[0], {}, {"old", "added"}) == {"added"}


def test_nested_array_projection_requires_each_level() -> None:
    shapes = _ModelShape(
        types={"matrix": frozenset({"list"}), "matrix[]": frozenset({"list"})}
    )
    for expression in (
        "tuple(tuple(row) for row in data['matrix'])",
        "tuple(sorted(canonicalize_json_containers(data['matrix'])))",
    ):
        assert not _fixture_uncovered(
            f"""
def validate(cls, data):
    data['matrix'] = {expression}
    return data
""",
            {"matrix"},
            shapes,
        )
    assert _fixture_uncovered(
        """
def validate(cls, data):
    rows = data['matrix']
    if data.get('enabled'):
        rows = canonicalize_json_containers(rows)
    data['matrix'] = tuple(rows)
    return data
""",
        {"matrix"},
        shapes,
    ) == {"matrix"}


def test_existing_raw_builder_elements_are_not_projected_by_append() -> None:
    assert _fixture_uncovered(
        """
def validate(cls, data):
    rows = data['matrix']
    rows.append((1, 2))
    data['matrix'] = tuple(rows)
    return data
""",
        {"matrix"},
        _ModelShape(types={"matrix[]": frozenset({"list"})}),
    ) == {"matrix"}


def test_outer_array_of_models_keeps_leaf_admission_boundary() -> None:
    assert not _fixture_uncovered(
        """
def validate(cls, data):
    data['children'] = tuple(data['children'])
    return data
""",
        {"children"},
        _ModelShape(types={"children[]": frozenset({"dict"})}),
    )


def test_later_raw_return_prevents_projection_coverage() -> None:
    assert _fixture_uncovered(
        """
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    if data.get('enabled'):
        return data
    return normalized
""",
        {"provenance"},
    ) == {"provenance"}


def test_mapping_update_invalidates_projected_fields() -> None:
    assert _fixture_uncovered(
        """
def validate(cls, data):
    raw = data['provenance']
    data = canonicalize_json_containers(data)
    data.update({'provenance': raw})
    return data
""",
        {"provenance"},
    ) == {"provenance"}


def test_mapping_alias_mutations_invalidate_projection() -> None:
    for mutation in (
        "alias.update(provenance=raw)",
        "alias.__setitem__('provenance', raw)",
        "alias.setdefault('provenance', raw)",
        "alias.custom_mutator()",
        "alias.update({key: raw})",
    ):
        assert _fixture_uncovered(
            f"""
def validate(cls, data):
    raw = data['provenance']
    data = canonicalize_json_containers(data)
    alias = data
    {mutation}
    return data
""",
            {"provenance"},
        ) == {"provenance"}


def test_safe_mapping_updates_preserve_projection() -> None:
    for mutation in (
        "data.update({'enabled': True})",
        "data.update({'provenance': canonicalize_json_containers(raw)})",
        "data.update(provenance=tuple(raw))",
    ):
        assert not _fixture_uncovered(
            f"""
def validate(cls, data):
    raw = data['provenance']
    data = canonicalize_json_containers(data)
    {mutation}
    return data
""",
            {"provenance"},
        )


def test_only_proved_exit_paths_exempt_later_raw_returns() -> None:
    assert not _fixture_uncovered(
        """
def validate(cls, data):
    if isinstance(data, dict):
        try:
            return canonicalize_json_containers(data)
        except CanonicalizationError:
            raise ValueError()
    return data
""",
        {"provenance"},
    )
    assert _fixture_uncovered(
        """
def validate(cls, data):
    if data.get('enabled'):
        return canonicalize_json_containers(data)
    return data
""",
        {"provenance"},
    ) == {"provenance"}


def test_helper_side_effects_invalidate_incoming_projection() -> None:
    for final in ("return restore(data, raw)", "restore(data, raw)\n    return data"):
        assert _fixture_uncovered(
            f"""
def restore(data, raw):
    data.update({{'provenance': raw}})
    return data
def validate(cls, data):
    raw = data['provenance']
    data = canonicalize_json_containers(data)
    {final}
""",
            {"provenance"},
        ) == {"provenance"}


def test_augmented_union_invalidates_projection() -> None:
    assert _fixture_uncovered(
        """
def validate(cls, data):
    raw = data['provenance']
    data = canonicalize_json_containers(data)
    data |= {'provenance': raw}
    return data
""",
        {"provenance"},
    ) == {"provenance"}


def test_transitive_and_discarded_helper_mutations_are_visible() -> None:
    for operation in (
        "alias['provenance'] = raw",
        "alias |= {'provenance': raw}",
        "write = alias.update\n    write({'provenance': raw})",
        "unknown_mutator(alias, raw)",
    ):
        assert _fixture_uncovered(
            f"""
def mutate(data, raw):
    alias = data
    {operation}
def forward(data, raw):
    mutate(data, raw=raw)
    return data
def validate(cls, data):
    raw = data['provenance']
    data = canonicalize_json_containers(data)
    forward(data, raw)
    return data
""",
            {"provenance"},
        ) == {"provenance"}


def test_pure_helpers_and_preprojection_mutations_preserve_coverage() -> None:
    assert not _fixture_uncovered(
        """
def mutate(data):
    data['provenance'] = []
    return data
def validate(cls, data):
    return canonicalize_json_containers(mutate(data))
""",
        {"provenance"},
    )
    assert not _fixture_uncovered(
        """
def preserve(data):
    copied = dict(data)
    copied['provenance'] = tuple(copied['provenance'])
    copied |= {'enabled': True}
    return copied
def validate(cls, data):
    data = canonicalize_json_containers(data)
    return preserve(data)
""",
        {"provenance"},
    )


def test_safe_augmented_mapping_writes_remain_supported() -> None:
    for update in (
        "{'enabled': True}",
        "{'provenance': canonicalize_json_containers(raw)}",
    ):
        assert not _fixture_uncovered(
            f"""
def validate(cls, data):
    raw = data['provenance']
    data = canonicalize_json_containers(data)
    alias = data
    alias |= {update}
    return data
""",
            {"provenance"},
        )


def test_unrelated_none_guard_does_not_exempt_raw_return() -> None:
    assert _fixture_uncovered(
        """
def validate(cls, data):
    if data.get('enabled') is None:
        return data
    return canonicalize_json_containers(data)
""",
        {"provenance"},
    ) == {"provenance"}


def test_qualified_nested_arrays_require_deep_projection() -> None:
    tree = ast.parse("""
class Model(StrictModel):
    matrix: typing.Tuple[typing.Tuple[int, ...], ...]
""")
    model = tree.body[0]
    assert isinstance(model, ast.ClassDef)
    shapes = _json_shapes(model, {"Model": model}, {})
    assert _fixture_uncovered(
        """
def validate(cls, data):
    data['matrix'] = tuple(data['matrix'])
    return data
""",
        {"matrix"},
        shapes,
    ) == {"matrix"}


def test_destructuring_overwrites_invalidate_projection() -> None:
    assert _fixture_uncovered(
        """
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, ignored = data, None
    return normalized
""",
        {"provenance"},
    ) == {"provenance"}


def test_nested_destructuring_and_destructured_aliases_invalidate() -> None:
    for operation in (
        "[normalized, [ignored]] = [data, [None]]",
        "normalized['provenance'], ignored = data['provenance'], None",
        "alias, ignored = normalized, None\n    alias.update({'provenance': data['provenance']})",
        "write, ignored = normalized.update, None\n    write({'provenance': data['provenance']})",
        "normalized, ignored = unknown(data)",
    ):
        assert _fixture_uncovered(
            f"""
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    {operation}
    return normalized
""",
            {"provenance"},
        ) == {"provenance"}


def test_payload_element_from_helper_tuple_reaches_projection() -> None:
    assert not _fixture_uncovered(
        """
def preflight(data):
    payload = dict(data)
    return payload, 0
def validate(cls, data):
    data, count = preflight(data)
    return canonicalize_json_containers(data)
""",
        {"provenance"},
    )


def test_nested_annotation_wrappers_preserve_required_depth() -> None:
    for annotation in (
        "typing.Annotated[typing.Tuple[int, typing.Tuple[int, ...]], marker]",
        "typing.Optional[typing.Tuple[typing.Tuple[int, ...], ...]]",
        'typing.Tuple["typing.Tuple[int, ...]", ...]',
    ):
        model = ast.parse(
            f"class Model(StrictModel):\n    matrix: {annotation}\n"
        ).body[0]
        assert isinstance(model, ast.ClassDef)
        assert _fixture_uncovered(
            """
def validate(cls, data):
    data['matrix'] = tuple(data['matrix'])
    return data
""",
            {"matrix"},
            _json_shapes(model, {"Model": model}, {}),
        ) == {"matrix"}


def test_none_refusal_requires_field_dependency() -> None:
    assert not _fixture_uncovered(
        """
def validate(cls, data):
    if data['provenance'] is None:
        return data
    return canonicalize_json_containers(data)
""",
        {"provenance"},
    )
    assert _fixture_uncovered(
        """
def optional_result(data) -> tuple | None:
    return None if data.get('enabled') is None else ()
def validate(cls, data):
    result = optional_result(data)
    if result is None:
        return data
    return canonicalize_json_containers(data)
""",
        {"provenance"},
    ) == {"provenance"}


def test_annotation_and_projection_form_matrix() -> None:
    annotations = (
        ("", "tuple[tuple[int, ...], ...]"),
        ("", "typing.Tuple[typing.Tuple[int, ...], ...]"),
        ("", "t.Annotated[t.Optional[t.Tuple[t.Tuple[int, ...], ...]], marker]"),
        ("Rows = typing.Tuple[int, ...]\nMatrix = typing.Tuple[Rows, ...]\n", "Matrix"),
        ("Rows = tuple[int, ...]\n", '"typing.Tuple[Rows, ...]"'),
        ("type Rows = typing.Tuple[int, ...]\n", "typing.Tuple[Rows, ...]"),
    )
    forms: tuple[tuple[str, set[str]], ...] = (
        ("return canonicalize_json_containers(data)", set()),
        ("data['matrix'] = tuple(data['matrix'])\n    return data", {"matrix"}),
        (
            "normalized = canonicalize_json_containers(data)\n    if data.get('enabled') is None:\n        return data\n    return normalized",
            {"matrix"},
        ),
        (
            "normalized = canonicalize_json_containers(data)\n    normalized, ignored = data, None\n    return normalized",
            {"matrix"},
        ),
        (
            "normalized = canonicalize_json_containers(data)\n    normalized |= {'matrix': data['matrix']}\n    return normalized",
            {"matrix"},
        ),
        (
            "normalized = canonicalize_json_containers(data)\n    normalized.update({'matrix': data['matrix']})\n    return normalized",
            {"matrix"},
        ),
        (
            "normalized = canonicalize_json_containers(data)\n    alias, ignored = normalized, None\n    alias['matrix'] = data['matrix']\n    return normalized",
            {"matrix"},
        ),
    )
    for prefix, annotation in annotations:
        tree = ast.parse(
            f"{prefix}class Model(StrictModel):\n    matrix: {annotation}\n"
        )
        model = tree.body[-1]
        assert isinstance(model, ast.ClassDef)
        aliases = _declared_aliases(tree)
        assert _leaf_container_field(annotation, set(), aliases)
        shapes = _json_shapes(model, {"Model": model}, {model: aliases})
        assert "list" in shapes.types["matrix[]"]
        for body, expected in forms:
            assert (
                _fixture_uncovered(
                    f"def validate(cls, data):\n    {body}\n", {"matrix"}, shapes
                )
                == expected
            ), (annotation, body)


def test_projected_payload_survives_helper_tuple_destructuring() -> None:
    assert not _fixture_uncovered(
        """
def pair(payload):
    return payload, 0
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized)
    return normalized
""",
        {"provenance"},
    )


def test_helper_element_projection_follows_the_selected_slot() -> None:
    for returned in ("payload, raw", "[payload, raw]"):
        assert not _fixture_uncovered(
            f"""
def pair(payload, raw):
    result = {returned}
    return result
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, metadata = pair(normalized, raw=data)
    return normalized
""",
            {"provenance"},
        )
    assert _fixture_uncovered(
        """
def pair(payload, raw):
    return raw, payload
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, metadata = pair(normalized, data)
    return normalized
""",
        {"provenance"},
    ) == {"provenance"}


def test_helper_element_projection_rejects_mutations_and_raw_paths() -> None:
    for helper in (
        "def pair(payload, raw):\n    payload.update({'provenance': raw['provenance']})\n    return payload, 0",
        "def pair(payload, raw):\n    if raw.get('enabled'):\n        return raw, 0\n    return payload, 0",
        "def pair(payload, raw):\n    return unknown(payload), 0",
    ):
        assert _fixture_uncovered(
            f"""
{helper}
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, metadata = pair(normalized, data)
    return normalized
""",
            {"provenance"},
        ) == {"provenance"}


def test_helper_element_projection_accepts_canonical_return_paths() -> None:
    assert not _fixture_uncovered(
        """
def pair(payload):
    if payload.get('enabled'):
        return dict(payload), 1
    return payload, 0
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, metadata = pair(normalized)
    return normalized
""",
        {"provenance"},
    )


def test_conditionally_rebound_helper_tuple_is_not_projection_proof() -> None:
    assert _fixture_uncovered(
        """
def pair(payload, raw):
    result = (raw, 0)
    if raw.get('enabled'):
        result = (payload, 0)
    return result
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
        {"provenance"},
    ) == {"provenance"}


def test_helper_element_aliases_require_guaranteed_bindings() -> None:
    for returned in ("slot, 0", "dict(slot), 0"):
        assert _fixture_uncovered(
            f"""
def pair(payload, raw):
    slot = raw
    if raw.get('enabled'):
        slot = payload
    return {returned}
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
            {"provenance"},
        ) == {"provenance"}


def test_guaranteed_helper_tuple_rebindings_preserve_projection() -> None:
    for assignment in (
        "result = (payload, 0)",
        "if isinstance(payload, dict):\n        result = (payload, 0)",
    ):
        assert not _fixture_uncovered(
            f"""
def pair(payload, raw):
    result = (raw, 0)
    {assignment}
    return result
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
            {"provenance"},
        )


def test_conditional_raw_helper_tuple_remains_rejected() -> None:
    assert _fixture_uncovered(
        """
def pair(payload, raw):
    result = (payload, 0)
    if raw.get('enabled'):
        result = (raw, 0)
    return result
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
        {"provenance"},
    ) == {"provenance"}


def test_mutated_helper_list_slot_is_not_projected() -> None:
    assert _fixture_uncovered(
        """
def pair(payload, raw):
    result = [payload, 0]
    result[0] = raw
    return result
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
        {"provenance"},
    ) == {"provenance"}


def test_early_loop_exit_does_not_guarantee_tuple_binding() -> None:
    for control in ("break", "continue"):
        assert _fixture_uncovered(
            f"""
def pair(payload, raw):
    result = (raw, 0)
    for unused in [1]:
        if raw.get('enabled'):
            {control}
        result = (payload, 0)
    return result
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
            {"provenance"},
        ) == {"provenance"}


def test_helper_sequence_mutations_through_aliases_are_rejected() -> None:
    for mutation in (
        "alias[0] = raw",
        "alias[:] = [raw, 0]",
        "alias.__setitem__(0, raw)",
        "setter = alias.__setitem__\n    setter(0, raw)",
        "alias.insert(0, raw)",
        "alias.reverse()",
        "unknown_mutator(alias, raw)",
    ):
        assert _fixture_uncovered(
            f"""
def pair(payload, raw):
    result = [payload, 0]
    alias = result
    {mutation}
    return alias
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
            {"provenance"},
        ) == {"provenance"}


def test_unchanged_selected_sequence_slot_remains_supported() -> None:
    for preparation in ("result[1] = 1", "count = len(result)", "alias = result"):
        assert not _fixture_uncovered(
            f"""
def pair(payload):
    result = [payload, 0]
    {preparation}
    return result
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized)
    return normalized
""",
            {"provenance"},
        )


def test_loop_control_after_guaranteed_binding_remains_supported() -> None:
    for control in ("break", "continue"):
        assert not _fixture_uncovered(
            f"""
def pair(payload, raw):
    result = (raw, 0)
    for unused in [1]:
        result = (payload, 0)
        {control}
    return result
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
            {"provenance"},
        )


def test_shared_sequence_elements_and_shadowed_calls_are_not_read_only() -> None:
    for preparation in (
        "result = [payload, payload]\n    result[1] |= {'provenance': raw['provenance']}",
        "result = [payload, 0]\n    len(result, raw)",
    ):
        assert _fixture_uncovered(
            f"""
def len(values, raw):
    values[0] = raw
def pair(payload, raw):
    {preparation}
    return result
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
            {"provenance"},
        ) == {"provenance"}


def test_loop_target_rebinding_sequence_is_rejected() -> None:
    assert _fixture_uncovered(
        """
def pair(payload, raw):
    result = (payload, 0)
    for result in [(raw, 0)]:
        pass
    return result
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
        {"provenance"},
    ) == {"provenance"}


def test_container_held_sequence_alias_is_rejected() -> None:
    assert _fixture_uncovered(
        """
def pair(payload, raw):
    result = [payload, 0]
    holder = [result]
    holder[0][0] = raw
    return result
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
        {"provenance"},
    ) == {"provenance"}


def test_positional_only_helper_arguments_keep_their_order() -> None:
    for returned, uncovered in (("payload", set()), ("raw", {"provenance"})):
        assert (
            _fixture_uncovered(
                f"""
def pair(payload, /, raw):
    return {returned}, 0
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
                {"provenance"},
            )
            == uncovered
        )


def test_pattern_capture_invalidates_canonical_helper_parameter() -> None:
    assert _fixture_uncovered(
        """
def pair(payload, raw):
    match [raw]:
        case [payload]:
            pass
    return payload, 0
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
        {"provenance"},
    ) == {"provenance"}


def test_context_manager_target_invalidates_stored_sequence() -> None:
    assert _fixture_uncovered(
        """
def pair(payload, raw):
    result = (payload, 0)
    with context as result:
        pass
    return result
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
        {"provenance"},
    ) == {"provenance"}


def test_compound_sequence_alias_mutations_are_rejected() -> None:
    for alias in ("result or []", "result if raw else []"):
        assert _fixture_uncovered(
            f"""
def pair(payload, raw):
    result = [payload, 0]
    alias = {alias}
    alias[0] = raw
    return result
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
            {"provenance"},
        ) == {"provenance"}


def test_captured_helper_mutation_is_rejected() -> None:
    assert _fixture_uncovered(
        """
def pair(payload, raw):
    def mutate():
        payload['provenance'] = raw['provenance']
    mutate()
    return payload, 0
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
        {"provenance"},
    ) == {"provenance"}


def test_starred_empty_loop_does_not_guarantee_binding() -> None:
    assert _fixture_uncovered(
        """
def pair(payload, raw):
    result = (raw, 0)
    for unused in [*[]]:
        result = (payload, 0)
    return result
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
        {"provenance"},
    ) == {"provenance"}


def test_import_rebindings_invalidate_canonical_parameters() -> None:
    for statement in (
        "from elsewhere import raw_payload as payload",
        "import elsewhere as payload",
    ):
        assert _fixture_uncovered(
            f"""
def pair(payload, raw):
    {statement}
    return payload, 0
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
            {"provenance"},
        ) == {"provenance"}


def test_nonassignment_subscript_targets_invalidate_projection() -> None:
    for write in (
        "for payload['provenance'] in [raw['provenance']]:\n        pass",
        "with context as payload['provenance']:\n        pass",
    ):
        assert _fixture_uncovered(
            f"""
def pair(payload, raw):
    {write}
    return payload, 0
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
            {"provenance"},
        ) == {"provenance"}


def test_shadowed_structural_guard_does_not_guarantee_binding() -> None:
    assert _fixture_uncovered(
        """
def isinstance(value, kind):
    return False
def pair(payload, raw):
    result = (raw, 0)
    if isinstance(payload, dict):
        result = (payload, 0)
    return result
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
        {"provenance"},
    ) == {"provenance"}


def test_locally_rebound_tuple_helper_is_not_trusted() -> None:
    assert _fixture_uncovered(
        """
def pair(payload, raw):
    return payload, 0
def evil(payload, raw):
    return raw, 0
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    pair = evil
    normalized, count = pair(normalized, data)
    return normalized
""",
        {"provenance"},
    ) == {"provenance"}


def test_container_held_payload_alias_mutation_is_rejected() -> None:
    for setup, write in (
        (
            "holder = {'value': payload}",
            "holder['value']['provenance'] = raw['provenance']",
        ),
        ("holder = [payload]", "holder[0]['provenance'] = raw['provenance']"),
    ):
        assert _fixture_uncovered(
            f"""
def pair(payload, raw):
    {setup}
    {write}
    return payload, 0
def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
            {"provenance"},
        ) == {"provenance"}


def test_shadowing_a_projection_constructor_is_not_projection_proof() -> None:
    """A module-level ``dict`` replaces the builtin the tracer would trust."""

    for replacement in ("dict", "cast"):
        assert _fixture_uncovered(
            f"""
def {replacement}(value, *rest):
    value['provenance'] = raw
    return value

def pair(payload, raw):
    return {replacement}(payload), 0

def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
            {"provenance"},
        ) == {"provenance"}


def test_attribute_stored_canonical_value_is_an_escape() -> None:
    """A canonical value in an attribute can be written back through unseen."""

    assert _fixture_uncovered(
        """
class Holder:
    pass

def pair(payload, raw):
    holder = Holder()
    holder.value = payload
    holder.value['provenance'] = raw['provenance']
    return payload, 0

def validate(cls, data):
    normalized = canonicalize_json_containers(data)
    normalized, count = pair(normalized, data)
    return normalized
""",
        {"provenance"},
    ) == {"provenance"}


@pytest.mark.parametrize("base_projects", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_inherited_validator_helpers_use_defining_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    base_projects: bool,
    override: bool,
) -> None:
    """A subclass cannot hide or supply its inherited validator's globals."""

    base_return = "canonicalize_json_containers(data)" if base_projects else "data"
    child_return = "data" if base_projects else "canonicalize_json_containers(data)"
    (tmp_path / "base.py").write_text(
        f"""
def project(data):
    return {base_return}

class Base(StrictModel):
    @model_validator(mode='before')
    def validate(cls, data):
        return project(data)
""",
        encoding="utf-8",
    )
    override_source = (
        "    @model_validator(mode='before')\n"
        "    def validate(cls, data):\n"
        "        return project(data)\n"
        if override
        else ""
    )
    (tmp_path / "child.py").write_text(
        f"""
from base import Base as ImportedBase

def project(data):
    return {child_return}

class Child(ImportedBase):
    entries: tuple[tuple[int, ...], ...]
{override_source}
""",
        encoding="utf-8",
    )
    tables = _build_tables(tmp_path)
    monkeypatch.setitem(globals(), "_build_tables", lambda root: tables)
    if base_projects != override:
        test_before_validators_cover_every_leaf_container_field()
    else:
        with pytest.raises(AssertionError, match=r"Child\.entries is never projected"):
            test_before_validators_cover_every_leaf_container_field()


@pytest.mark.parametrize("base_first", [False, True])
@pytest.mark.parametrize("projects", [False, True])
def test_function_scope_cache_retains_imported_validator_helpers(
    tmp_path: Path, base_first: bool, projects: bool
) -> None:
    """Looking up an importer must not replace a later owner's full scope."""

    result = "canonicalize_json_containers(data)" if projects else "data"
    (tmp_path / "helpers.py").write_text(
        f"def project(data):\n    return {result}\n", encoding="utf-8"
    )
    (tmp_path / "base.py").write_text(
        "from helpers import project\n"
        "class Base(StrictModel):\n"
        "    entries: tuple[int, ...]\n"
        "    @model_validator(mode='before')\n"
        "    def validate(cls, data):\n"
        "        return project(data)\n",
        encoding="utf-8",
    )
    (tmp_path / "child.py").write_text(
        "from base import Base\n"
        "def project(data):\n"
        "    return canonicalize_json_containers(data)\n"
        "class Child(Base):\n"
        "    pass\n",
        encoding="utf-8",
    )
    tables = _build_tables(tmp_path)
    cache: dict[str, dict[str, ast.FunctionDef]] = {}
    order = ("base.py", "child.py") if base_first else ("child.py", "base.py")
    for module in order:
        _table_functions(module, tables, tables.texts, cache)
    validator = _table_validator(("base.py", "Base", "validate"), tables, tables.texts)
    uncovered = _uncovered_leaf_fields(
        validator, _table_functions("base.py", tables, tables.texts, cache), {"entries"}
    )
    assert uncovered == (set() if projects else {"entries"})


def test_inherited_partial_projection_keeps_new_subclass_fields_uncovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "base.py").write_text(
        "def project(data):\n"
        "    data['entries'] = canonicalize_json_containers(data['entries'])\n"
        "    return data\n"
        "class Base(StrictModel):\n"
        "    entries: tuple[int, ...]\n"
        "    @model_validator(mode='before')\n"
        "    def validate(cls, data):\n"
        "        return project(data)\n",
        encoding="utf-8",
    )
    (tmp_path / "child.py").write_text(
        "from base import Base\nclass Child(Base):\n    extra: tuple[int, ...]\n",
        encoding="utf-8",
    )
    tables = _build_tables(tmp_path)
    monkeypatch.setitem(globals(), "_build_tables", lambda root: tables)
    with pytest.raises(AssertionError, match=r"Child\.extra is never projected"):
        test_before_validators_cover_every_leaf_container_field()
