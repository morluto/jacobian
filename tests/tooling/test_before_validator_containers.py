"""Guard strict-JSON container handling in owner-local preflight validators."""

from __future__ import annotations

import ast
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


def _module_aliases(
    path: Path, root: Path, cache: dict[Path, dict[str, set[str]]]
) -> dict[str, set[str]]:
    """Resolve explicit imports in their source module before applying local names."""

    if path in cache:
        return cache[path]
    cache[path] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        imported_path = _import_path(node, path, root)
        if imported_path is None:
            continue
        imported = _module_aliases(imported_path, root, cache)
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

    return visit(ast.parse(source, mode="eval").body, frozenset())


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
    ) -> None:
        scoped = _own_nodes(function)
        dead = _shadowed_assignments(scoped)
        self.nodes = [node for node in scoped if id(node) not in dead]
        self.functions = functions
        self.arguments = arguments
        self.covered = covered
        self.stack = stack
        self.required_fields = required_fields
        self.live_cache: set[ast.AST] | None = None
        self.checking_guards = False
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

    def binding(self, node: ast.Name, seen: frozenset[str]) -> frozenset[str]:
        bindings = [
            part.value
            for part in self.nodes
            if isinstance(part, (ast.Assign, ast.AnnAssign))
            and part.value is not None
            and (part.value.end_lineno or part.lineno, part.value.end_col_offset or 0)
            < (node.lineno, node.col_offset)
            and any(
                isinstance(target, ast.Name) and target.id == node.id
                for target in (
                    part.targets if isinstance(part, ast.Assign) else [part.target]
                )
            )
        ]
        if bindings:
            return self.value(
                max(bindings, key=lambda item: item.lineno), seen | {node.id}
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
                    helper.args.args, node.args, strict=False
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
        if isinstance(node, ast.Name):
            return self.binding(node, seen)
        if isinstance(node, ast.Subscript) and self.value(node.value, seen) == _WHOLE:
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
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    receiver = (
                        target.value if isinstance(target, ast.Subscript) else target
                    )
                    if isinstance(receiver, ast.Name) and receiver.id in names:
                        live.add(node)
                        live.update(ast.walk(node.value))
            if len(live) == previous:
                self.live_cache = live
                return live

    def guard_value(self, test: ast.expr, field: str) -> bool | None:
        """Evaluate only structural guards known for a present JSON array field."""

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
            isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
            and isinstance(test.ops[0], (ast.Is, ast.IsNot))
            and self.value(test.left) in {_WHOLE, frozenset({field})}
        ):
            return isinstance(test.ops[0], ast.IsNot)
        if (
            isinstance(test.ops[0], ast.In)
            and self.keys(test.left) == frozenset({field})
            and self.value(test.comparators[0]) == _WHOLE
        ):
            return True
        if (
            isinstance(test.left, ast.Call)
            and getattr(test.left.func, "id", "") == "type"
            and len(test.left.args) == 1
            and isinstance(test.ops[0], (ast.Is, ast.IsNot))
        ):
            origin = self.value(test.left.args[0])
            expected = (
                "dict"
                if origin == _WHOLE
                else "list"
                if origin == frozenset({field})
                else None
            )
            if expected is not None:
                equal = getattr(test.comparators[0], "id", "") == expected
                return equal if isinstance(test.ops[0], ast.Is) else not equal
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
            return set(fields)
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
                    result.update(self.keys(target.slice))
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
            for key, value in zip(parent.keys, parent.values, strict=True):
                if value is node and key is not None:
                    return set(self.keys(key))
        if isinstance(parent, (ast.Dict, ast.IfExp)):
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

    def refusal_guard(self, test: ast.expr) -> bool:
        """Recognize type/shape failures that already make strict input invalid."""

        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return (
                isinstance(test.operand, ast.Call)
                and getattr(test.operand.func, "id", "") == "isinstance"
            )
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
            return all(self.refusal_guard(value) for value in test.values)
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            if isinstance(test.ops[0], ast.NotIn):
                return isinstance(test.left, ast.Constant) or (
                    isinstance(test.left, ast.Call)
                    and getattr(test.left.func, "id", "") == "type"
                )
            if (
                isinstance(test.left, ast.Call)
                and getattr(test.left.func, "id", "") == "len"
            ):
                bound = test.comparators[0]
                if (
                    isinstance(test.ops[0], (ast.Gt, ast.GtE))
                    and isinstance(bound, ast.Name)
                    and bound.id.isupper()
                ):
                    return True
            return (
                isinstance(test.ops[0], ast.IsNot)
                and isinstance(test.left, ast.Call)
                and getattr(test.left.func, "id", "") == "type"
            ) or (
                isinstance(test.ops[0], ast.Is)
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value is None
            )
        return False

    def projected_subscript(
        self, value: ast.Subscript, field: str, seen: frozenset[int]
    ) -> bool:
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

    def projected_call(self, value: ast.Call, field: str, seen: frozenset[int]) -> bool:
        name = getattr(value.func, "id", "")
        return (
            name in {"canonicalize_json_containers", "tuple"}
            or self.normalizes_outer_array(name)
            or (
                name in {"dict", "cast"}
                and bool(value.args)
                and self.projected_return(value.args[-1], field, seen | {id(value)})
            )
        )

    def projected_return(
        self, value: ast.expr, field: str, seen: frozenset[int] = frozenset()
    ) -> bool:
        if id(value) in seen:
            return False
        if isinstance(value, ast.Tuple):
            return True
        if isinstance(value, ast.Subscript):
            return self.projected_subscript(value, field, seen)
        if isinstance(value, ast.Call):
            return self.projected_call(value, field, seen)
        if isinstance(value, ast.Dict):
            for key, item in reversed(list(zip(value.keys, value.values, strict=True))):
                if key is not None and field in self.keys(key):
                    return isinstance(item, ast.Call) and getattr(
                        item.func, "id", ""
                    ) in {"tuple", "canonicalize_json_containers"}
                if key is None and self.projected_return(
                    item, field, seen | {id(value)}
                ):
                    return True
        if isinstance(value, ast.Name):
            writes = [
                node
                for node in self.nodes
                if isinstance(node, ast.Assign) and node.lineno < value.lineno
            ]
            for write in sorted(writes, key=lambda item: item.lineno, reverse=True):
                for target in write.targets:
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
                        return isinstance(write.value, ast.Call) and getattr(
                            write.value.func, "id", ""
                        ) in {"tuple", "canonicalize_json_containers"}
        return False

    def earlier_raw_return(self, node: ast.AST, field: str) -> bool:
        for returned in self.nodes:
            if not isinstance(returned, ast.Return) or returned.lineno >= getattr(
                node, "lineno", 0
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
                        known is None and in_body and self.refusal_guard(parent.test)
                    ):
                        invalid_shape = True
                    if (
                        not in_body
                        and isinstance(parent.test, ast.Call)
                        and getattr(parent.test.func, "id", "") == "isinstance"
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
                    isinstance(parent.iter, (ast.Tuple, ast.List)) and parent.iter.elts
                ):
                    return False
                if child in parent.orelse:
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

    def overwritten_after(self, node: ast.AST, field: str) -> bool:
        """A possibly executed later raw write invalidates earlier projection."""

        location = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
        for write in self.nodes:
            if (
                not isinstance(write, (ast.Assign, ast.AnnAssign))
                or write.value is None
                or write not in self.returned_nodes()
            ):
                continue
            if (write.lineno, write.col_offset) <= location:
                continue
            targets = write.targets if isinstance(write, ast.Assign) else [write.target]
            for target in targets:
                if (
                    isinstance(target, ast.Subscript)
                    and self.value(target.value) == _WHOLE
                    and (
                        not self.keys(target.slice) or field in self.keys(target.slice)
                    )
                    and not self.projected_return(write.value, field)
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
                    not self.earlier_raw_return(node, field)
                    and self.execution_guaranteed(node, field)
                    and not self.overwritten_after(node, field)
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
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", "")
                in {"canonicalize_json_containers", "tuple"}
            ):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and self.value(target.value) == _WHOLE
                    ):
                        self.record(node, self.keys(target.slice))
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
    ).analyze()
    return set() if "*" in covered else leaf_fields - covered


def _inherited_fields(
    model: ast.ClassDef,
    classes: dict[str, ast.ClassDef],
    seen: frozenset[str] = frozenset(),
    contexts: dict[ast.ClassDef, dict[str, set[str]]] | None = None,
) -> dict[str, str]:
    """Collect known base declarations, preserving local annotation overrides."""

    if model.name in seen:
        return {}
    fields: dict[str, str] = {}
    for base in reversed(model.bases):
        parent = classes.get(getattr(base, "id", ""))
        if parent is not None:
            fields.update(
                _inherited_fields(parent, classes, seen | {model.name}, contexts)
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
) -> list[ast.FunctionDef]:
    if model.name in seen:
        return []
    methods: dict[str, ast.FunctionDef] = {}
    for base in reversed(model.bases):
        parent = classes.get(getattr(base, "id", ""))
        if parent is not None:
            methods.update(
                {
                    method.name: method
                    for method in _inherited_validators(
                        parent, classes, seen | {model.name}
                    )
                }
            )
    for method in model.body:
        if isinstance(method, ast.FunctionDef):
            methods.pop(method.name, None)
            if any(_is_before_validator(item) for item in method.decorator_list):
                methods[method.name] = method
    return list(methods.values())


def test_before_validators_cover_every_leaf_container_field() -> None:
    """Every declared outer array needs projection, including arrays of models.

    Bare nested models retain their own admission/canonicalization boundary.
    Unsupported annotations are left unenforced; textual field mentions never
    establish projection coverage.
    """

    source_root = Path(__file__).parents[2] / "src" / "jacobian" / "math"
    violations: list[str] = []
    definitions: dict[str, list[ast.ClassDef]] = {}
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(source_root.rglob("*.py"))
    }
    cache: dict[Path, dict[str, set[str]]] = {}
    contexts: dict[ast.ClassDef, dict[str, set[str]]] = {}
    for path, tree in trees.items():
        module_aliases = _module_aliases(path, source_root, cache)
        for definition in tree.body:
            if isinstance(definition, ast.ClassDef):
                definitions.setdefault(definition.name, []).append(definition)
                contexts[definition] = module_aliases
    unique_classes = {
        name: nodes[0] for name, nodes in definitions.items() if len(nodes) == 1
    }
    function_cache: dict[Path, dict[str, ast.FunctionDef]] = {}
    for path, tree in trees.items():
        known_classes = {
            **unique_classes,
            **{node.name: node for node in tree.body if isinstance(node, ast.ClassDef)},
        }
        functions = dict(_module_functions(path, source_root, trees, function_cache))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.setdefault(node.name, node)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            before = _inherited_validators(node, known_classes)
            if not before:
                continue
            fields = _inherited_fields(node, known_classes, contexts=contexts)
            leaf_fields = {
                name
                for name, annotation in fields.items()
                if _leaf_container_field(annotation, set(), {})
            }
            if not leaf_fields:
                continue
            for validator in before:
                uncovered = _uncovered_leaf_fields(validator, functions, leaf_fields)
                for name in sorted(uncovered):
                    violations.append(
                        f"{path.relative_to(source_root)}:{validator.lineno} "
                        f"{node.name}.{name} is never projected"
                    )

    assert not violations, (
        "before-validator projections must reach every leaf container field; "
        "project the payload, derive ownership from cls.model_fields, or name "
        "the field in reachable projection code: " + ", ".join(sorted(violations))
    )


def _fixture_uncovered(source: str, fields: set[str]) -> set[str]:
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    return _uncovered_leaf_fields(functions["validate"], functions, fields)


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
