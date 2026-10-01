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


def _global_type_index(root: Path) -> tuple[set[str], dict[str, set[str]]]:
    """Every class name plus every module-level ``Name = ...`` alias source."""

    classes: set[str] = set()
    aliases: dict[str, set[str]] = {}
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.add(node.name)
            elif isinstance(node, ast.Assign) and node.value is not None:
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
    return classes, aliases


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


def _shadowed_assignments(nodes: list[ast.AST]) -> set[int]:
    """Assignments whose write cannot reach any return.

    An earlier write to a name is dead only when a later write to the same
    name sits in the same straight-line statement list with no return, loop,
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
                read.id
                for read in ast.walk(statement)
                if isinstance(read, ast.Name) and isinstance(read.ctx, ast.Load)
            }
            seen = {name: write for name, write in seen.items() if name not in reads}
            if isinstance(statement, ast.Assign):
                targets: list[ast.expr] = list(statement.targets)
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                targets = [statement.target]
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    if target.id in seen:
                        dead.add(id(seen[target.id]))
                    seen[target.id] = statement
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
    ) -> None:
        scoped = _own_nodes(function)
        dead = _shadowed_assignments(scoped)
        self.nodes = [node for node in scoped if id(node) not in dead]
        self.functions = functions
        self.arguments = arguments
        self.covered = covered
        self.stack = stack
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
                helper, self.functions, arguments, helper_covered, self.stack | {name}
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

    def record(self, node: ast.AST, fields: set[str] | frozenset[str]) -> None:
        """Unknown conditional execution cannot prove projection coverage."""

        if self.checking_guards:
            return
        self.checking_guards = True
        try:
            for field in fields:
                if self.execution_guaranteed(node, field):
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
    ).analyze()
    return set() if "*" in covered else leaf_fields - covered


def _inherited_fields(
    model: ast.ClassDef,
    classes: dict[str, ast.ClassDef],
    seen: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Collect known base declarations, preserving local annotation overrides."""

    if model.name in seen:
        return {}
    fields: dict[str, str] = {}
    for base in reversed(model.bases):
        parent = classes.get(getattr(base, "id", ""))
        if parent is not None:
            fields.update(_inherited_fields(parent, classes, seen | {model.name}))
    fields.update(
        {
            item.target.id: ast.unparse(item.annotation)
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
    classes, aliases = _global_type_index(source_root)
    violations: list[str] = []
    definitions: dict[str, list[ast.ClassDef]] = {}
    for path in sorted(source_root.rglob("*.py")):
        for definition in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(definition, ast.ClassDef):
                definitions.setdefault(definition.name, []).append(definition)
    unique_classes = {
        name: nodes[0] for name, nodes in definitions.items() if len(nodes) == 1
    }
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        known_classes = {
            **unique_classes,
            **{node.name: node for node in tree.body if isinstance(node, ast.ClassDef)},
        }
        functions: dict[str, ast.FunctionDef] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.setdefault(node.name, node)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            before = _inherited_validators(node, known_classes)
            if not before:
                continue
            fields = _inherited_fields(node, known_classes)
            leaf_fields = {
                name
                for name, annotation in fields.items()
                if _leaf_container_field(annotation, classes, aliases)
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
