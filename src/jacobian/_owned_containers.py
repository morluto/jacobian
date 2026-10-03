"""Bounded projection plans for the supported strict-model ownership grammar.

This module reads resolved core schemas, never runs validators to discover shape.
Plans are request-local so ``model_rebuild`` cannot leave a stale ownership cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import AliasChoices, AliasPath, BaseModel, ValidationError

from jacobian.canonical import CanonicalizationError

if TYPE_CHECKING:
    from jacobian._models import StrictModel

_MAX_PROJECTION_DEPTH = 64
_MAX_PROJECTION_CELLS = 100_000
_MAX_SCHEMA_NODES = 1_024
_SCALARS = frozenset(
    {
        "bool",
        "int",
        "float",
        "str",
        "bytes",
        "none",
        "literal",
        "decimal",
        "date",
        "datetime",
        "time",
        "timedelta",
        "uuid",
        "url",
        "multi-host-url",
    }
)


@dataclass
class _Plan:
    kind: str
    fields: list[_Field] = field(default_factory=list)
    children: list[_Plan] = field(default_factory=list)
    minimum: int = 0
    maximum: int = 0
    variadic: bool = False
    discriminators: tuple[str, ...] = ()
    choices: dict[str, _Plan] = field(default_factory=dict)


@dataclass(frozen=True)
class _Field:
    name: str
    aliases: tuple[str, ...]
    plan: _Plan
    loc_by_alias: bool


def _unsupported(detail: str) -> TypeError:
    return TypeError(f"unsupported owned-container schema: {detail}")


def _alias_names(alias: Any) -> tuple[str, ...]:
    if type(alias) is str:
        return (alias,)
    if type(alias) is list and all(
        type(path) is list and len(path) == 1 and type(path[0]) is str for path in alias
    ):
        return tuple(path[0] for path in alias)
    raise _unsupported(
        "aliases/discriminators must be field names, not paths or callables"
    )


class _Compiler:
    def __init__(self) -> None:
        self.definitions: dict[str, Any] = {}
        self.plans: dict[int, _Plan] = {}

    def compile(self, schema: Any, depth: int = 0) -> _Plan:
        if depth > _MAX_PROJECTION_DEPTH or len(self.plans) >= _MAX_SCHEMA_NODES:
            raise _unsupported("schema exceeds the supported depth or node budget")
        if type(schema) is not dict:
            raise _unsupported("model must have a completed core schema")
        if schema["type"] == "definitions":
            self.definitions.update(
                (item["ref"], item) for item in schema["definitions"]
            )
            return self.compile(schema["schema"], depth + 1)
        if schema["type"] == "definition-ref":
            target = self.definitions.get(schema["schema_ref"])
            if target is None or not self._is_model(target):
                raise _unsupported(
                    "only model references are supported, not recursive aliases"
                )
            return self.compile(target, depth + 1)
        if id(schema) in self.plans:
            return self.plans[id(schema)]
        kind = schema["type"]
        plan = _Plan(kind)
        self.plans[id(schema)] = plan
        self._populate(plan, schema, depth)
        return plan

    def _is_model(self, schema: Any) -> bool:
        seen: set[int] = set()
        while id(schema) not in seen:
            seen.add(id(schema))
            if schema["type"] == "function-after":
                schema = schema["schema"]
            elif schema["type"] == "definition-ref":
                schema = self.definitions.get(schema["schema_ref"], {})
            else:
                return bool(schema.get("type") == "model")
        return False

    def _populate(self, plan: _Plan, schema: Any, depth: int) -> None:
        kind = plan.kind
        if kind in _SCALARS:
            plan.kind = "scalar"
        elif kind in {"default", "nullable"}:
            plan.children = [self.compile(schema["schema"], depth + 1)]
        elif kind == "model":
            self._model(plan, schema, depth)
        elif kind == "tuple":
            self._tuple(plan, schema, depth)
        elif kind == "tagged-union":
            self._tagged(plan, schema, depth)
        elif kind == "union":
            plan.children = [
                self.compile(choice, depth + 1) for choice in schema["choices"]
            ]
            if any(child.kind != "scalar" for child in plan.children):
                raise _unsupported("untagged structured unions")
            plan.kind = "scalar"
        elif kind == "function-after" and self._is_model(schema):
            plan.children = [self.compile(schema["schema"], depth + 1)]
        else:
            raise _unsupported(
                f"{kind!r}; container transformations require an owner adapter"
            )

    def _model(self, plan: _Plan, schema: Any, depth: int) -> None:
        from jacobian._models import StrictModel

        model = schema["cls"]
        if not issubclass(model, StrictModel) or schema.get("root_model"):
            raise _unsupported("only ordinary StrictModel envelopes are supported")
        if model.__pydantic_generic_metadata__.get(
            "origin"
        ) or model.__pydantic_generic_metadata__.get("parameters"):
            raise _unsupported("generic models")
        if any(
            "__get_pydantic_core_schema__" in base.__dict__
            for base in model.__mro__
            if base not in {BaseModel, object}
        ):
            raise _unsupported("custom model core-schema hooks")
        config = schema.get("config", {})
        if config.get("extra_fields_behavior") != "forbid":
            raise _unsupported("model envelopes must forbid extras")
        fields = schema["schema"]
        # Model preflights are callers of this primitive, not shape transforms.
        # Do not execute them (which would recurse back into compilation).
        while fields["type"] == "function-before":
            fields = fields["schema"]
        if fields["type"] != "model-fields":
            raise _unsupported("model wrap/plain validators")
        used_aliases: set[str] = set()
        for name, item in fields["fields"].items():
            alias = model.model_fields[name].validation_alias
            if isinstance(alias, AliasPath) or (
                isinstance(alias, AliasChoices)
                and any(isinstance(choice, AliasPath) for choice in alias.choices)
            ):
                raise _unsupported("AliasPath carriers are not closed model envelopes")
            names = (
                _alias_names(item["validation_alias"])
                if "validation_alias" in item and config.get("validate_by_alias", True)
                else ()
            )
            if "validation_alias" not in item or config.get("validate_by_name", False):
                names = (*names, name)
            if used_aliases.intersection(names):
                raise _unsupported("input aliases shared by multiple fields")
            used_aliases.update(names)
            plan.fields.append(
                _Field(
                    name,
                    names,
                    self.compile(item["schema"], depth + 1),
                    config.get("loc_by_alias", True),
                )
            )

    def _tuple(self, plan: _Plan, schema: Any, depth: int) -> None:
        items = schema["items_schema"]
        variadic = schema.get("variadic_item_index")
        if variadic is not None and (variadic != 0 or len(items) != 1):
            raise _unsupported("mixed fixed/variadic tuples")
        plan.variadic = variadic is not None
        if plan.variadic and "max_length" not in schema:
            raise _unsupported("variadic tuples need an explicit max_length")
        plan.minimum = schema.get("min_length", 0) if plan.variadic else len(items)
        plan.maximum = schema.get("max_length", len(items))
        if not plan.variadic:
            plan.minimum = max(plan.minimum, schema.get("min_length", 0))
            plan.maximum = min(plan.maximum, len(items))
        plan.children = [self.compile(item, depth + 1) for item in items]

    def _tagged(self, plan: _Plan, schema: Any, depth: int) -> None:
        plan.discriminators = _alias_names(schema["discriminator"])
        if any(type(tag) is not str for tag in schema["choices"]):
            raise _unsupported("discriminated unions require string literal tags")
        plan.choices = {
            tag: self.compile(choice, depth + 1)
            for tag, choice in schema["choices"].items()
        }
        if any(not self._is_model(choice) for choice in schema["choices"].values()):
            raise _unsupported("tagged branches must be StrictModel envelopes")


@dataclass
class _Projection:
    title: str
    remaining: int = _MAX_PROJECTION_CELLS
    ancestors: set[int] = field(default_factory=set)

    def reserve(self, count: int) -> None:
        if count > self.remaining:
            raise CanonicalizationError(
                f"owned containers exceed {_MAX_PROJECTION_CELLS} projected cells"
            )
        self.remaining -= count

    def project(
        self, value: Any, plan: _Plan, loc: tuple[str | int, ...], depth: int
    ) -> Any:
        from jacobian._models import StrictModel

        # Transparent wrappers do not own storage. Resolve them before input
        # type checks, which can invoke a scalar's user-defined __class__.
        while plan.kind in {"default", "nullable"}:
            plan = plan.children[0]
        if plan.kind == "scalar" or isinstance(value, StrictModel):
            return value
        if depth > _MAX_PROJECTION_DEPTH:
            raise CanonicalizationError(
                f"owned container nesting exceeds {_MAX_PROJECTION_DEPTH} levels"
            )
        if plan.kind == "function-after":
            return self.project(value, plan.children[0], loc, depth)
        if plan.kind == "tagged-union":
            return self.tagged(value, plan, loc, depth)
        expected = (dict,) if plan.kind == "model" else (list, tuple)
        if type(value) not in expected:
            if isinstance(value, (dict, list, tuple)):
                raise CanonicalizationError("owned containers require exact builtins")
            return value
        if id(value) in self.ancestors:
            raise CanonicalizationError("cyclic owned containers are not allowed")
        self.ancestors.add(id(value))
        try:
            if plan.kind == "model":
                return self.model(value, plan, loc, depth)
            return self.sequence(value, plan, loc, depth)
        finally:
            self.ancestors.remove(id(value))

    def extra(self, loc: tuple[str | int, ...]) -> None:
        # Do not even fetch an unknown value: rendering errors must not call its
        # repr or traverse its storage. Early refusal intentionally omits other
        # validation errors and the unknown input from diagnostics.
        raise ValidationError.from_exception_data(
            self.title, [{"type": "extra_forbidden", "loc": loc, "input": None}]
        )

    def model(
        self, value: dict[str, Any], plan: _Plan, loc: tuple[str | int, ...], depth: int
    ) -> Any:
        allowed = {name for item in plan.fields for name in item.aliases}
        for key in value:
            if type(key) is not str:
                raise CanonicalizationError("model envelopes require exact string keys")
            if key not in allowed:
                self.extra((*loc, key))
        selected = [
            (item, next((key for key in item.aliases if key in value), None))
            for item in plan.fields
        ]
        used = {key for _, key in selected if key is not None}
        for key in value:
            if key not in used:
                self.extra((*loc, key))
        self.reserve(len(used))
        return {
            key: self.project(
                value[key],
                item.plan,
                (*loc, key if item.loc_by_alias else item.name),
                depth + 1,
            )
            for item, key in selected
            if key is not None
        }

    def sequence(
        self, value: Any, plan: _Plan, loc: tuple[str | int, ...], depth: int
    ) -> Any:
        length = len(value)
        if length < plan.minimum or length > plan.maximum:
            raise CanonicalizationError(
                f"owned tuple length must be between {plan.minimum} and {plan.maximum}"
            )
        self.reserve(length)
        return tuple(
            self.project(
                item,
                plan.children[0 if plan.variadic else index],
                (*loc, index),
                depth + 1,
            )
            for index, item in enumerate(value)
        )

    def tagged(
        self, value: Any, plan: _Plan, loc: tuple[str | int, ...], depth: int
    ) -> Any:
        if type(value) is not dict:
            if isinstance(value, dict):
                raise CanonicalizationError("owned containers require exact builtins")
            return value
        # Check keys before lookups so a user-defined key cannot receive equality
        # callbacks from dict lookup. This scan is bounded independently of tags.
        if len(value) > self.remaining:
            self.reserve(len(value))
        if any(type(key) is not str for key in value):
            raise CanonicalizationError("model envelopes require exact string keys")
        for key in plan.discriminators:
            if key in value:
                tag = value[key]
                if type(tag) is not str or tag not in plan.choices:
                    return value  # Pydantic owns missing/invalid discriminator errors.
                return self.project(value, plan.choices[tag], (*loc, tag), depth)
        return value


def project_owned_containers(value: Any, model: type[StrictModel]) -> Any:
    """Project declared containers without visiting unknown or scalar values.

    Supported schemas: closed ordinary StrictModels, scalar core types, bounded
    homogeneous and fixed tuples, defaults/nullable fields, scalar unions, and
    string-tagged model unions. Resolved core schemas retain Annotated bound
    ordering and discriminator metadata. String aliases and AliasChoices use
    the model's configured validation policy; first-present wins even for None,
    and unused aliases remain forbidden extras. AliasPath, generic/recursive
    aliases, untagged structured unions, arbitrary container transformations,
    and per-call by_alias/by_name overrides are unsupported. Unsupported schemas
    raise TypeError before any input is read. No callbacks or defaults are run.

    Only exact builtin dict/list/tuple storage is traversed. Native StrictModel
    instances and scalar leaves stay opaque. Bounds are checked before copying;
    path cycles, depth and total output occurrences (including repeated DAG
    edges) are bounded. Unknown keys raise early extra_forbidden with nested
    locations, without retrieving their values; error aggregation and unknown
    input diagnostics intentionally differ from ordinary Pydantic validation.

    Model preflights must preserve declared ownership. Their later transformations
    and traversal of invalid scalar leaves remain the owner's responsibility.
    This is a projection step, not validation or a general Pydantic interpreter.
    """
    from jacobian._models import StrictModel

    if not isinstance(model, type) or not issubclass(model, StrictModel):
        raise _unsupported("model must be a StrictModel class")
    plan = _Compiler().compile(model.__pydantic_core_schema__)
    projection = _Projection(model.__name__)
    projection.reserve(1)
    return projection.project(value, plan, (), 0)
