"""Adversarial contracts for the opt-in owned-container projection grammar."""

from __future__ import annotations

from typing import Annotated, Any, Generic, Literal, TypeVar

import pytest
from annotated_types import MaxLen
from pydantic import (
    AliasChoices,
    AliasPath,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    PlainValidator,
    Tag,
    ValidationError,
    WrapValidator,
    create_model,
    model_validator,
)

from jacobian._models import StrictModel, project_owned_containers
from jacobian.canonical import CanonicalizationError

type RecursiveAlias = tuple["RecursiveAlias", ...]


class HostileList(list[Any]):
    def __iter__(self) -> Any:
        raise AssertionError("iterated hostile storage")

    def __len__(self) -> int:
        raise AssertionError("measured hostile storage")

    def __getitem__(self, index: Any) -> Any:
        raise AssertionError("indexed hostile storage")

    def __repr__(self) -> str:
        raise AssertionError("rendered hostile storage")


class Leaf(StrictModel):
    own: tuple[int, ...] = Field(max_length=3)


class Root(StrictModel):
    children: tuple[Leaf, ...] = Field(max_length=3)
    own: tuple[int, ...] = Field(max_length=3)
    scalar: int = 0
    optional: tuple[int, ...] | None = Field(None, max_length=3)


def test_nested_ownership_input_immutability_and_native_identity() -> None:
    native = Leaf(own=(8,))
    raw = {"children": [{"own": [1, 2]}, native], "own": [3]}
    projected = project_owned_containers(raw, Root)
    assert projected == {"children": ({"own": (1, 2)}, native), "own": (3,)}
    assert raw == {"children": [{"own": [1, 2]}, native], "own": [3]}
    assert projected["children"][1] is native
    assert project_owned_containers(native, Leaf) is native
    assert Root.model_validate(projected, strict=True).children == (
        Leaf(own=(1, 2)),
        native,
    )


@pytest.mark.parametrize("nested", [False, True])
def test_unknown_values_are_not_even_rendered(nested: bool) -> None:
    raw: dict[str, Any] = {"children": [{"own": [1]}], "own": [2]}
    target = raw["children"][0] if nested else raw
    target["alien"] = HostileList()
    with pytest.raises(ValidationError) as error:
        project_owned_containers(raw, Root)
    assert error.value.errors(include_url=False)[0] == {
        "type": "extra_forbidden",
        "loc": ("children", 0, "alien") if nested else ("alien",),
        "msg": "Extra inputs are not permitted",
        "input": None,
    }
    assert "alien" in str(error.value)


def test_nested_model_uses_its_own_fields() -> None:
    class Child(StrictModel):
        child: tuple[int, ...] = Field(max_length=2)

    class Parent(StrictModel):
        own: tuple[int, ...] = Field(max_length=2)
        child: Child

    hostile = HostileList()
    with pytest.raises(ValidationError) as error:
        project_owned_containers({"own": [], "child": {"own": hostile}}, Parent)
    assert error.value.errors()[0]["loc"] == ("child", "own")
    assert project_owned_containers({"own": [], "child": {"child": [1]}}, Parent) == {
        "own": (),
        "child": {"child": (1,)},
    }


def test_scalar_storage_is_opaque_and_left_for_scalar_validation() -> None:
    hostile = HostileList()
    raw = {"children": [], "own": [], "scalar": hostile}
    assert project_owned_containers(raw, Root)["scalar"] is hostile
    with pytest.raises(ValidationError) as error:
        Root.model_validate(project_owned_containers(raw, Root), strict=True)
    assert error.value.errors()[0]["type"] == "int_type"


@pytest.mark.parametrize("annotation", [int, int | str])
@pytest.mark.parametrize("nullable", [False, True])
@pytest.mark.parametrize("defaulted", [False, True])
def test_scalar_metadata_is_opaque_through_transparent_wrappers(
    annotation: Any, nullable: bool, defaulted: bool
) -> None:
    class Scalar:
        @property  # type: ignore[misc]
        def __class__(self) -> Any:
            raise AssertionError("scalar metadata executed")

    model: Any = create_model(
        "ScalarModel",
        __base__=StrictModel,
        value=(
            annotation | None if nullable else annotation,
            Field(7 if defaulted else ...),
        ),
    )
    scalar = Scalar()
    projected = project_owned_containers({"value": scalar}, model)
    assert projected["value"] is scalar
    with pytest.raises(ValidationError) as error:
        model.model_validate(projected, strict=True)
    assert [item["type"] for item in error.value.errors()] == (
        ["int_type"] if annotation is int else ["int_type", "string_type"]
    )
    assert (
        model.model_validate(
            project_owned_containers({"value": 3}, model), strict=True
        ).value
        == 3
    )

    absent = project_owned_containers({}, model)
    assert absent == {}
    if defaulted:
        default = model.model_validate(absent, strict=True)
        assert default.value == 7
        assert default.model_fields_set == set()
    else:
        with pytest.raises(ValidationError) as error:
            model.model_validate(absent, strict=True)
        assert error.value.errors()[0]["type"] == "missing"

    present_null = project_owned_containers({"value": None}, model)
    assert present_null == {"value": None}
    if nullable:
        null = model.model_validate(present_null, strict=True)
        assert null.value is None
        assert null.model_fields_set == {"value"}
    else:
        with pytest.raises(ValidationError) as error:
            model.model_validate(present_null, strict=True)
        assert error.value.errors()[0]["type"] == "int_type"


def test_scalar_default_factory_runs_only_during_validation() -> None:
    calls: list[str] = []

    def default() -> int:
        calls.append("default")
        return 7

    class Defaults(StrictModel):
        value: int | None = Field(default_factory=default)

    absent = project_owned_containers({}, Defaults)
    present = project_owned_containers({"value": None}, Defaults)
    assert absent == {}
    assert calls == []
    assert Defaults.model_validate(present, strict=True).value is None
    assert calls == []
    assert Defaults.model_validate(absent, strict=True).value == 7
    assert calls == ["default"]


def test_wrapped_model_instances_and_after_validators_stay_opaque() -> None:
    calls: list[str] = []

    class Child(StrictModel):
        own: tuple[int, ...] = Field(max_length=3)

        @model_validator(mode="after")
        def record(self) -> Child:
            calls.append("after")
            return self

    class Parent(StrictModel):
        child: Child | None = None
        pair: tuple[Child] | None = None

    native = Child(own=(1,))
    calls.clear()
    assert project_owned_containers(native, Child) is native
    projected = project_owned_containers({"child": native, "pair": [native]}, Parent)
    assert projected["child"] is native
    assert projected["pair"][0] is native
    assert calls == []
    assert project_owned_containers({"pair": native}, Parent)["pair"] is native
    assert calls == []
    projected_raw = project_owned_containers(
        {"child": {"own": [2]}, "pair": [{"own": [3]}]}, Parent
    )
    assert projected_raw == {"child": {"own": (2,)}, "pair": ({"own": (3,)},)}
    assert calls == []
    validated = Parent.model_validate(projected_raw, strict=True)
    assert validated.child is not None and validated.child.own == (2,)
    assert validated.pair is not None and validated.pair[0].own == (3,)
    assert calls == ["after", "after"]


def test_variadic_bounds_precede_element_projection() -> None:
    assert project_owned_containers({"own": [1, 2, 3]}, Leaf) == {"own": (1, 2, 3)}

    class Container(StrictModel):
        children: tuple[Leaf, ...] = Field(max_length=2)

    with pytest.raises(CanonicalizationError, match="tuple length"):
        project_owned_containers({"children": [HostileList()] * 3}, Container)


def test_fixed_arity_is_checked_before_elements() -> None:
    class Pair(StrictModel):
        pair: tuple[int, Leaf]

    assert project_owned_containers({"pair": [3, {"own": [1]}]}, Pair) == {
        "pair": (3, {"own": (1,)}),
    }
    for raw in ([HostileList()], [HostileList()] * 3):
        with pytest.raises(CanonicalizationError, match="tuple length"):
            project_owned_containers({"pair": raw}, Pair)


def test_annotated_bounds_use_resolved_order_not_minimum() -> None:
    class LastBound(StrictModel):
        own: Annotated[tuple[int, ...], MaxLen(1), MaxLen(3)]

    projected = project_owned_containers({"own": [1, 2, 3]}, LastBound)
    assert LastBound.model_validate(projected, strict=True).own == (1, 2, 3)
    with pytest.raises(CanonicalizationError, match="tuple length"):
        project_owned_containers({"own": [1, 2, 3, 4]}, LastBound)


def test_absent_null_defaults_and_fields_set_remain_distinct() -> None:
    absent = Root.model_validate(
        project_owned_containers({"children": [], "own": []}, Root), strict=True
    )
    present = Root.model_validate(
        project_owned_containers({"children": [], "own": [], "optional": None}, Root),
        strict=True,
    )
    assert absent.optional is present.optional is None
    assert absent.model_fields_set == {"children", "own"}
    assert present.model_fields_set == {"children", "own", "optional"}


def test_default_factory_is_not_evaluated() -> None:
    def forbidden() -> tuple[int, ...]:
        raise AssertionError("evaluated a default")

    class Defaults(StrictModel):
        own: tuple[int, ...] = Field(default_factory=forbidden, max_length=3)

    assert project_owned_containers({}, Defaults) == {}


def test_alias_choices_first_present_none_and_unused_alias_stays_extra() -> None:
    class Aliased(StrictModel):
        own: tuple[int, ...] | None = Field(
            max_length=3, validation_alias=AliasChoices("first", "second")
        )

    assert project_owned_containers({"first": None}, Aliased) == {"first": None}
    assert project_owned_containers({"second": [1]}, Aliased) == {"second": (1,)}
    with pytest.raises(ValidationError) as error:
        project_owned_containers({"first": None, "second": HostileList()}, Aliased)
    assert error.value.errors()[0]["loc"] == ("second",)


@pytest.mark.parametrize(
    "by_alias,by_name", [(True, False), (True, True), (False, True)]
)
def test_alias_policy_uses_class_configuration(by_alias: bool, by_name: bool) -> None:
    class Aliased(StrictModel):
        model_config = ConfigDict(validate_by_alias=by_alias, validate_by_name=by_name)
        own: tuple[int, ...] = Field(alias="external", max_length=3)

    for key, accepted in (("own", by_name), ("external", by_alias)):
        if accepted:
            projected = project_owned_containers({key: [1]}, Aliased)
            assert Aliased.model_validate(projected, strict=True).own == (1,)
        else:
            with pytest.raises(ValidationError) as error:
                project_owned_containers({key: HostileList()}, Aliased)
            assert error.value.errors()[0]["type"] == "extra_forbidden"


def test_alias_generator_and_nested_error_name_configuration() -> None:
    class Aliased(StrictModel):
        model_config = ConfigDict(alias_generator=str.upper, loc_by_alias=False)
        child: Leaf

    with pytest.raises(ValidationError) as error:
        project_owned_containers({"CHILD": {"alien": HostileList()}}, Aliased)
    assert error.value.errors()[0]["loc"] == ("child", "alien")


class A(StrictModel):
    kind: Literal["a"] = Field(alias="tag")
    xs: tuple[int, ...] = Field(max_length=2)


class B(StrictModel):
    kind: Literal["b"] = Field(alias="tag")
    ys: tuple[int, ...] = Field(max_length=2)


class Tagged(StrictModel):
    branch: Annotated[A | B, Field(discriminator="kind")]


def test_tagged_union_uses_only_selected_branch_and_retains_metadata() -> None:
    raw = {"branch": {"tag": "b", "ys": [1, 2]}}
    projected = project_owned_containers(raw, Tagged)
    assert Tagged.model_validate(projected, strict=True).branch == B(tag="b", ys=(1, 2))
    with pytest.raises(ValidationError) as error:
        project_owned_containers(
            {"branch": {"tag": "b", "ys": [], "xs": HostileList()}}, Tagged
        )
    assert error.value.errors()[0]["loc"] == ("branch", "b", "xs")


def test_conflicting_discriminator_aliases_cannot_project_wrong_branch() -> None:
    with pytest.raises(ValidationError) as error:
        project_owned_containers(
            {"branch": {"kind": "a", "tag": "b", "ys": HostileList()}}, Tagged
        )
    assert error.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize("tag", [None, "unknown"])
def test_invalid_tag_does_not_visit_branch_values(tag: Any) -> None:
    branch = {"tag": tag, "xs": HostileList()}
    assert project_owned_containers({"branch": branch}, Tagged)["branch"] is branch


class Recursive(StrictModel):
    children: tuple[Recursive, ...] = Field(max_length=2)


def test_cycles_fail_before_recursion_exhaustion() -> None:
    raw: dict[str, Any] = {"children": []}
    raw["children"].append(raw)
    with pytest.raises(CanonicalizationError, match="cyclic"):
        project_owned_containers(raw, Recursive)


def test_deep_valid_structure_has_bounded_failure_and_shallow_success() -> None:
    raw: dict[str, Any] = {"children": []}
    for _ in range(8):
        raw = {"children": [raw]}
    assert project_owned_containers(raw, Recursive)["children"]
    for _ in range(70):
        raw = {"children": [raw]}
    with pytest.raises(CanonicalizationError, match="nesting"):
        project_owned_containers(raw, Recursive)


def test_transparent_wrappers_preserve_structural_depth_boundary() -> None:
    class Node(StrictModel):
        value: int = 0
        child: Node | None = None

    raw: dict[str, Any] = {"value": 1}
    for _ in range(64):
        raw = {"child": raw}
    projected = project_owned_containers(raw, Node)
    for _ in range(64):
        projected = projected["child"]
    assert projected == {"value": 1}

    with pytest.raises(CanonicalizationError, match="nesting"):
        project_owned_containers({"child": raw}, Node)


def test_repeated_dag_edges_are_charged_per_output_occurrence() -> None:
    raw: dict[str, Any] = {"children": []}
    for _ in range(8):
        raw = {"children": [raw, raw]}
    assert len(project_owned_containers(raw, Recursive)["children"]) == 2
    for _ in range(10):
        raw = {"children": [raw, raw]}
    with pytest.raises(CanonicalizationError, match="projected cells"):
        project_owned_containers(raw, Recursive)


def test_total_budget_is_checked_before_materializing_outer_array() -> None:
    class Large(StrictModel):
        own: tuple[Leaf, ...] = Field(max_length=100_001)

    with pytest.raises(CanonicalizationError, match="projected cells"):
        project_owned_containers({"own": [HostileList()] * 100_001}, Large)


def test_nonbuiltin_storage_is_never_iterated() -> None:
    with pytest.raises(CanonicalizationError, match="exact builtins"):
        project_owned_containers({"own": HostileList()}, Leaf)


def test_rebuilding_schema_cannot_leave_stale_ownership_plan() -> None:
    class Rebuilt(StrictModel):
        own: tuple[int, ...] = Field(max_length=3)

    assert project_owned_containers({"own": [1, 2]}, Rebuilt)["own"] == (1, 2)
    Rebuilt.model_fields["own"].metadata = [MaxLen(1)]
    Rebuilt.model_rebuild(force=True)
    with pytest.raises(CanonicalizationError, match="tuple length"):
        project_owned_containers({"own": [1, 2]}, Rebuilt)


def test_model_preflight_compilation_does_not_invoke_itself() -> None:
    calls: list[str] = []

    class Pilot(StrictModel):
        own: tuple[int, ...] = Field(max_length=3)

        @model_validator(mode="before")
        @classmethod
        def preflight(cls, value: Any) -> Any:
            calls.append("called")
            return project_owned_containers(value, cls)

    assert Pilot.model_validate_json('{"own": [1, 2]}', strict=True).own == (1, 2)
    assert calls == ["called"]


@pytest.mark.parametrize(
    "annotation,field_info,match",
    [
        (tuple[int, ...], Field(), "explicit max_length"),
        (Leaf | tuple[int, int], Field(), "untagged structured"),
        (
            tuple[int, ...],
            Field(max_length=3, validation_alias=AliasPath("carrier", "own")),
            "AliasPath",
        ),
        (
            Annotated[tuple[int, ...], BeforeValidator(lambda value: value)],
            Field(max_length=3),
            "transformations",
        ),
        (list[int], Field(max_length=3), "transformations"),
    ],
)
def test_unsupported_schemas_fail_before_input_access(
    annotation: Any, field_info: Any, match: str
) -> None:
    model = create_model(
        "Unsupported", __base__=StrictModel, own=(annotation, field_info)
    )
    with pytest.raises(TypeError, match=match):
        project_owned_containers(HostileList(), model)


def test_unsupported_unselected_branch_is_rejected_without_input_access() -> None:
    class Bad(StrictModel):
        kind: Literal["bad"]
        own: list[int]

    class Good(StrictModel):
        kind: Literal["good"]
        own: tuple[int, int]

    class Union(StrictModel):
        branch: Bad | Good = Field(discriminator="kind")

    with pytest.raises(TypeError, match="transformations"):
        project_owned_containers({"branch": {"kind": "good", "own": [1, 2]}}, Union)


def test_generic_models_and_recursive_aliases_are_explicitly_unsupported() -> None:
    parameter = TypeVar("parameter")

    class GenericModel(StrictModel, Generic[parameter]):
        own: tuple[parameter, ...] = Field(max_length=3)

    with pytest.raises(TypeError, match="generic"):
        project_owned_containers({}, GenericModel[int])

    model = create_model(
        "RecursiveAliasModel",
        __base__=StrictModel,
        own=(RecursiveAlias, Field(max_length=3)),
    )
    with pytest.raises(TypeError, match="recursive aliases"):
        project_owned_containers({}, model)


def test_nested_legacy_validator_does_not_see_unknown_storage() -> None:
    calls: list[str] = []

    class Legacy(StrictModel):
        own: tuple[int, ...] = Field(max_length=3)

        @model_validator(mode="before")
        @classmethod
        def preflight(cls, value: Any) -> Any:
            calls.append("legacy")
            return value

    class Pilot(StrictModel):
        child: Legacy

        @model_validator(mode="before")
        @classmethod
        def preflight(cls, value: Any) -> Any:
            return project_owned_containers(value, cls)

    with pytest.raises(ValidationError) as error:
        Pilot.model_validate(
            {"child": {"own": [], "alien": HostileList()}}, strict=True
        )
    assert error.value.errors()[0]["loc"] == ("child", "alien")
    assert calls == []


def test_iterators_and_nonstring_keys_receive_no_callbacks() -> None:
    class NotAContainer:
        def __iter__(self) -> Any:
            raise AssertionError("arbitrary iterable was traversed")

    raw = NotAContainer()
    assert project_owned_containers({"own": raw}, Leaf)["own"] is raw

    class HostileKey:
        def __hash__(self) -> int:
            return hash("own")

        def __eq__(self, other: object) -> bool:
            raise AssertionError("key was compared")

    with pytest.raises(CanonicalizationError, match="exact string keys"):
        project_owned_containers({HostileKey(): []}, Leaf)


def test_shared_field_aliases_are_explicitly_unsupported() -> None:
    class Overlap(StrictModel):
        first: tuple[int, ...] = Field(alias="both", max_length=2)
        second: tuple[int, ...] = Field(alias="both", max_length=2)

    with pytest.raises(TypeError, match="shared by multiple fields"):
        project_owned_containers(HostileList(), Overlap)


@pytest.mark.parametrize(
    "wrapper",
    [
        WrapValidator(lambda value, handler: handler(value)),
        PlainValidator(lambda value: value),
    ],
)
def test_container_wrap_and_plain_transformations_are_not_guessed(wrapper: Any) -> None:
    model = create_model(
        "Transformed",
        __base__=StrictModel,
        own=(Annotated[tuple[int, ...], wrapper], Field(max_length=3)),
    )
    with pytest.raises(TypeError, match="transformations"):
        project_owned_containers(HostileList(), model)


def test_callable_discriminator_is_refused_without_invocation() -> None:
    def choose(value: Any) -> str:
        raise AssertionError("discriminator was invoked")

    class CallableTagged(StrictModel):
        branch: Annotated[
            Annotated[A, Tag("a")] | Annotated[B, Tag("b")], Discriminator(choose)
        ]

    with pytest.raises(TypeError, match="callables"):
        project_owned_containers(HostileList(), CallableTagged)


def test_total_cell_boundary_accepts_cap_and_refuses_cap_plus_one() -> None:
    class Large(StrictModel):
        own: tuple[int, ...] = Field(max_length=100_000)

    # Root + one field slot + 99,998 scalar slots = 100,000 cells.
    assert len(project_owned_containers({"own": [1] * 99_998}, Large)["own"]) == 99_998
    with pytest.raises(CanonicalizationError, match="projected cells"):
        project_owned_containers({"own": [1] * 99_999}, Large)
