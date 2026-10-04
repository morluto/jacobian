"""Rank/nullspace union diagnostics preserve parsing and caller-owned paths."""

from __future__ import annotations

import json
from typing import Annotated, Any

import pytest
from pydantic import ConfigDict, TypeAdapter, ValidationError, WrapValidator
from pydantic_core import ErrorDetails, PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.math.matrices._operation_models import (
    MAX_INPUT_SCALAR_DIGITS,
    MatrixRankRequest,
    _rank_matrix_branch,
    _rank_matrix_message_context,
    _rank_matrix_validation,
)
from jacobian.math.matrices._rational_input import (
    RationalMatrixInput,
    RationalMatrixInputEnvelope,
    SparseRationalMatrixInput,
)
from jacobian.math.matrices.values import (
    MAX_RATIONAL_MATRIX_ORDER,
    MAX_SPARSE_RATIONAL_MATRIX_NONZEROS,
    RationalMatrix,
    SparseRationalMatrix,
    SparseRationalMatrixEntry,
)


# Original union order and annotations provide the parsing/schema control.
class _OriginalRankRequest(MatrixRankRequest):
    """One bounded rectangular matrix whose exact rank is requested."""

    model_config = ConfigDict(title="MatrixRankRequest")
    matrix: (
        Annotated[
            RationalMatrixInput, RationalMatrixInputEnvelope(MAX_RATIONAL_MATRIX_ORDER)
        ]
        | SparseRationalMatrixInput
    )


def _errors(matrix: Any, *, native: bool = False) -> list[ErrorDetails]:
    with pytest.raises(ValidationError) as rejected:
        if native:
            MatrixRankRequest.model_validate({"matrix": matrix}, strict=True)
        else:
            MatrixRankRequest.model_validate_json(
                json.dumps({"matrix": matrix}), strict=True
            )
    return rejected.value.errors(include_url=False)


@pytest.mark.parametrize("native", [False, True])
def test_dense_shape_error_has_only_the_caller_matrix_path(native: bool) -> None:
    scalar = {"num": 1, "den": 1} if native else {"num": "1", "den": "1"}
    errors = _errors(
        {"row_count": 2, "column_count": 2, "entries": ((scalar,),)}, native=native
    )
    assert len(errors) == 1
    assert errors[0]["loc"] == ("matrix",)
    assert errors[0]["type"] == "matrix.shape_mismatch"
    assert errors[0]["msg"] == "matrix entries must match the declared shape"


@pytest.mark.parametrize("native", [False, True])
def test_dense_cell_error_retains_location_code_context_and_input(native: bool) -> None:
    scalar = {"num": 1, "den": "bad"} if native else {"num": "1", "den": "bad"}
    payload = {"matrix": {"entries": ((scalar,),)}}

    def validate(model: type[MatrixRankRequest]) -> MatrixRankRequest:
        if native:
            return model.model_validate(payload, strict=True)
        return model.model_validate_json(json.dumps(payload), strict=True)

    with pytest.raises(ValidationError) as original:
        validate(_OriginalRankRequest)
    with pytest.raises(ValidationError) as projected:
        validate(MatrixRankRequest)
    expected = original.value.errors()[0]
    expected["loc"] = ("matrix", "entries", 0, 0, "den")
    assert projected.value.errors() == [expected]
    assert expected["input"] == "bad"


@pytest.mark.parametrize("native", [False, True])
def test_sparse_missing_coordinate_and_bound_errors_keep_native_metadata(
    native: bool,
) -> None:
    scalar = {"num": 1, "den": 1} if native else {"num": "1", "den": "1"}
    errors = _errors(
        {
            "row_count": 1,
            "column_count": 1,
            "entries": ({"row": -1, "value": scalar},),
        },
        native=native,
    )
    assert [(error["loc"], error["type"]) for error in errors] == [
        (("matrix", "entries", 0, "row"), "greater_than_equal"),
        (("matrix", "entries", 0, "column"), "missing"),
    ]
    assert errors[0]["ctx"] == {"ge": 0}
    assert errors[0]["input"] == -1
    assert errors[1]["input"] == {"row": -1, "value": scalar}


def test_sparse_coordinate_relation_error_drops_only_dense_alternative() -> None:
    errors = _errors(
        {
            "row_count": 1,
            "column_count": 1,
            "entries": [{"row": 1, "column": 0, "value": {"num": "1", "den": "1"}}],
        }
    )
    assert len(errors) == 1
    assert errors[0]["loc"] == ("matrix",)
    assert errors[0]["msg"] == "sparse matrix coordinates exceed declared axes"


@pytest.mark.parametrize(
    "entries",
    [
        [{"num": "1", "den": "1"}],
        [{"unknown": "x"}],
        [{"value": {"num": "1", "den": "1"}}],
        [{"row": 0, "value": {"num": "1", "den": "1"}}, {"num": "1", "den": "1"}],
        [{}],
    ],
)
@pytest.mark.parametrize("native", [False, True])
def test_uncertain_entry_mappings_retain_missing_dense_row_error(
    entries: list[dict[str, Any]], native: bool
) -> None:
    errors = _errors({"entries": entries}, native=native)
    dense = [error for error in errors if error["msg"].startswith("Dense matrix: ")]
    sparse = [error for error in errors if error["msg"].startswith("Sparse matrix: ")]
    assert [(error["loc"], error["type"]) for error in dense] == [
        (("matrix", "entries", index), "tuple_type") for index in range(len(entries))
    ]
    assert sparse
    assert len(dense) + len(sparse) == len(errors)


def test_coordinate_carrier_nested_in_native_row_retains_both_alternatives() -> None:
    entry = SparseRationalMatrixEntry(
        row=0, column=0, value=CanonicalRational(num=1, den=1)
    )
    errors = _errors({"entries": ((entry,),)}, native=True)
    assert any(error["msg"].startswith("Dense matrix: ") for error in errors)
    assert any(error["msg"].startswith("Sparse matrix: ") for error in errors)


def test_coordinate_mapping_nested_in_json_row_keeps_raw_preflight_error() -> None:
    errors = _errors(
        {"entries": [[{"row": 0, "column": 0, "value": {"num": "1", "den": "1"}}]]}
    )
    assert [(error["loc"], error["type"], error["msg"]) for error in errors] == [
        (
            (),
            "matrix.shape_mismatch",
            "matrix input rational scalar contains unknown fields",
        )
    ]


@pytest.mark.parametrize("entries", [None, [], [7], [[], {}]])
@pytest.mark.parametrize("native", [False, True])
def test_ambiguous_storage_retains_both_alternatives_outside_paths(
    entries: Any, native: bool
) -> None:
    matrix: dict[str, Any] = {"row_count": -1}
    if entries is not None:
        matrix["entries"] = entries
    payload = {"matrix": matrix}
    with pytest.raises(ValidationError) as original:
        if native:
            _OriginalRankRequest.model_validate(payload, strict=True)
        else:
            _OriginalRankRequest.model_validate_json(json.dumps(payload), strict=True)
    actual = _errors(matrix, native=native)
    expected = original.value.errors(include_url=False)
    branches = list(dict.fromkeys(error["loc"][1] for error in expected))
    assert len(branches) == 2
    prefixes = dict(zip(branches, ("Dense matrix: ", "Sparse matrix: "), strict=True))
    for error in expected:
        tag = error["loc"][1]
        error["loc"] = ("matrix", *error["loc"][2:])
        error["msg"] = prefixes[tag] + error["msg"]
    assert actual == expected


@pytest.mark.parametrize(
    "matrix",
    [
        {},
        {"entries": []},
        {"row_count": 0, "column_count": 3, "entries": []},
        {"row_count": 3, "column_count": 0, "entries": [[], [], []]},
        {"row_count": 2, "column_count": 3, "entries": []},
        {"row_count": 8192, "column_count": 8192, "entries": []},
        {"entries": [[{"num": "2", "den": "-4"}]]},
        {
            "row_count": 2,
            "column_count": 3,
            "entries": [{"row": 0, "column": 1, "value": {"num": "2", "den": "4"}}],
        },
    ],
)
def test_successful_union_carriers_and_normalization_are_unchanged(
    matrix: dict[str, Any],
) -> None:
    payload = json.dumps({"matrix": matrix})
    original = _OriginalRankRequest.model_validate_json(payload, strict=True)
    actual = MatrixRankRequest.model_validate_json(payload, strict=True)
    assert type(actual.matrix) is type(original.matrix)
    assert actual.matrix == original.matrix
    assert actual.model_dump_json() == original.model_dump_json()
    assert (
        type(actual.matrix).model_validate_json(
            actual.matrix.model_dump_json(), strict=True
        )
        == actual.matrix
    )


def test_native_carriers_return_the_same_instances_and_remain_strict() -> None:
    scalar = CanonicalRational(num=1, den=2)
    for matrix in (
        RationalMatrix(entries=((scalar,),)),
        SparseRationalMatrix(
            row_count=1,
            column_count=1,
            entries=(SparseRationalMatrixEntry(row=0, column=0, value=scalar),),
        ),
    ):
        assert MatrixRankRequest(matrix=matrix).matrix is matrix
    errors = _errors({"entries": (({"num": 2, "den": 4},),)}, native=True)
    assert len(errors) == 1
    assert errors[0]["type"] == "canonical_rational.noncanonical_representation"
    with pytest.raises(ValidationError):
        RationalMatrix.model_validate_json(
            '{"entries":[[{"num":"2","den":"4"}]]}', strict=True
        )


def test_raw_digit_preflight_still_precedes_ratio_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("oversized raw input reached Fraction")

    monkeypatch.setattr("jacobian.math.matrices._rational_input.Fraction", forbidden)
    oversized = "2" + "0" * MAX_INPUT_SCALAR_DIGITS
    errors = _errors({"entries": [[{"num": oversized, "den": oversized}]]})
    assert len(errors) == 1
    assert errors[0]["type"] == "matrix.budget_exceeded"
    assert errors[0]["loc"] == ()


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_request_schema_is_byte_identical_to_original_union(mode: Any) -> None:
    assert json.dumps(MatrixRankRequest.model_json_schema(mode=mode)) == json.dumps(
        _OriginalRankRequest.model_json_schema(mode=mode)
    )


@pytest.mark.parametrize(
    "key",
    [
        "dense",
        "sparse",
        "function-after[caller_key()]",
        "json-or-python[my_key]",
        "dots.brackets[0]",
    ],
)
def test_only_the_owned_leading_union_tag_is_removed(key: str) -> None:
    # A handler-provided error lets caller keys exercise the projection without
    # the older matrix preflight rejecting unknown fields before field parsing.
    def reject(value: Any, handler: Any) -> Any:
        raise ValidationError.from_exception_data(
            "Rank input",
            [{"type": "extra_forbidden", "loc": ("dense", key, key), "input": value}],
        )

    adapter: TypeAdapter[Any] = TypeAdapter(
        Annotated[Any, WrapValidator(reject), WrapValidator(_rank_matrix_validation)]
    )
    value: dict[str, Any] = {"entries": [[]]}
    with pytest.raises(ValidationError) as rejected:
        adapter.validate_python(value)
    assert rejected.value.errors()[0]["loc"] == (key, key)
    assert rejected.value.errors()[0]["input"] is value


@pytest.mark.parametrize(
    "message,context,projected",
    [
        ("literal {missing}", {"limit": 3}, True),
        ("literal {limit}", {"limit": 3}, False),
    ],
)
def test_custom_error_context_and_rendered_braces_are_preserved(
    message: str, context: dict[str, Any], projected: bool
) -> None:
    # Context substitution inserts a literal placeholder into the final message.
    # Reusing that rendered message as a template must not substitute it again.
    original = PydanticCustomError(
        "matrix.custom", "{text}", {**context, "text": message}
    )

    def reject(value: Any, handler: Any) -> Any:
        raise ValidationError.from_exception_data(
            "Rank input",
            [{"type": original, "loc": ("dense", "entries"), "input": value}],
        )

    adapter: TypeAdapter[Any] = TypeAdapter(
        Annotated[Any, WrapValidator(reject), WrapValidator(_rank_matrix_validation)]
    )
    with pytest.raises(ValidationError) as rejected:
        adapter.validate_python({"entries": [[]]})
    error = rejected.value.errors(include_url=False)[0]
    assert error["loc"] == (("entries",) if projected else ("dense", "entries"))
    assert error["type"] == original.type
    assert error["msg"] == original.message()
    assert error["ctx"] == original.context


def test_branch_evidence_does_not_inspect_hostile_subclasses_or_scalars() -> None:
    class HostileMetaclass(type):
        def __eq__(cls, other: Any) -> bool:
            raise AssertionError("custom class equality")

    class HostileList(list[Any]):
        def __iter__(self) -> Any:
            raise AssertionError("custom iteration")

        def __len__(self) -> int:
            raise AssertionError("custom length")

    class HostileDict(dict[str, Any]):
        def get(self, *args: Any) -> Any:
            raise AssertionError("custom lookup")

    class HostileKey(str):
        def __eq__(self, other: Any) -> bool:
            raise AssertionError("custom key equality")

        __hash__ = str.__hash__

    class HostileScalar(metaclass=HostileMetaclass):
        def __getattribute__(self, name: str) -> Any:
            raise AssertionError("custom class property")

        def __repr__(self) -> str:
            raise AssertionError("custom repr")

    class DenseSubclass(RationalMatrix):
        def __getattribute__(self, name: str) -> Any:
            if name == "entries":
                raise AssertionError("native subclass property")
            return super().__getattribute__(name)

    value: object
    for value in (
        HostileScalar(),
        HostileDict(entries=[]),
        {"entries": HostileList()},
        {"entries": HostileScalar()},
        {"entries": [HostileList()]},
        {"entries": [HostileDict()]},
        {"entries": [HostileScalar()]},
        {"entries": [{HostileKey("row"): 0}]},
        {"entries": [[{HostileKey("row"): 0}]]},
        {"entries": [[HostileDict()]]},
        {"entries": [[HostileScalar()]]},
        {"entries": [{"row": 0, "column": 0, "value": {}, "unknown": 0}]},
        {"entries": [[{"row": 0, "column": 0, "value": {}}]]},
        {"entries": [[{}] * (MAX_RATIONAL_MATRIX_ORDER + 1)]},
        {"entries": [[]] * (MAX_RATIONAL_MATRIX_ORDER + 1)},
        DenseSubclass.model_construct(),
        {"entries": [()] * (MAX_SPARSE_RATIONAL_MATRIX_NONZEROS + 1)},
    ):
        assert _rank_matrix_branch(value) is None
    assert _rank_matrix_branch({"entries": [[{"num": 1, "den": 1}]]}) == "dense"
    assert _rank_matrix_branch({"entries": [{"row": 0}]}) == "sparse"
    assert _rank_matrix_branch({"entries": [{"column": 0}]}) == "sparse"
    assert _rank_matrix_branch(RationalMatrix()) == "dense"
    assert (
        _rank_matrix_branch(SparseRationalMatrix(row_count=0, column_count=0))
        == "sparse"
    )


def test_custom_error_reusing_native_code_preserves_its_message() -> None:
    original = PydanticCustomError("int_type", "matrix-specific integer error")

    def reject(value: Any, handler: Any) -> Any:
        raise ValidationError.from_exception_data(
            "Rank input",
            [{"type": original, "loc": ("dense", "entries"), "input": value}],
        )

    adapter: TypeAdapter[Any] = TypeAdapter(
        Annotated[Any, WrapValidator(reject), WrapValidator(_rank_matrix_validation)]
    )
    value: dict[str, Any] = {"entries": [[]]}
    with pytest.raises(ValidationError) as rejected:
        adapter.validate_python(value)
    assert rejected.value.errors() == [
        {
            "type": "int_type",
            "loc": ("entries",),
            "msg": "matrix-specific integer error",
            "input": value,
        }
    ]


@pytest.mark.parametrize(
    "entries,message_length",
    [([[]], 4097), ([], 4096 - len("Dense matrix: ") + 1)],
)
def test_long_custom_messages_without_context_retain_original_error(
    entries: list[Any], message_length: int
) -> None:
    message = "x" * message_length

    def reject(value: Any, handler: Any) -> Any:
        raise ValidationError.from_exception_data(
            "Rank input",
            [
                {
                    "type": PydanticCustomError("matrix.custom", message),
                    "loc": ("dense", "entries"),
                    "input": value,
                }
            ],
        )

    adapter: TypeAdapter[Any] = TypeAdapter(
        Annotated[Any, WrapValidator(reject), WrapValidator(_rank_matrix_validation)]
    )
    value = {"entries": entries}
    with pytest.raises(ValidationError) as rejected:
        adapter.validate_python(value)
    assert rejected.value.errors() == [
        {
            "type": "matrix.custom",
            "loc": ("dense", "entries"),
            "msg": message,
            "input": value,
        }
    ]


def test_context_reconstruction_is_bounded_without_custom_inspection() -> None:
    class HostileMetaclass(type):
        def __eq__(cls, other: Any) -> bool:
            raise AssertionError("custom class equality")

    class HostileDict(dict[str, Any]):
        def __len__(self) -> int:
            raise AssertionError("custom context length")

    class HostileString(str):
        def __len__(self) -> int:
            raise AssertionError("custom key length")

    class HostileValue(metaclass=HostileMetaclass):
        def __repr__(self) -> str:
            raise AssertionError("custom context repr")

    for context in (
        HostileDict(),
        {HostileString("limit"): 3},
        {"value": HostileValue()},
        {"limit": 1 << 1024},
        {"value": "x" * 1025},
        {"limit": float("inf")},
        {"limit": float("nan")},
        {"x" * 129: 1},
        {str(index): index for index in range(9)},
    ):
        assert not _rank_matrix_message_context("invalid matrix", context)
    assert not _rank_matrix_message_context("x" * 4097, {})
    assert not _rank_matrix_message_context("x" * 4097, None)
    assert _rank_matrix_message_context("invalid matrix", {"ge": 0})
