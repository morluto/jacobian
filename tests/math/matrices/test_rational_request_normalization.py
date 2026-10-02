"""Native/canonical boundaries remain strict around opt-in JSON request ratios."""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Annotated, Any

import pytest
from pydantic import TypeAdapter, ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.matrices._operation_models import (
    MatrixDeterminantRequest,
    MatrixRankRequest,
    SquareRationalMatrixRequest,
)
from jacobian.math.matrices._rational_input import RationalInputEncoding
from jacobian.math.matrices.operations import determinant_result
from jacobian.math.matrices.rational_linear._models import (
    LinearRationalSolutionFindRequest,
    LinearRationalSystem,
)
from jacobian.math.matrices.values import (
    RationalMatrix,
    SparseRationalMatrix,
    SparseRationalMatrixEntry,
)


@pytest.mark.parametrize("num,den", [(2, 4), (2, -4), (-2, -4), (0, 5), (0, -5)])
def test_canonical_scalars_and_native_requests_stay_strict(num: int, den: int) -> None:
    native = {"num": num, "den": den}
    wire = {"num": str(num), "den": str(den)}
    with pytest.raises(ValidationError):
        CanonicalRational.model_validate(native, strict=True)
    with pytest.raises(ValidationError):
        CanonicalRational.model_validate_json(json.dumps(wire), strict=True)
    for model in (MatrixDeterminantRequest, SquareRationalMatrixRequest):
        with pytest.raises(ValidationError):
            model.model_validate({"matrix": {"entries": ((native,),)}}, strict=True)
        parsed = model.model_validate_json(
            json.dumps({"matrix": {"entries": [[wire]]}}), strict=True
        )
        assert type(parsed.matrix) is RationalMatrix
        assert parsed.matrix.entries[0][0].as_fraction() == Fraction(num, den)
        assert type(parsed.matrix.entries[0][0]) is CanonicalRational
    with pytest.raises(ValidationError):
        RationalMatrix.model_validate_json(
            json.dumps({"entries": [[wire]]}), strict=True
        )


def test_native_request_and_result_round_trips_keep_native_exact_values() -> None:
    rational = CanonicalRational(num=1, den=2)
    matrix = RationalMatrix(entries=((rational,),))
    native = MatrixDeterminantRequest(matrix=matrix)
    assert native.matrix is matrix
    assert native.model_dump()["matrix"]["entries"] == (({"num": 1, "den": 2},),)
    assert (
        MatrixDeterminantRequest.model_validate_json(
            native.model_dump_json(), strict=True
        )
        == native
    )
    assert determinant_result(native.matrix).determinant == rational
    with pytest.raises(ValidationError):
        MatrixDeterminantRequest.model_validate(
            {"matrix": {"entries": (({"num": "1", "den": "2"},),)}}, strict=True
        )


def test_nested_sparse_system_types_and_variable_axis_stay_canonical() -> None:
    payload = {
        "system": {
            "variables": ["z", "a"],
            "coefficients": {
                "row_count": 1,
                "column_count": 2,
                "entries": [{"row": 0, "column": 1, "value": {"num": "2", "den": "4"}}],
            },
            "rhs": [{"num": "2", "den": "-4"}],
        }
    }
    request = LinearRationalSolutionFindRequest.model_validate_json(
        json.dumps(payload), strict=True
    )
    assert type(request.system) is LinearRationalSystem
    assert request.system.variables == ("z", "a")
    assert type(request.system.coefficients) is SparseRationalMatrix
    assert type(request.system.coefficients.entries[0]) is SparseRationalMatrixEntry
    assert type(request.system.rhs[0]) is CanonicalRational
    assert request.system.coefficients.entries[0].value == CanonicalRational(
        num=1, den=2
    )
    assert request.system.rhs == (CanonicalRational(num=-1, den=2),)
    with pytest.raises(ValidationError):
        LinearRationalSystem.model_validate_json(
            json.dumps(payload["system"]), strict=True
        )
    with pytest.raises(ValidationError):
        SparseRationalMatrix.model_validate_json(
            json.dumps(payload["system"]["coefficients"]), strict=True
        )
    encoded = request.model_dump_json()
    assert (
        LinearRationalSolutionFindRequest.model_validate_json(encoded, strict=True)
        == request
    )
    assert (
        LinearRationalSystem.model_validate_json(
            json.dumps(request.model_dump(mode="json")["system"]), strict=True
        )
        == request.system
    )


def test_scalar_helper_obeys_caller_raw_bound_before_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter: TypeAdapter[CanonicalRational] = TypeAdapter(
        Annotated[CanonicalRational, RationalInputEncoding(max_digits=3)]
    )
    assert adapter.validate_json(
        '{"num":"-998","den":"-996"}', strict=True
    ) == CanonicalRational(num=499, den=498)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Fraction must not run on an unadmitted raw ratio")

    monkeypatch.setattr("jacobian.math.matrices._rational_input.Fraction", forbidden)
    for payload in [
        '{"num":"2000","den":"4000"}',
        '{"num":"1","den":"0"}',
        '{"num":"2","den":"4","extra":true}',
        '{"num":2,"den":"4"}',
    ]:
        with pytest.raises(ValidationError):
            adapter.validate_json(payload, strict=True)


@pytest.mark.parametrize(
    "rows, columns, expected_type",
    [
        (0, 8192, RationalMatrix),
        (1, 8192, SparseRationalMatrix),
        (8192, 0, SparseRationalMatrix),
    ],
)
def test_empty_union_keeps_existing_branch_choice_and_dimensions(
    rows: int, columns: int, expected_type: type[RationalMatrix | SparseRationalMatrix]
) -> None:
    source = SparseRationalMatrix(row_count=rows, column_count=columns)
    request = MatrixRankRequest(matrix=source)
    assert request.matrix is source
    parsed = MatrixRankRequest.model_validate_json(
        request.model_dump_json(), strict=True
    )
    # Empty zero-row JSON has always matched the dense branch first. Do not
    # change union representation semantics while normalizing scalar entries.
    assert type(parsed.matrix) is expected_type
    assert (parsed.matrix.row_count, parsed.matrix.column_count) == (rows, columns)
    assert parsed.matrix.entries == ()
