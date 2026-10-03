"""All bounded matrix rational ingress positions normalize without widening values."""

from __future__ import annotations

import copy
import json
from fractions import Fraction
from typing import Any

import pytest
from pydantic import ValidationError
from tests.dispatch._rational_request_cases import (
    RATIONAL_REQUEST_CASES,
    RationalRequestCase,
    dense,
    ratio,
)

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import (
    OperationRequestValidationError,
    invoke_operation,
    parse_operation_input,
)
from jacobian.math.matrices._operation_models import (
    MatrixDeterminantRequest,
    MatrixRankRequest,
    RationalLinearSolveRequest,
)


@pytest.mark.parametrize("case", RATIONAL_REQUEST_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(
    "num,den", [("2", "4"), ("2", "-4"), ("-2", "-4"), ("0", "5"), ("0", "-5")]
)
def test_every_rational_argument_normalizes_at_dispatch(
    case: RationalRequestCase, num: str, den: str
) -> None:
    payload = case.payload(ratio(num, den))
    unchanged = copy.deepcopy(payload)
    value = Fraction(int(num), int(den))
    canonical_payload = case.payload(
        ratio(str(value.numerator), str(value.denominator))
    )
    catalog = Catalog.open()
    if case.sparse and value == 0:
        with pytest.raises(OperationRequestValidationError) as caught:
            invoke_operation(case.operation_id, payload, catalog)
        assert any(
            "must not store explicit zeros" in error["msg"]
            for error in caught.value.errors()
        )
    else:
        actual = invoke_operation(case.operation_id, payload, catalog).output
        expected = invoke_operation(
            case.operation_id, canonical_payload, catalog
        ).output
        assert actual == expected
        declaration = catalog.operation(case.operation_id)
        assert declaration is not None
        decoded = declaration.result_type.model_validate_json(
            json.dumps(actual), strict=True
        )
        assert decoded.model_dump(mode="json") == actual
        request = parse_operation_input(declaration.request_type, payload)
        assert request.model_dump(mode="json") == parse_operation_input(
            declaration.request_type, canonical_payload
        ).model_dump(mode="json")
        # Replacing an authored retained source/result ratio by an unreduced one
        # must still fail the *result* codec, even when the request accepts it.
        tampered = copy.deepcopy(actual)

        def replace_first_ratio(node: Any) -> bool:
            if isinstance(node, dict):
                if set(node) == {"num", "den"}:
                    node.update(num="2", den="4")
                    return True
                return any(replace_first_ratio(item) for item in node.values())
            if isinstance(node, list):
                return any(replace_first_ratio(item) for item in node)
            return False

        assert replace_first_ratio(tampered)
        with pytest.raises(ValidationError) as exc_info:
            declaration.result_type.model_validate_json(
                json.dumps(tampered), strict=True
            )
        # The tampered ratio can sit behind container codecs, so the
        # rational rejection is one of the reported errors, not always the first.
        assert any(
            error["type"] == "canonical_rational.noncanonical_representation"
            for error in exc_info.value.errors()
        )
    assert payload == unchanged


@pytest.mark.parametrize("case", RATIONAL_REQUEST_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("digits", [256, 257])
def test_raw_component_bound_precedes_reduction(
    case: RationalRequestCase, digits: int
) -> None:
    component = "2" + "0" * (digits - 1)
    payload = case.payload(ratio(component, component))
    if digits == 256:
        assert (
            invoke_operation(case.operation_id, payload, Catalog.open()).output
            == invoke_operation(
                case.operation_id, case.payload(ratio("1", "1")), Catalog.open()
            ).output
        )
    else:
        with pytest.raises(OperationRequestValidationError) as caught:
            invoke_operation(case.operation_id, payload, Catalog.open())
        assert caught.value.errors()[0]["type"] == (
            "matrix.rational_bound"
            if case.operation_id.startswith("linear.")
            else "matrix.budget_exceeded"
        )


@pytest.mark.parametrize(
    "component",
    [True, False, 2, 2.0, "+2", "02", "-0", " 2", "2 ", "2\n", "٢", "\uff12", "2e0"],
)
@pytest.mark.parametrize("field", ["num", "den"])
@pytest.mark.parametrize(
    "case",
    [
        RATIONAL_REQUEST_CASES[0],
        RATIONAL_REQUEST_CASES[9],
        RATIONAL_REQUEST_CASES[13],
        RATIONAL_REQUEST_CASES[-1],
    ],
)
def test_request_does_not_relax_component_grammar(
    case: RationalRequestCase, field: str, component: Any
) -> None:
    value: dict[str, Any] = ratio("2", "4")
    value[field] = component
    with pytest.raises(OperationRequestValidationError):
        invoke_operation(case.operation_id, case.payload(value), Catalog.open())


@pytest.mark.parametrize("case", RATIONAL_REQUEST_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(
    "value",
    [
        {"num": "1", "den": "0"},
        {"num": "2", "den": "4", "unexpected": "x"},
        {"num": "2"},
    ],
)
def test_every_rational_argument_rejects_malformed_ratio(
    case: RationalRequestCase, value: dict[str, Any]
) -> None:
    with pytest.raises(OperationRequestValidationError):
        invoke_operation(case.operation_id, case.payload(value), Catalog.open())


@pytest.mark.parametrize(
    "operation_id,axis",
    [
        ("matrix.determinant.compute", 128),
        ("matrix.permanent.compute", 128),
        ("matrix.characteristic_polynomial.compute", 128),
        ("matrix.multiply.compute", 128),
        ("matrix.kronecker_product.compute", 32),
        ("matrix.normal_form.rref.compute", 128),
        ("matrix.rank.compute", 128),
        ("matrix.nullspace.compute", 128),
        ("matrix.rational_linear_system.solve", 32),
        ("matrix.partial_trace.compute", 32),
    ],
)
def test_raw_dense_axis_boundary_is_unchanged(operation_id: str, axis: int) -> None:
    case = next(
        case for case in RATIONAL_REQUEST_CASES if case.operation_id == operation_id
    )
    payload = case.payload(ratio("2", "4"))
    field = "left" if case.argument == "left" else "matrix"
    payload[field] = {
        "row_count": 1,
        "column_count": axis,
        "entries": [[ratio("2", "4")] * axis],
    }
    operation = Catalog.open().operation(operation_id)
    assert operation is not None
    request = parse_operation_input(operation.request_type, payload)
    assert getattr(request, field).column_count == axis
    payload[field]["column_count"] += 1
    payload[field]["entries"][0].append(ratio("1", "0"))
    with pytest.raises(ValidationError) as caught:
        parse_operation_input(operation.request_type, payload)
    assert caught.value.errors()[0]["type"] == "matrix.budget_exceeded"


def test_solve_rhs_preflight_keeps_32_before_reduction() -> None:
    payload: dict[str, Any] = {"matrix": dense(ratio()), "rhs": [ratio("2", "4")] * 32}
    assert len(parse_operation_input(RationalLinearSolveRequest, payload).rhs) == 32
    payload["rhs"].append(ratio("1", "0"))
    with pytest.raises(ValidationError) as caught:
        parse_operation_input(RationalLinearSolveRequest, payload)
    assert caught.value.errors()[0]["type"] == "matrix.budget_exceeded"


@pytest.mark.parametrize(
    "case",
    [
        RATIONAL_REQUEST_CASES[9],
        RATIONAL_REQUEST_CASES[11],
        RATIONAL_REQUEST_CASES[15],
        RATIONAL_REQUEST_CASES[17],
    ],
)
@pytest.mark.parametrize(
    "mutation",
    ["unsorted", "duplicate", "outside", "domain", "zero", "extra", "scalar_extra"],
)
def test_sparse_normalization_preserves_coordinate_and_domain_contract(
    case: RationalRequestCase, mutation: str
) -> None:
    payload = case.payload(ratio("2", "4"))
    matrix = (
        payload["system"]["coefficients"] if "system" in payload else payload["matrix"]
    )
    matrix.update(row_count=2, column_count=2)
    matrix["entries"] = [
        {"row": 0, "column": 0, "value": ratio("2", "4")},
        {"row": 1, "column": 1, "value": ratio("2", "4")},
    ]
    if "system" in payload:
        payload["system"].update(variables=["y", "x"], rhs=[ratio(), ratio()])
    if mutation == "unsorted":
        matrix["entries"].reverse()
    elif mutation == "duplicate":
        matrix["entries"][1] = copy.deepcopy(matrix["entries"][0])
    elif mutation == "outside":
        matrix["entries"][1]["row"] = 2
    elif mutation == "domain":
        matrix["domain"] = "ZZ"
    elif mutation == "zero":
        matrix["entries"][1]["value"] = ratio("0", "-5")
    elif mutation == "extra":
        matrix["entries"][0]["num"] = "2"
    else:
        matrix["entries"][0]["value"]["extra"] = "ignored"
    with pytest.raises(OperationRequestValidationError):
        invoke_operation(case.operation_id, payload, Catalog.open())


def test_sparse_axis_and_stored_count_preflight_remain_bounded() -> None:
    matrix: dict[str, Any] = {
        "domain": "QQ",
        "row_count": 8192,
        "column_count": 8192,
        "entries": [
            {"row": row, "column": column, "value": ratio("2", "4")}
            for row in range(4)
            for column in range(8192)
        ],
    }
    request = parse_operation_input(MatrixRankRequest, {"matrix": matrix})
    assert len(request.matrix.entries) == 32768
    matrix["entries"].append({"row": 4, "column": 0, "value": ratio("1", "0")})
    with pytest.raises(ValidationError) as caught:
        parse_operation_input(MatrixRankRequest, {"matrix": matrix})
    assert caught.value.errors()[0]["type"] == "matrix.budget_exceeded"
    with pytest.raises(ValidationError):
        parse_operation_input(
            MatrixRankRequest,
            {"matrix": {"row_count": 8193, "column_count": 0, "entries": []}},
        )


def test_request_preflight_and_dense_shapes_are_not_repaired() -> None:
    for payload in [
        {"matrix": dense(ratio("2", "4")), "extra": ratio()},
        {"matrix": dict(dense(ratio("2", "4")), extra=ratio())},
        {"matrix": dict(dense(ratio("2", "4")), row_count=2)},
        {"matrix": dict(dense(ratio("2", "4")), domain="ZZ")},
    ]:
        with pytest.raises(ValidationError):
            parse_operation_input(MatrixDeterminantRequest, payload)


def test_downstream_dense_work_admission_is_not_widened() -> None:
    payload = {"matrix": {"entries": [[ratio("2", "4")] * 65]}}
    with pytest.raises(OperationDomainValidationError) as exc_info:
        invoke_operation("matrix.rank.compute", payload, Catalog.open())
    assert exc_info.value.errors()[0]["type"] == "matrix.budget_exceeded"


def test_real_product_output_feeds_requests_without_translation() -> None:
    catalog = Catalog.open()
    product = invoke_operation(
        "matrix.multiply.compute",
        {"left": dense(ratio("2", "4")), "right": dense(ratio("-6", "-8"))},
        catalog,
    ).output["product"]
    assert product["entries"] == [[ratio("3", "8")]]
    assert invoke_operation(
        "matrix.determinant.compute", {"matrix": product}, catalog
    ).output["determinant"] == ratio("3", "8")
    result = invoke_operation(
        "matrix.rational_linear_system.solve",
        {"matrix": product, "rhs": [ratio("6", "16")]},
        catalog,
    ).output
    assert result["solution"] == [ratio("1", "1")]
    retained = invoke_operation(
        "linear.rational_solution.compute",
        RATIONAL_REQUEST_CASES[15].payload(ratio("2", "4")),
        catalog,
    ).output["system"]
    assert (
        invoke_operation(
            "linear.rational_inconsistency.compute", {"system": retained}, catalog
        ).output["status"]
        == "CONSISTENT"
    )


def test_empty_sparse_and_dense_axes_survive_normalization() -> None:
    for rows, columns in [(0, 5), (4, 0), (0, 0)]:
        matrix = {
            "domain": "QQ",
            "row_count": rows,
            "column_count": columns,
            "entries": [] if rows == 0 else [[] for _ in range(rows)],
        }
        request = parse_operation_input(MatrixRankRequest, {"matrix": matrix})
        assert (request.matrix.row_count, request.matrix.column_count) == (
            rows,
            columns,
        )
        empty_sparse = {
            "domain": "QQ",
            "row_count": rows,
            "column_count": columns,
            "entries": [],
        }
        assert (
            invoke_operation(
                "matrix.rank.compute", {"matrix": empty_sparse}, Catalog.open()
            ).output["matrix"]
            == empty_sparse
        )


@pytest.mark.parametrize(
    "operation_id",
    ["linear.rational_solution.compute", "linear.rational_inconsistency.compute"],
)
@pytest.mark.parametrize("axis", ["variables", "rhs", "entries"])
def test_rational_linear_raw_container_limits_precede_normalization(
    operation_id: str, axis: str
) -> None:
    case = next(
        case for case in RATIONAL_REQUEST_CASES if case.operation_id == operation_id
    )
    operation = Catalog.open().operation(operation_id)
    assert operation is not None
    payload = case.payload(ratio("2", "4"))
    system = payload["system"]
    if axis == "variables":
        system["variables"] = [f"x_{index}" for index in range(8192)]
        system["coefficients"]["column_count"] = 8192
        assert (
            len(parse_operation_input(operation.request_type, payload).system.variables)
            == 8192
        )
        system["variables"].append("overflow")
    elif axis == "rhs":
        system["rhs"] = [ratio("2", "4")] * 8192
        system["coefficients"]["row_count"] = 8192
        assert (
            len(parse_operation_input(operation.request_type, payload).system.rhs)
            == 8192
        )
        system["rhs"].append(ratio("1", "0"))
    else:
        system["variables"] = [f"x_{index}" for index in range(8192)]
        system["rhs"] = [ratio("2", "4")] * 4
        system["coefficients"] = {
            "row_count": 4,
            "column_count": 8192,
            "entries": [
                {"row": row, "column": column, "value": ratio("2", "4")}
                for row in range(4)
                for column in range(8192)
            ],
        }
        assert (
            len(
                parse_operation_input(
                    operation.request_type, payload
                ).system.coefficients.entries
            )
            == 32768
        )
        system["coefficients"]["entries"].append(
            {"row": 0, "column": 0, "value": ratio("1", "0")}
        )
    with pytest.raises(ValidationError) as caught:
        parse_operation_input(operation.request_type, payload)
    assert caught.value.errors()[0]["type"] == "matrix.budget_exceeded"
