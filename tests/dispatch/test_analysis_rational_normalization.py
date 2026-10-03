"""Full-carrier rational normalization preserves analysis admission and values."""

import copy
import json
from fractions import Fraction
from typing import Any

import pytest
from pydantic import ValidationError
from tests.dispatch._analysis_rational_cases import (
    ANALYSIS_RATIONAL_CASES,
    AnalysisRationalCase,
)
from tests.dispatch._rational_request_cases import dense, ratio

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import (
    OperationRequestValidationError,
    invoke_operation,
    parse_operation_input,
)
from jacobian.math.matrices.completion._models import PartialSymmetricRationalMatrix
from jacobian.math.matrices.values import RationalMatrix, SparseRationalMatrixEntry


def run(case: AnalysisRationalCase, value: dict[str, Any]) -> dict[str, Any]:
    return invoke_operation(case.operation, case.payload(value), Catalog.open()).output


@pytest.mark.parametrize("case", ANALYSIS_RATIONAL_CASES)
@pytest.mark.parametrize(
    "num,den", [("2", "4"), ("-2", "-4"), ("0", "5"), ("0", "-5"), ("2", "-4")]
)
def test_all_positions_normalize_without_changing_semantics(
    case: AnalysisRationalCase, num: str, den: str
) -> None:
    value = Fraction(int(num), int(den))
    canonical = ratio(str(value.numerator), str(value.denominator))
    raw = ratio(num, den)
    invalid_domain = (
        "collatz" in case.operation
        and (value < 0 or (case.position == "vector" and value == 0))
    ) or ("decompose" in case.operation and value < 0)
    if invalid_domain:
        with pytest.raises(OperationDomainValidationError) as expected:
            run(case, canonical)
        with pytest.raises(OperationDomainValidationError) as actual:
            run(case, raw)
        assert actual.value.errors() == expected.value.errors()
        return
    payload = case.payload(raw)
    unchanged = copy.deepcopy(payload)
    output = invoke_operation(case.operation, payload, Catalog.open()).output
    assert output == run(case, canonical)
    assert payload == unchanged
    operation = Catalog.open().operation(case.operation)
    assert operation is not None
    parsed = parse_operation_input(operation.request_type, payload)
    matrix = parsed.matrix
    assert type(matrix) is (
        PartialSymmetricRationalMatrix
        if "completion" in case.operation
        else RationalMatrix
    )
    if "completion" in case.operation:
        assert isinstance(matrix, PartialSymmetricRationalMatrix)
        assert type(matrix.specified_entries[0]) is SparseRationalMatrixEntry
        scalar = matrix.specified_entries[0].value
    else:
        assert isinstance(matrix, RationalMatrix)
        scalar = matrix.entries[0][0]
    assert type(scalar) is CanonicalRational
    if "collatz" in case.operation:
        assert all(type(v) is CanonicalRational for v in parsed.vector)
    assert (
        operation.result_type.model_validate_json(
            json.dumps(output), strict=True
        ).model_dump(mode="json")
        == output
    )
    corrupted = copy.deepcopy(output)
    if "completion" in case.operation:
        corrupted["matrix"]["specified_entries"][0]["value"] = raw
    elif case.position == "vector":
        corrupted["vector"][0] = raw
    else:
        corrupted["matrix"]["entries"][0][0] = raw
    with pytest.raises(ValidationError):
        operation.result_type.model_validate_json(json.dumps(corrupted), strict=True)
    # Native/Python request construction retains strict canonical scalar rules.
    with pytest.raises(ValidationError):
        operation.request_type.model_validate(payload, strict=True)


@pytest.mark.parametrize("case", ANALYSIS_RATIONAL_CASES)
@pytest.mark.parametrize("digits", [32768, 32769])
@pytest.mark.parametrize("sign", ["", "-"])
def test_raw_component_limit_precedes_reduction(
    case: AnalysisRationalCase, digits: int, sign: str
) -> None:
    component = sign + "9" * digits
    if digits == 32768:
        assert run(case, ratio(component, component)) == run(case, ratio("1", "1"))
    else:
        with pytest.raises(OperationRequestValidationError):
            run(case, ratio(component, component))


@pytest.mark.parametrize("case", ANALYSIS_RATIONAL_CASES)
def test_existing_4096_digit_sources_remain_accepted(
    case: AnalysisRationalCase,
) -> None:
    output = run(case, ratio("9" * 4096, "1"))
    assert output


def test_inertia_retains_full_carrier_and_other_owner_resource_refusals() -> None:
    assert run(ANALYSIS_RATIONAL_CASES[0], ratio("9" * 32768, "1"))["n_positive"] == 1
    for case in ANALYSIS_RATIONAL_CASES[1:]:
        with pytest.raises(OperationDomainValidationError) as canonical:
            run(case, ratio("9" * 32768, "1"))
        with pytest.raises(OperationDomainValidationError) as normalized:
            run(case, ratio("-" + "9" * 32768, "-1"))
        assert normalized.value.errors() == canonical.value.errors()
        assert type(normalized.value) is type(canonical.value)
        if "chordal" in case.operation:
            assert isinstance(normalized.value, OperationResourceAdmissionError)


@pytest.mark.parametrize("case", ANALYSIS_RATIONAL_CASES)
@pytest.mark.parametrize(
    "raw", [True, 2, 2.0, "+2", "02", "-0", " 2", "2\n", "٢", "2e0"]
)
@pytest.mark.parametrize("field", ["num", "den"])
def test_component_grammar_remains_strict(
    case: AnalysisRationalCase, raw: Any, field: str
) -> None:
    value: dict[str, Any] = ratio("2", "4")
    value[field] = raw
    with pytest.raises(OperationRequestValidationError):
        run(case, value)


@pytest.mark.parametrize("case", ANALYSIS_RATIONAL_CASES)
@pytest.mark.parametrize(
    "value",
    [{"num": "1", "den": "0"}, {"num": "2"}, {"num": "2", "den": "4", "extra": 1}],
)
def test_ratio_shape_remains_strict(
    case: AnalysisRationalCase, value: dict[str, Any]
) -> None:
    with pytest.raises(OperationRequestValidationError):
        run(case, value)


def test_graph_and_ratio_normalization_compose_without_moving_axes() -> None:
    matrix: dict[str, Any] = {
        "graph": {"vertex_count": 2, "edges": [[1, 0]]},
        "specified_entries": [
            {"row": 0, "column": 0, "value": ratio("4", "2")},
            {"row": 0, "column": 1, "value": ratio("0", "-5")},
            {"row": 1, "column": 1, "value": ratio("-6", "-2")},
        ],
    }
    op = "matrix.chordal_psd_completion.compute"
    output = invoke_operation(op, {"matrix": matrix}, Catalog.open()).output
    assert output["matrix"]["graph"]["edges"] == [[0, 1]]
    assert output["outcome"]["completion"]["entries"] == [
        [ratio("2", "1"), ratio("0", "1")],
        [ratio("0", "1"), ratio("3", "1")],
    ]
    canonical = copy.deepcopy(output["matrix"])
    assert invoke_operation(op, {"matrix": canonical}, Catalog.open()).output == output
    for entries in (
        matrix["specified_entries"][::-1],
        matrix["specified_entries"][:1] + matrix["specified_entries"][2:],
        matrix["specified_entries"] + matrix["specified_entries"][:1],
    ):
        with pytest.raises(OperationRequestValidationError):
            invoke_operation(
                op, {"matrix": {**matrix, "specified_entries": entries}}, Catalog.open()
            )
    with pytest.raises(ValidationError):
        PartialSymmetricRationalMatrix.model_validate_json(
            json.dumps(matrix), strict=True
        )


def test_nonsymmetric_inertia_and_uncovered_psd_support_still_reject() -> None:
    matrix = {
        "entries": [
            [ratio("2", "2"), ratio("2", "4")],
            [ratio("0", "5"), ratio("2", "2")],
        ]
    }
    with pytest.raises(OperationDomainValidationError):
        invoke_operation("matrix.inertia.compute", {"matrix": matrix}, Catalog.open())
    matrix["entries"][1][0] = ratio("2", "4")
    with pytest.raises(OperationDomainValidationError):
        invoke_operation(
            "matrix.chordal_psd.decompose",
            {"matrix": matrix, "graph": {"vertex_count": 2, "edges": []}},
            Catalog.open(),
        )


def test_raw_sequence_counts_are_checked_before_scalar_reduction() -> None:
    # Invalid ratio elements would produce scalar errors if visited first.
    op = Catalog.open().operation(ANALYSIS_RATIONAL_CASES[1].operation)
    assert op is not None
    with pytest.raises(ValidationError) as vector:
        parse_operation_input(
            op.request_type, {"matrix": dense(ratio()), "vector": [None] * 8193}
        )
    assert vector.value.errors()[0]["type"] == "matrix.budget_exceeded"
    op = Catalog.open().operation(ANALYSIS_RATIONAL_CASES[-1].operation)
    assert op is not None
    with pytest.raises(ValidationError) as entries:
        parse_operation_input(
            op.request_type,
            {
                "matrix": {
                    "graph": {"vertex_count": 0, "edges": []},
                    "specified_entries": [None] * 66561,
                }
            },
        )
    assert entries.value.errors()[0]["type"] == "matrix.budget_exceeded"


def test_inertia_embedded_arm_remains_canonical_and_selects_the_embedding() -> None:
    from jacobian.math.matrices.values import EmbeddedRealSimpleNumberFieldMatrix
    from jacobian.math.number_theory.number_fields import (
        RealNumberFieldEmbedding,
        SimpleNumberFieldElement,
        SimpleNumberFieldPresentation,
        embeddings,
    )

    presentation = SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -2))
    embedding = embeddings(presentation).records[1].embedding
    assert isinstance(embedding, RealNumberFieldEmbedding)
    element = SimpleNumberFieldElement(
        presentation=presentation,
        coefficients_ascending=(
            CanonicalRational(num=0, den=1),
            CanonicalRational(num=1, den=1),
        ),
    )
    matrix = EmbeddedRealSimpleNumberFieldMatrix(
        embedding=embedding, entries=((element,),)
    )
    payload = {"matrix": matrix.model_dump(mode="json")}
    operation = Catalog.open().operation("matrix.inertia.compute")
    assert operation is not None
    parsed = parse_operation_input(operation.request_type, payload)
    assert type(parsed.matrix) is EmbeddedRealSimpleNumberFieldMatrix
    assert parsed.matrix == matrix
    output = invoke_operation(operation.operation_id, payload, Catalog.open()).output
    assert output["n_positive"] == 1
    payload["matrix"]["entries"][0][0]["coefficients_ascending"][1] = ratio("2", "2")
    with pytest.raises(OperationRequestValidationError):
        invoke_operation(operation.operation_id, payload, Catalog.open())


@pytest.mark.parametrize(
    "case",
    [
        ANALYSIS_RATIONAL_CASES[0],
        ANALYSIS_RATIONAL_CASES[3],
        ANALYSIS_RATIONAL_CASES[4],
    ],
)
def test_empty_axes_remain_canonical(case: AnalysisRationalCase) -> None:
    payload: dict[str, Any]
    if "completion" in case.operation:
        payload = {
            "matrix": {
                "graph": {"vertex_count": 0, "edges": []},
                "specified_entries": [],
            }
        }
    else:
        payload = {
            "matrix": {"domain": "QQ", "row_count": 0, "column_count": 0, "entries": []}
        }
        if "decompose" in case.operation:
            payload["graph"] = {"vertex_count": 0, "edges": []}
    operation = Catalog.open().operation(case.operation)
    assert operation is not None
    if case.operation == "matrix.inertia.compute":
        with pytest.raises(OperationDomainValidationError) as exc_info:
            invoke_operation(case.operation, payload, Catalog.open())
        assert exc_info.value.errors()[0]["type"] == "matrix.shape_mismatch"
        return
    output = invoke_operation(case.operation, payload, Catalog.open()).output
    assert (
        operation.result_type.model_validate_json(
            json.dumps(output), strict=True
        ).model_dump(mode="json")
        == output
    )


def test_native_collatz_vector_does_not_gain_list_coercion() -> None:
    from jacobian.math.matrices.collatz_wielandt._models import CollatzWielandtRequest

    matrix = RationalMatrix(entries=((CanonicalRational(num=1, den=1),),))
    scalar = CanonicalRational(num=1, den=1)
    assert CollatzWielandtRequest(matrix=matrix, vector=(scalar,)).vector == (scalar,)
    with pytest.raises(ValidationError):
        CollatzWielandtRequest.model_validate(
            {"matrix": matrix, "vector": [scalar]}, strict=True
        )
