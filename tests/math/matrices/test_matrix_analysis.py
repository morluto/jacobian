"""Tests for matrix analysis operations."""

import copy
from fractions import Fraction

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.analysis._models import (
    FarkasCertificateRequest,
    InertiaResult,
    SymmetricMatrixRequest,
)
from jacobian.math.matrices.analysis._tools import (
    check_farkas_certificate,
    compute_inertia,
)
from jacobian.math.matrices.analysis.operations import (
    check_farkas_certificate as check_farkas_certificate_native,
)
from jacobian.math.matrices.analysis.operations import (
    compute_inertia as compute_inertia_native,
)
from jacobian.math.matrices.values import (
    MAX_MATRIX_DIMENSION,
    MAX_RATIONAL_MATRIX_ORDER,
    RationalMatrix,
)


class TestInertia:
    def test_native_api_accepts_canonical_matrix(self) -> None:
        matrix = RationalMatrix(entries=((CanonicalRational(num="1", den="1"),),))
        result = compute_inertia_native(matrix)
        assert result.matrix == matrix
        assert (result.n_positive, result.n_negative, result.n_zero) == (1, 0, 0)

    def test_wire_request_consumes_a_serialized_rational_result_matrix(self) -> None:
        matrix = RationalMatrix(
            entries=(
                (
                    CanonicalRational(num="1", den="1"),
                    CanonicalRational(num="0", den="1"),
                ),
                (
                    CanonicalRational(num="0", den="1"),
                    CanonicalRational(num="-1", den="1"),
                ),
            )
        )
        produced = compute_inertia_native(matrix)
        request = SymmetricMatrixRequest.model_validate_json(
            encode_strict_json({"matrix": produced.matrix.model_dump(mode="json")}),
            strict=True,
        )

        round_tripped = compute_inertia(request)

        assert round_tripped == produced

    def test_identity(self) -> None:
        req = _inertia_request(3, {(0, 0): "1", (1, 1): "1", (2, 2): "1"})
        result = compute_inertia(req)
        assert result.n_positive == 3
        assert result.n_negative == 0
        assert result.n_zero == 0
        assert result.definiteness == "positive_definite"

    def test_negative_identity(self) -> None:
        req = _inertia_request(2, {(0, 0): "-1", (1, 1): "-1"})
        result = compute_inertia(req)
        assert result.n_positive == 0
        assert result.n_negative == 2
        assert result.definiteness == "negative_definite"

    def test_indefinite(self) -> None:
        req = _inertia_request(2, {(0, 0): "1", (1, 1): "-1"})
        result = compute_inertia(req)
        assert result.n_positive == 1
        assert result.n_negative == 1
        assert result.definiteness == "indefinite"

    def test_singular_all_ones_matrix_uses_original_schur_entries(self) -> None:
        req = _inertia_request(
            3,
            {(row, column): "1" for row in range(3) for column in range(3)},
        )

        result = compute_inertia(req)

        assert (result.n_positive, result.n_negative, result.n_zero) == (1, 0, 2)

    def test_off_diagonal_hyperbolic_pair(self) -> None:
        req = _inertia_request(2, {(0, 1): "1"})
        result = compute_inertia(req)
        assert result.n_positive == 1
        assert result.n_negative == 1
        assert result.n_zero == 0
        assert result.definiteness == "indefinite"

    def test_rejects_asymmetric_canonical_matrix_as_a_domain_error(self) -> None:
        request = SymmetricMatrixRequest(
            matrix=RationalMatrix(
                entries=(
                    (
                        CanonicalRational(num="0", den="1"),
                        CanonicalRational(num="1", den="1"),
                    ),
                    (
                        CanonicalRational(num="2", den="1"),
                        CanonicalRational(num="0", den="1"),
                    ),
                )
            )
        )
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_inertia(request)
        assert exc_info.value.errors()[0]["type"] == "matrix.shape_mismatch"


class TestFarkas:
    def test_native_api_accepts_canonical_certificate_fields(self) -> None:
        one = CanonicalRational(num="1", den="1")
        negative_one = CanonicalRational(num="-1", den="1")
        result = check_farkas_certificate_native(
            ((one, one), (negative_one, negative_one)),
            (negative_one, negative_one),
            (one, one),
        )
        assert result.valid is True

    def test_valid_certificate(self) -> None:
        # System: x1 + x2 <= -1, x1 + x2 >= 1 is infeasible.
        # A = [[1, 1], [-1, -1]], b = [-1, -1]
        # y = (1, 1), y^T A = (1-1, 1-1) = (0, 0), y^T b = -1 + -1 = -2 < 0 => valid
        req = FarkasCertificateRequest.model_validate(
            {
                "constraint_matrix": [
                    ({"num": "1", "den": "1"}, {"num": "1", "den": "1"}),
                    ({"num": "-1", "den": "1"}, {"num": "-1", "den": "1"}),
                ],
                "rhs_vector": (
                    {"num": "-1", "den": "1"},
                    {"num": "-1", "den": "1"},
                ),
                "multipliers": (
                    {"num": "1", "den": "1"},
                    {"num": "1", "den": "1"},
                ),
            }
        )
        result = check_farkas_certificate(req)
        assert result.valid is True

    def test_rejects_nonrectangular_matrix(self) -> None:
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FarkasCertificateRequest.model_validate(
                {
                    "constraint_matrix": [
                        ({"num": "1", "den": "1"}, {"num": "1", "den": "1"}),
                        ({"num": "-1", "den": "1"},),
                    ],
                    "rhs_vector": (
                        {"num": "-1", "den": "1"},
                        {"num": "-1", "den": "1"},
                    ),
                    "multipliers": (
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                    ),
                }
            )


# ---------------------------------------------------------------------------
# Source-bound inertia regressions (#2297)
# ---------------------------------------------------------------------------


def _inertia_request(
    dimension: int, entries: dict[tuple[int, int], str]
) -> SymmetricMatrixRequest:
    zero = CanonicalRational(num="0", den="1")
    dense = [[zero for _ in range(dimension)] for _ in range(dimension)]
    for (row, column), encoded in entries.items():
        value = CanonicalRational(
            num=encoded.split("/")[0],
            den=encoded.split("/")[1] if "/" in encoded else "1",
        )
        dense[row][column] = value
        dense[column][row] = value
    return SymmetricMatrixRequest(
        matrix=RationalMatrix(entries=tuple(tuple(row) for row in dense))
    )


@pytest.mark.parametrize(
    ("dimension", "entries", "counts", "label"),
    (
        # zero matrix (explicit zero entry)
        (2, {(0, 0): "0"}, (0, 0, 2), "zero"),
        # singular psd
        (2, {(0, 0): "1", (1, 1): "0"}, (1, 0, 1), "positive_semidefinite"),
        # negative semidefinite
        (2, {(0, 0): "-1", (1, 1): "0"}, (0, 1, 1), "negative_semidefinite"),
        # positive definite with rational entries
        (2, {(0, 0): "3/2", (1, 1): "2"}, (2, 0, 0), "positive_definite"),
        # negative definite
        (2, {(0, 0): "-1", (1, 1): "-5"}, (0, 2, 0), "negative_definite"),
        # indefinite off-diagonal
        (
            2,
            {(0, 0): "0", (1, 1): "0", (0, 1): "1"},
            (1, 1, 0),
            "indefinite",
        ),
    ),
)
def test_inertia_results_round_trip_known_answers(
    dimension: int,
    entries: dict[tuple[int, int], str],
    counts: tuple[int, int, int],
    label: str,
) -> None:
    request = _inertia_request(dimension, entries)
    result = compute_inertia(request)
    assert (result.n_positive, result.n_negative, result.n_zero) == counts
    assert result.definiteness == label
    assert InertiaResult.model_validate(result.model_dump()) == result


def test_inertia_result_rejects_structural_mutations() -> None:
    request = _inertia_request(2, {(0, 0): "1"})
    result = compute_inertia(request)
    dumped = result.model_dump(mode="json")

    count_sum = copy.deepcopy(dumped)
    count_sum["n_zero"] = 5
    with pytest.raises(ValidationError):
        InertiaResult.model_validate(count_sum)

    wrong_label = copy.deepcopy(dumped)
    wrong_label["definiteness"] = "indefinite"
    with pytest.raises(ValidationError):
        InertiaResult.model_validate(wrong_label)

    foreign_source = copy.deepcopy(dumped)
    foreign_source["matrix"]["entries"][0][0] = {"num": "-1", "den": "1"}
    supplied = InertiaResult.model_validate(foreign_source)
    assert isinstance(supplied.matrix, RationalMatrix)
    assert supplied.matrix.entries[0][0].as_fraction() == Fraction(-1)

    asymmetric_source = copy.deepcopy(dumped)
    asymmetric_source["matrix"]["entries"][0][1] = {"num": "3", "den": "1"}
    with pytest.raises(ValidationError):
        InertiaResult.model_validate(asymmetric_source)

    nonsquare_source = copy.deepcopy(dumped)
    nonsquare_source["matrix"]["entries"] = (
        ({"num": "1", "den": "1"},),
        ({"num": "0", "den": "1"},),
    )
    with pytest.raises(ValidationError):
        InertiaResult.model_validate(nonsquare_source)

    forged_counts = copy.deepcopy(dumped)
    forged_counts["n_positive"] = 0
    forged_counts["n_negative"] = 1
    with pytest.raises(ValidationError):
        InertiaResult.model_validate(forged_counts)


def test_inertia_congruence_invariance() -> None:
    """Invertible rational change of basis preserves the exact inertia counts."""

    from sympy import Matrix

    request = _inertia_request(2, {(0, 0): "4", (1, 1): "9/4", (0, 1): "3"})
    result = compute_inertia(request)
    assert result.definiteness in {
        "positive_definite",
        "positive_semidefinite",
    }

    dense = Matrix([[4, 3], [3, Fraction(9, 4)]])
    change = Matrix([[2, 1], [0, 3]])
    congruent = change.T * dense * change
    entries = {
        (i, j): f"{congruent[i, j].p}/{congruent[i, j].q}"
        for i in range(2)
        for j in range(i, 2)
        if congruent[i, j] != 0 or (i, i) == (j, j)
    }
    transformed = compute_inertia(_inertia_request(2, entries))
    assert (
        transformed.n_positive,
        transformed.n_negative,
        transformed.n_zero,
    ) == (result.n_positive, result.n_negative, result.n_zero)


def test_inertia_result_retains_domain_canonical_matrix() -> None:
    request = _inertia_request(2, {(0, 0): "3/2"})
    result = compute_inertia(request)
    assert isinstance(result.matrix, RationalMatrix)
    assert result.matrix.domain == "QQ"
    assert result.matrix.entries == (
        (CanonicalRational(num="3", den="2"), CanonicalRational(num="0", den="1")),
        (CanonicalRational(num="0", den="1"), CanonicalRational(num="0", den="1")),
    )


def test_inertia_canonical_request_has_one_schema_truthful_matrix_field() -> None:
    schema = SymmetricMatrixRequest.model_json_schema()
    assert schema["required"] == ["matrix"]
    assert set(schema["properties"]) == {"matrix"}
    assert schema["additionalProperties"] is False
    assert schema["x-jacobian-bounds"] == {
        "max_matrix_order": MAX_RATIONAL_MATRIX_ORDER,
        "max_algebraic_field_degree": 8,
        "max_exact_digit_work": 500_000_000,
        "result_envelope_reserve_bytes": 1_024,
        "diagonal_fast_path": True,
    }
    valid = {
        "matrix": {
            "domain": "QQ",
            "entries": [[{"num": "1", "den": "1"}]],
        }
    }
    missing_discriminator = copy.deepcopy(valid)
    del missing_discriminator["matrix"]["domain"]
    validator = Draft202012Validator(schema)
    assert not list(validator.iter_errors(valid))
    assert list(validator.iter_errors(missing_discriminator))
    with pytest.raises(ValidationError):
        SymmetricMatrixRequest.model_validate(missing_discriminator)


def test_inertia_result_schema_matches_strict_discrimination_and_labels() -> None:
    result = compute_inertia(_inertia_request(1, {(0, 0): "1"}))
    payload = result.model_dump(mode="json")
    missing_discriminator = copy.deepcopy(payload)
    del missing_discriminator["matrix"]["domain"]
    unknown_label = copy.deepcopy(payload)
    unknown_label["definiteness"] = "unknown"

    validator = Draft202012Validator(InertiaResult.model_json_schema())
    assert not list(validator.iter_errors(payload))
    for invalid in (missing_discriminator, unknown_label):
        assert list(validator.iter_errors(invalid))
        with pytest.raises(ValidationError):
            InertiaResult.model_validate(invalid)


def test_inertia_retained_matrix_reconstructs_the_source() -> None:
    request = _inertia_request(3, {(0, 1): "2/3", (2, 2): "-5"})
    retained = compute_inertia(request).matrix
    assert isinstance(retained, RationalMatrix)
    dense = [[entry.as_fraction() for entry in row] for row in retained.entries]
    assert dense == [
        [Fraction(0), Fraction(2, 3), Fraction(0)],
        [Fraction(2, 3), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(-5)],
    ]


def test_inertia_request_admits_order_33_diagonal_source() -> None:
    # The canonical dense RationalMatrix retains sources up to its own order
    # envelope, so a previously valid order 33 request must still parse,
    # compute, and return a source-bound typed result.
    request = _inertia_request(33, {(index, index): "1" for index in range(33)})
    result = compute_inertia(request)

    assert (result.n_positive, result.n_negative, result.n_zero) == (33, 0, 0)
    assert result.definiteness == "positive_definite"
    assert len(result.matrix.entries) == 33
    assert isinstance(result.matrix, RationalMatrix)


def test_inertia_accepts_diagonal_carrier_boundary() -> None:
    result = compute_inertia(
        _inertia_request(
            MAX_RATIONAL_MATRIX_ORDER,
            {(0, 0): "1", (MAX_RATIONAL_MATRIX_ORDER - 1,) * 2: "-1"},
        )
    )
    assert (result.n_positive, result.n_negative, result.n_zero) == (
        1,
        1,
        MAX_RATIONAL_MATRIX_ORDER - 2,
    )


def _encoded_inertia_payload_near_limit(offset: int) -> bytes:
    """Encode an inertia request whose normalized dense source echo lands
    exactly ``offset`` bytes below the canonical output limit, so the echo
    plus the reserved envelope may exceed the identical output limit while
    the payload still fits the input limit."""

    import functools

    from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
    from jacobian.canonical import CanonicalLimits, encode_strict_json

    @functools.cache
    def build(offset: int) -> bytes:
        limits = CanonicalLimits()
        dimension = MAX_MATRIX_DIMENSION
        cells = [(r, c) for r in range(dimension) for c in range(r, dimension)]

        def dense_echo(digits: dict[tuple[int, int], int]) -> bytes:
            rows = [
                [
                    {
                        "num": "9" * digits[(min(r, c), max(r, c))],
                        "den": "1",
                    }
                    for c in range(dimension)
                ]
                for r in range(dimension)
            ]
            return encode_strict_json({"domain": "QQ", "entries": rows})

        target = limits.max_output_bytes - offset
        low = len(dense_echo(dict.fromkeys(cells, 1)))
        uniform = max(1, (target - low) // (dimension * dimension))
        digits = dict.fromkeys(cells, uniform)
        gap = target - len(dense_echo(digits))
        first, second = cells[0], cells[1]
        adjusted = digits[first] + gap
        if adjusted < 1:
            digits[second] += adjusted - 1
            adjusted = 1
        elif adjusted > MAX_CANONICAL_RATIONAL_DIGITS:
            digits[second] += adjusted - MAX_CANONICAL_RATIONAL_DIGITS
            adjusted = MAX_CANONICAL_RATIONAL_DIGITS
        assert 1 <= digits[second] <= MAX_CANONICAL_RATIONAL_DIGITS
        digits[first] = adjusted
        assert len(dense_echo(digits)) == target
        dense = dense_echo(digits)
        encoded = encode_strict_json({"matrix": __import__("json").loads(dense)})
        assert len(encoded) <= limits.max_input_bytes
        return encoded

    return build(offset)


@pytest.mark.scale
def test_inertia_request_admission_reserves_output_headroom_for_source_echo() -> None:
    from jacobian.canonical import CanonicalLimits

    encoded = _encoded_inertia_payload_near_limit(offset=512)
    assert len(encoded) <= CanonicalLimits().max_output_bytes
    request = SymmetricMatrixRequest.model_validate_json(encoded)
    with pytest.raises(OperationDomainValidationError):
        compute_inertia(request)


@pytest.mark.scale
def test_inertia_request_admission_accepts_payload_inside_reserved_budget() -> None:
    encoded = _encoded_inertia_payload_near_limit(offset=2048)
    request = SymmetricMatrixRequest.model_validate_json(encoded)
    assert len(request.matrix.entries) == MAX_MATRIX_DIMENSION
