"""Tests for matrix analysis operations."""

import copy
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.matrices.analysis._models import (
    MAX_SYMMETRIC_MATRIX_DIMENSION,
    FarkasCertificateRequest,
    InertiaResult,
    MatrixEntry,
    SymmetricMatrixRequest,
)
from jacobian.math.matrices.analysis._operations import (
    check_farkas_certificate,
    compute_inertia,
    verify_inertia_result,
)
from jacobian.math.matrices.values import (
    MAX_MATRIX_DIMENSION,
    MAX_RATIONAL_MATRIX_ORDER,
    RationalMatrix,
)


class TestInertia:
    def test_identity(self):
        req = SymmetricMatrixRequest(
            dimension=3,
            entries=(
                MatrixEntry(row=0, col=0, value={"num": "1", "den": "1"}),
                MatrixEntry(row=1, col=1, value={"num": "1", "den": "1"}),
                MatrixEntry(row=2, col=2, value={"num": "1", "den": "1"}),
            ),
        )
        result = compute_inertia(req)
        assert result.n_positive == 3
        assert result.n_negative == 0
        assert result.n_zero == 0
        assert result.definiteness == "positive_definite"

    def test_negative_identity(self):
        req = SymmetricMatrixRequest(
            dimension=2,
            entries=(
                MatrixEntry(row=0, col=0, value={"num": "-1", "den": "1"}),
                MatrixEntry(row=1, col=1, value={"num": "-1", "den": "1"}),
            ),
        )
        result = compute_inertia(req)
        assert result.n_positive == 0
        assert result.n_negative == 2
        assert result.definiteness == "negative_definite"

    def test_indefinite(self):
        req = SymmetricMatrixRequest(
            dimension=2,
            entries=(
                MatrixEntry(row=0, col=0, value={"num": "1", "den": "1"}),
                MatrixEntry(row=1, col=1, value={"num": "-1", "den": "1"}),
            ),
        )
        result = compute_inertia(req)
        assert result.n_positive == 1
        assert result.n_negative == 1
        assert result.definiteness == "indefinite"

    def test_off_diagonal_hyperbolic_pair(self):
        req = SymmetricMatrixRequest(
            dimension=2,
            entries=(MatrixEntry(row=0, col=1, value={"num": "1", "den": "1"}),),
        )
        result = compute_inertia(req)
        assert result.n_positive == 1
        assert result.n_negative == 1
        assert result.n_zero == 0
        assert result.definiteness == "indefinite"

    def test_rejects_conflicting_symmetric_entries(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SymmetricMatrixRequest(
                dimension=2,
                entries=(
                    MatrixEntry(row=0, col=1, value={"num": "1", "den": "1"}),
                    MatrixEntry(row=1, col=0, value={"num": "2", "den": "1"}),
                ),
            )


class TestFarkas:
    def test_valid_certificate(self):
        # System: x1 + x2 <= -1, x1 + x2 >= 1 is infeasible.
        # A = [[1, 1], [-1, -1]], b = [-1, -1]
        # y = (1, 1), y^T A = (1-1, 1-1) = (0, 0), y^T b = -1 + -1 = -2 < 0 => valid
        req = FarkasCertificateRequest(
            constraint_matrix=[
                ({"num": "1", "den": "1"}, {"num": "1", "den": "1"}),
                ({"num": "-1", "den": "1"}, {"num": "-1", "den": "1"}),
            ],
            rhs_vector=(
                {"num": "-1", "den": "1"},
                {"num": "-1", "den": "1"},
            ),
            multipliers=(
                {"num": "1", "den": "1"},
                {"num": "1", "den": "1"},
            ),
        )
        result = check_farkas_certificate(req)
        assert result.valid is True

    def test_rejects_nonrectangular_matrix(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FarkasCertificateRequest(
                constraint_matrix=[
                    ({"num": "1", "den": "1"}, {"num": "1", "den": "1"}),
                    ({"num": "-1", "den": "1"},),
                ],
                rhs_vector=(
                    {"num": "-1", "den": "1"},
                    {"num": "-1", "den": "1"},
                ),
                multipliers=(
                    {"num": "1", "den": "1"},
                    {"num": "1", "den": "1"},
                ),
            )


# ---------------------------------------------------------------------------
# Source-bound inertia regressions (#2297)
# ---------------------------------------------------------------------------


def _inertia_request(dimension: int, entries: dict[tuple[int, int], str]):
    from jacobian.math.matrices.analysis._models import MatrixEntry

    return SymmetricMatrixRequest(
        dimension=dimension,
        entries=tuple(
            MatrixEntry(
                row=r,
                col=c,
                value=CanonicalRational(
                    num=v.split("/")[0], den=(v.split("/")[1] if "/" in v else "1")
                ),
            )
            for (r, c), v in entries.items()
        ),
    )


@pytest.mark.parametrize(
    ("dimension", "entries", "counts", "label"),
    (
        # zero matrix (explicit zero entry)
        (2, {(0, 0): "0"}, (0, 0, 2), "positive_semidefinite"),
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
def test_inertia_results_replay_known_answers(
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
    assert verify_inertia_result(supplied) is False

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
    """Invertible rational change of basis preserves the replayed counts."""

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


def test_inertia_results_are_representation_invariant() -> None:
    upper = compute_inertia(_inertia_request(3, {(0, 1): "2", (2, 2): "-5"}))
    lower = compute_inertia(_inertia_request(3, {(1, 0): "2", (2, 2): "-5"}))
    reordered = compute_inertia(_inertia_request(3, {(2, 2): "-5", (0, 1): "2"}))
    padded = compute_inertia(
        _inertia_request(
            3,
            {(0, 0): "0", (0, 1): "2", (1, 1): "0", (2, 2): "-5"},
        )
    )
    results = [upper, lower, reordered, padded]
    assert all(result == upper for result in results[1:])
    assert len({result.model_dump_json() for result in results}) == 1
    assert (upper.n_positive, upper.n_negative, upper.n_zero) == (1, 2, 0)


def test_inertia_retained_matrix_reconstructs_the_source() -> None:
    from jacobian.math.matrices.analysis._operations import _dense_fractions

    request = _inertia_request(3, {(0, 1): "2/3", (2, 2): "-5"})
    dense = _dense_fractions(compute_inertia(request).matrix)
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


@pytest.mark.parametrize(
    "dimension",
    range(MAX_SYMMETRIC_MATRIX_DIMENSION + 1, MAX_RATIONAL_MATRIX_ORDER + 1),
)
def test_inertia_request_rejects_dimensions_widened_by_the_canonical_value(
    dimension: int,
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        _inertia_request(dimension, {(0, 0): "1"})
    assert excinfo.value.errors()[0]["type"] == "less_than_equal"


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
        encoded = encode_strict_json(
            {
                "dimension": dimension,
                "entries": [
                    {
                        "row": r,
                        "col": c,
                        "value": {"num": "9" * digits[(r, c)], "den": "1"},
                    }
                    for (r, c) in cells
                ],
            }
        )
        assert len(encoded) <= limits.max_input_bytes
        return encoded

    return build(offset)


def test_inertia_request_admission_reserves_output_headroom_for_source_echo() -> None:
    from jacobian.canonical import CanonicalLimits

    encoded = _encoded_inertia_payload_near_limit(offset=512)
    assert len(encoded) <= CanonicalLimits().max_output_bytes
    with pytest.raises(ValidationError):
        SymmetricMatrixRequest.model_validate_json(encoded)


def test_inertia_request_admission_accepts_payload_inside_reserved_budget() -> None:
    encoded = _encoded_inertia_payload_near_limit(offset=2048)
    request = SymmetricMatrixRequest.model_validate_json(encoded)
    assert request.dimension == MAX_MATRIX_DIMENSION


def test_inertia_request_admission_rejects_echo_beyond_output_limit_as_typed_error() -> (
    None
):
    """A fitting sparse request whose dense echo exceeds the whole output
    budget must still be rejected as a typed error, not overflow encoding."""

    from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
    from jacobian.canonical import CanonicalLimits, encode_strict_json

    digits = "9" * (MAX_CANONICAL_RATIONAL_DIGITS // 3)
    payload = {
        "dimension": MAX_MATRIX_DIMENSION,
        "entries": [
            {"row": r, "col": c, "value": {"num": digits, "den": "1"}}
            for r in range(MAX_MATRIX_DIMENSION)
            for c in range(r, MAX_MATRIX_DIMENSION)
        ],
    }
    encoded = encode_strict_json(payload)
    assert len(encoded) <= CanonicalLimits().max_input_bytes
    with pytest.raises(ValidationError):
        SymmetricMatrixRequest.model_validate_json(encoded)
