"""Tests for matrix analysis operations."""

import copy
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.matrices.analysis._models import (
    FarkasCertificateRequest,
    InertiaResult,
    MatrixEntry,
    SymmetricMatrixRequest,
)
from jacobian.math.matrices.analysis._operations import (
    check_farkas_certificate,
    compute_inertia,
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

        with pytest.raises(ValidationError, match="conflict"):
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

        with pytest.raises(ValidationError, match="rectangular"):
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


def test_inertia_result_rejects_mutations() -> None:
    request = _inertia_request(2, {(0, 0): "1"})
    result = compute_inertia(request)
    dumped = result.model_dump()

    count_sum = copy.deepcopy(dumped)
    count_sum["n_zero"] = 5
    with pytest.raises(ValidationError, match="sum to the matrix dimension"):
        InertiaResult.model_validate(count_sum)

    wrong_label = copy.deepcopy(dumped)
    wrong_label["definiteness"] = "indefinite"
    with pytest.raises(ValidationError, match="agree with the counts"):
        InertiaResult.model_validate(wrong_label)

    foreign_source = copy.deepcopy(dumped)
    foreign_source["matrix"]["entries"][0]["value"] = {"num": "-1", "den": "1"}
    with pytest.raises(ValidationError, match="Sylvester inertia"):
        InertiaResult.model_validate(foreign_source)

    forged_counts = copy.deepcopy(dumped)
    forged_counts["n_positive"] = 0
    forged_counts["n_negative"] = 1
    with pytest.raises(ValidationError, match="Sylvester inertia"):
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


def _encoded_inertia_payload_near_limit(offset: int) -> bytes:
    """Encode an inertia request whose echoed source matrix lands exactly
    ``offset`` bytes below the canonical output limit, so the payload fits
    the identical input limit while the result echo plus the reserved
    envelope may not."""

    import functools

    from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
    from jacobian.canonical import CanonicalLimits, encode_strict_json

    @functools.cache
    def build(offset: int) -> bytes:
        limits = CanonicalLimits()
        digits = "9" * MAX_CANONICAL_RATIONAL_DIGITS
        template = {"row": 0, "col": 0, "value": {"num": digits, "den": "1"}}
        base = len(encode_strict_json({"dimension": 50, "entries": []}))
        step = len(encode_strict_json({"dimension": 50, "entries": [template]})) - base
        target = limits.max_output_bytes - offset
        count = max(0, (target - base) // step)
        cells = [(r, c) for r in range(50) for c in range(r, 50)]
        assert count + 2 <= len(cells)
        fixed_entries = [
            {"row": r, "col": c, "value": {"num": digits, "den": "1"}}
            for r, c in cells[:count]
        ]
        trim_cells = cells[count : count + 2]
        trim_lengths = [1, 1]
        encoded = b""
        for _ in range(6):
            trimmed = [
                {
                    "row": r,
                    "col": c,
                    "value": {"num": "9" * length, "den": "1"},
                }
                for (r, c), length in zip(trim_cells, trim_lengths, strict=True)
            ]
            encoded = encode_strict_json(
                {"dimension": 50, "entries": fixed_entries + trimmed}
            )
            delta = target - len(encoded)
            if not delta:
                break
            first = trim_lengths[0] + delta
            if first < 1:
                trim_lengths[1] += first - 1
                first = 1
            elif first > MAX_CANONICAL_RATIONAL_DIGITS:
                trim_lengths[1] += first - MAX_CANONICAL_RATIONAL_DIGITS
                first = MAX_CANONICAL_RATIONAL_DIGITS
            assert 1 <= trim_lengths[1] <= MAX_CANONICAL_RATIONAL_DIGITS
            trim_lengths[0] = first
        assert len(encoded) == target
        return encoded

    return build(offset)


def test_inertia_request_admission_reserves_output_headroom_for_source_echo() -> None:
    from jacobian.canonical import CanonicalLimits

    encoded = _encoded_inertia_payload_near_limit(offset=512)
    assert len(encoded) <= CanonicalLimits().max_output_bytes
    with pytest.raises(ValidationError, match="canonical output limit"):
        SymmetricMatrixRequest.model_validate_json(encoded)


def test_inertia_request_admission_accepts_payload_inside_reserved_budget() -> None:
    encoded = _encoded_inertia_payload_near_limit(offset=2048)
    request = SymmetricMatrixRequest.model_validate_json(encoded)
    assert request.dimension == 50


def test_dispatch_rejects_unfittable_inertia_request_as_typed_error() -> None:
    import json

    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import OperationRequestValidationError, invoke_operation

    with pytest.raises(OperationRequestValidationError) as excinfo:
        invoke_operation(
            "matrix.inertia.compute",
            json.loads(_encoded_inertia_payload_near_limit(offset=512)),
            Catalog.open(),
        )
    assert "canonical output limit" in str(excinfo.value.cause)


def test_large_fitting_inertia_request_returns_typed_result() -> None:
    from jacobian.canonical import canonicalize_json
    from jacobian.catalog.catalog import Catalog
    from jacobian.dispatch import invoke_operation

    digits = "9" * 4096
    payload = {
        "dimension": 50,
        "entries": [
            {"row": r, "col": r, "value": {"num": digits, "den": "1"}}
            for r in range(50)
        ],
    }
    assert len(canonicalize_json(payload)) > 100_000
    result = invoke_operation("matrix.inertia.compute", payload, Catalog.open())
    assert result.output["n_positive"] == 50
    assert result.output["n_negative"] == 0
    assert result.output["n_zero"] == 0
    assert result.output["definiteness"] == "positive_definite"
