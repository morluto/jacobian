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
