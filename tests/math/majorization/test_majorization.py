"""Tests for majorization and matrix mixing operations."""

from jacobian._exact import CanonicalRational
from jacobian.math.majorization._models import (
    BirkhoffDecompositionRequest,
    DoublyStochasticCheckRequest,
    MajorizationCheckRequest,
    RationalMatrix,
    RationalVector,
    SchurHornCheckRequest,
    TTransformSequenceRequest,
    WeakMajorizationCheckRequest,
)
from jacobian.math.majorization._operations import (
    compute_birkhoff_decomposition,
    compute_doubly_stochastic_check,
    compute_majorization_check,
    compute_schur_horn_check,
    compute_t_transform_sequence,
    compute_weak_majorization_check,
)


def cr(n: str, d: str = "1") -> CanonicalRational:
    return CanonicalRational(num=n, den=d)


def rv(labels: list[str], vals: list[tuple[str, str]]) -> RationalVector:
    return RationalVector(
        labels=tuple(labels),
        values=tuple(CanonicalRational(num=n, den=d) for n, d in vals),
    )


class TestMajorizationCheck:
    def test_majorizes(self) -> None:
        """(3,1) majorizes (2,2): 3>=2, 3+1=2+2."""
        result = compute_majorization_check(
            MajorizationCheckRequest(
                x=rv(["a", "b"], [("3", "1"), ("1", "1")]),
                y=rv(["a", "b"], [("2", "1"), ("2", "1")]),
            )
        )
        assert result.majorizes is True
        assert result.total_sum_match is True
        assert result.first_failed_prefix is None

    def test_not_majorizes_sum_mismatch(self) -> None:
        """(3,1) does not majorize (3,2): sums don't match."""
        result = compute_majorization_check(
            MajorizationCheckRequest(
                x=rv(["a", "b"], [("3", "1"), ("1", "1")]),
                y=rv(["a", "b"], [("3", "1"), ("2", "1")]),
            )
        )
        assert result.majorizes is False
        assert result.total_sum_match is False

    def test_not_majorizes_prefix_fail(self) -> None:
        """(2,2) does not majorize (3,1): prefix sum 2 < 3."""
        result = compute_majorization_check(
            MajorizationCheckRequest(
                x=rv(["a", "b"], [("2", "1"), ("2", "1")]),
                y=rv(["a", "b"], [("3", "1"), ("1", "1")]),
            )
        )
        assert result.majorizes is False
        assert result.first_failed_prefix is not None

    def test_equal_vectors_majorize(self) -> None:
        """A vector majorizes itself."""
        result = compute_majorization_check(
            MajorizationCheckRequest(
                x=rv(["a", "b"], [("5", "1"), ("3", "1")]),
                y=rv(["a", "b"], [("5", "1"), ("3", "1")]),
            )
        )
        assert result.majorizes is True

    def test_three_dim(self) -> None:
        """(4,2,0) majorizes (2,2,2): 4>=2, 6>=4, 6=6."""
        result = compute_majorization_check(
            MajorizationCheckRequest(
                x=rv(["a", "b", "c"], [("4", "1"), ("2", "1"), ("0", "1")]),
                y=rv(["a", "b", "c"], [("2", "1"), ("2", "1"), ("2", "1")]),
            )
        )
        assert result.majorizes is True


class TestWeakMajorization:
    def test_weak_sub_holds(self) -> None:
        """Weak submajorization holds when x >= y in prefix sums."""
        result = compute_weak_majorization_check(
            WeakMajorizationCheckRequest(
                x=rv(["a", "b"], [("4", "1"), ("1", "1")]),
                y=rv(["a", "b"], [("2", "1"), ("2", "1")]),
                direction="sub",
            )
        )
        assert result.holds is True

    def test_weak_sub_fails(self) -> None:
        """Weak submajorization fails when prefix sum is negative."""
        result = compute_weak_majorization_check(
            WeakMajorizationCheckRequest(
                x=rv(["a", "b"], [("1", "1"), ("1", "1")]),
                y=rv(["a", "b"], [("3", "1"), ("0", "1")]),
                direction="sub",
            )
        )
        assert result.holds is False

    def test_weak_super(self) -> None:
        """Weak supermajorization."""
        result = compute_weak_majorization_check(
            WeakMajorizationCheckRequest(
                x=rv(["a", "b"], [("1", "1"), ("3", "1")]),
                y=rv(["a", "b"], [("2", "1"), ("2", "1")]),
                direction="super",
            )
        )
        assert result.holds is True


class TestDoublyStochastic:
    def test_identity(self) -> None:
        result = compute_doubly_stochastic_check(
            DoublyStochasticCheckRequest(
                matrix=RationalMatrix(
                    row_labels=["a", "b"],
                    col_labels=["a", "b"],
                    entries=(
                        (cr("1", "1"), cr("0", "1")),
                        (cr("0", "1"), cr("1", "1")),
                    ),
                )
            )
        )
        assert result.is_doubly_stochastic is True

    def test_not_ds(self) -> None:
        result = compute_doubly_stochastic_check(
            DoublyStochasticCheckRequest(
                matrix=RationalMatrix(
                    row_labels=["a", "b"],
                    col_labels=["a", "b"],
                    entries=(
                        (cr("2", "1"), cr("0", "1")),
                        (cr("0", "1"), cr("1", "1")),
                    ),
                )
            )
        )
        assert result.is_doubly_stochastic is False
        assert result.first_bad_row is not None

    def test_negative_entry(self) -> None:
        result = compute_doubly_stochastic_check(
            DoublyStochasticCheckRequest(
                matrix=RationalMatrix(
                    row_labels=["a", "b"],
                    col_labels=["a", "b"],
                    entries=(
                        (cr("-1", "1"), cr("2", "1")),
                        (cr("2", "1"), cr("-1", "1")),
                    ),
                )
            )
        )
        assert result.is_doubly_stochastic is False
        assert result.first_negative_entry is not None


class TestBirkhoff:
    def test_permutation_matrix(self) -> None:
        """A permutation matrix decomposes into one term."""
        result = compute_birkhoff_decomposition(
            BirkhoffDecompositionRequest(
                matrix=RationalMatrix(
                    row_labels=["a", "b"],
                    col_labels=["a", "b"],
                    entries=(
                        (cr("0", "1"), cr("1", "1")),
                        (cr("1", "1"), cr("0", "1")),
                    ),
                )
            )
        )
        assert len(result.terms) == 1
        assert result.terms[0].weight.as_fraction() == 1

    def test_average_matrix(self) -> None:
        """The 2x2 averaging matrix decomposes into 2 terms of weight 1/2."""
        result = compute_birkhoff_decomposition(
            BirkhoffDecompositionRequest(
                matrix=RationalMatrix(
                    row_labels=["a", "b"],
                    col_labels=["a", "b"],
                    entries=(
                        (cr("1", "2"), cr("1", "2")),
                        (cr("1", "2"), cr("1", "2")),
                    ),
                )
            )
        )
        assert len(result.terms) == 2
        weights_sum = sum(t.weight.as_fraction() for t in result.terms)
        assert weights_sum == 1


class TestSchurHorn:
    def test_feasible(self) -> None:
        """Diagonal (1, 0) is feasible for eigenvalues (2, -1)."""
        result = compute_schur_horn_check(
            SchurHornCheckRequest(
                eigenvalues=(cr("2", "1"), cr("-1", "1")),
                diagonal=(cr("1", "1"), cr("0", "1")),
            )
        )
        assert result.feasible is True

    def test_infeasible(self) -> None:
        """Diagonal (2, 0) is not feasible for eigenvalues (1, 1)."""
        result = compute_schur_horn_check(
            SchurHornCheckRequest(
                eigenvalues=(cr("1", "1"), cr("1", "1")),
                diagonal=(cr("2", "1"), cr("0", "1")),
            )
        )
        assert result.feasible is False

    def test_sum_mismatch(self) -> None:
        """Sum mismatch makes it infeasible."""
        result = compute_schur_horn_check(
            SchurHornCheckRequest(
                eigenvalues=(cr("3", "1"), cr("1", "1")),
                diagonal=(cr("2", "1"), cr("1", "1")),
            )
        )
        assert result.feasible is False
        assert result.total_sum_match is False


class TestTTransform:
    def test_majorizes_t_transform(self) -> None:
        """When x majorizes y, compute the T-transform sequence."""
        result = compute_t_transform_sequence(
            TTransformSequenceRequest(
                x=rv(["a", "b"], [("4", "1"), ("0", "1")]),
                y=rv(["a", "b"], [("2", "1"), ("2", "1")]),
            )
        )
        assert result.majorizes is True
        assert len(result.steps) > 0

    def test_not_majorizes(self) -> None:
        """When x does not majorize y, return empty result."""
        result = compute_t_transform_sequence(
            TTransformSequenceRequest(
                x=rv(["a", "b"], [("1", "1"), ("1", "1")]),
                y=rv(["a", "b"], [("3", "1"), ("0", "1")]),
            )
        )
        assert result.majorizes is False
        assert len(result.steps) == 0
