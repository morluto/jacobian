"""Tests for discrepancy theory operations."""

from jacobian.math.discrepancy_theory._models import (
    DiscrepancyEvalRequest,
    DiscrepancyOptimumRequest,
    FiniteSetSystem,
)
from jacobian.math.discrepancy_theory._operations import (
    compute_discrepancy,
    compute_optimal_discrepancy,
)


class TestDiscrepancyEval:
    def test_simple_two_element(self):
        req = DiscrepancyEvalRequest(
            set_system=FiniteSetSystem(ground_set_size=2, sets=((0,), (1,))),
            coloring=(1, -1),
        )
        result = compute_discrepancy(req)
        assert result.signed_sums == (1, -1)
        assert result.max_absolute_imbalance == 1

    def test_empty_family(self):
        req = DiscrepancyEvalRequest(
            set_system=FiniteSetSystem(ground_set_size=3, sets=()),
            coloring=(1, 1, 1),
        )
        result = compute_discrepancy(req)
        assert result.signed_sums == ()
        assert result.max_absolute_imbalance == 0

    def test_balanced_coloring(self):
        req = DiscrepancyEvalRequest(
            set_system=FiniteSetSystem(ground_set_size=4, sets=((0, 1, 2, 3),)),
            coloring=(1, 1, -1, -1),
        )
        result = compute_discrepancy(req)
        assert result.signed_sums == (0,)
        assert result.max_absolute_imbalance == 0


class TestDiscrepancyOptimum:
    def test_triangle_system(self):
        req = DiscrepancyOptimumRequest(
            set_system=FiniteSetSystem(
                ground_set_size=3,
                sets=((0, 1), (1, 2), (0, 2)),
            ),
        )
        result = compute_optimal_discrepancy(req)
        assert result.optimal_discrepancy == 2
        assert result.exhaustive is True

    def test_empty_ground_set(self):
        req = DiscrepancyOptimumRequest(
            set_system=FiniteSetSystem(ground_set_size=0, sets=()),
        )
        result = compute_optimal_discrepancy(req)
        assert result.optimal_discrepancy == 0
        assert result.optimal_coloring == ()

    def test_single_set_optimum(self):
        req = DiscrepancyOptimumRequest(
            set_system=FiniteSetSystem(
                ground_set_size=2,
                sets=((0, 1),),
            ),
        )
        result = compute_optimal_discrepancy(req)
        assert result.optimal_discrepancy == 0
