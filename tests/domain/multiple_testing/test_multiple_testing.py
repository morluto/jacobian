"""Tests for multiple testing operations."""

from jacobian.contracts.multiple_testing import (
    BHStepUpRequest,
    FDPRequest,
    HypothesisSpec,
)
from jacobian.domains.multiple_testing.operations import (
    compute_bh_step_up,
    compute_fdp,
)


class TestBHStepUp:
    def test_all_rejected(self):
        req = BHStepUpRequest(
            hypotheses=(
                HypothesisSpec(hypothesis_id="h1", p_value={"num": "1", "den": "100"}),
                HypothesisSpec(hypothesis_id="h2", p_value={"num": "1", "den": "25"}),
                HypothesisSpec(hypothesis_id="h3", p_value={"num": "3", "den": "100"}),
            ),
            level={"num": "1", "den": "20"},
        )
        result = compute_bh_step_up(req)
        assert result.critical_index == 3
        assert len(result.rejected) == 3

    def test_partial_rejection(self):
        req = BHStepUpRequest(
            hypotheses=(
                HypothesisSpec(hypothesis_id="h1", p_value={"num": "1", "den": "100"}),
                HypothesisSpec(hypothesis_id="h2", p_value={"num": "2", "den": "25"}),
                HypothesisSpec(hypothesis_id="h3", p_value={"num": "9", "den": "100"}),
            ),
            level={"num": "1", "den": "20"},
        )
        result = compute_bh_step_up(req)
        assert result.critical_index == 1
        assert len(result.rejected) == 1


class TestFDP:
    def test_simple(self):
        req = FDPRequest(rejected_ids=("h1", "h2"), true_null_ids=("h2", "h3"))
        result = compute_fdp(req)
        assert result.false_discoveries == 1
        assert result.total_rejections == 2
        assert result.fdp == "1/2"

    def test_no_rejections(self):
        req = FDPRequest(rejected_ids=(), true_null_ids=("h1",))
        result = compute_fdp(req)
        assert result.fdp == "0"

    def test_rejects_out_of_range_p_value_and_level(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="p-value"):
            HypothesisSpec(hypothesis_id="h1", p_value={"num": "-1", "den": "1"})
        with pytest.raises(ValidationError, match="level"):
            BHStepUpRequest(
                hypotheses=(
                    HypothesisSpec(
                        hypothesis_id="h1",
                        p_value={"num": "1", "den": "2"},
                    ),
                ),
                level={"num": "2", "den": "1"},
            )
