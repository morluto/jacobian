"""Tests for multiple testing operations."""

from jacobian._exact import CanonicalRational
from jacobian.math.probability.multiple_testing._models import (
    BHStepUpRequest,
    FDPRequest,
    HypothesisSpec,
)
from jacobian.math.probability.multiple_testing.operations import (
    bh_step_up,
    false_discovery_proportion,
)


class TestBHStepUp:
    def test_all_rejected(self) -> None:
        req = BHStepUpRequest(
            hypotheses=(
                HypothesisSpec(
                    hypothesis_id="h1", p_value=CanonicalRational(num="1", den="100")
                ),
                HypothesisSpec(
                    hypothesis_id="h2", p_value=CanonicalRational(num="1", den="25")
                ),
                HypothesisSpec(
                    hypothesis_id="h3", p_value=CanonicalRational(num="3", den="100")
                ),
            ),
            level=CanonicalRational(num="1", den="20"),
        )
        result = bh_step_up(req.hypotheses, req.level)
        assert result.critical_index == 3
        assert len(result.rejected) == 3

    def test_partial_rejection(self) -> None:
        req = BHStepUpRequest(
            hypotheses=(
                HypothesisSpec(
                    hypothesis_id="h1", p_value=CanonicalRational(num="1", den="100")
                ),
                HypothesisSpec(
                    hypothesis_id="h2", p_value=CanonicalRational(num="2", den="25")
                ),
                HypothesisSpec(
                    hypothesis_id="h3", p_value=CanonicalRational(num="9", den="100")
                ),
            ),
            level=CanonicalRational(num="1", den="20"),
        )
        result = bh_step_up(req.hypotheses, req.level)
        assert result.critical_index == 1
        assert len(result.rejected) == 1


class TestFDP:
    def test_simple(self) -> None:
        req = FDPRequest(rejected_ids=("h1", "h2"), true_null_ids=("h2", "h3"))
        result = false_discovery_proportion(req.rejected_ids, req.true_null_ids)
        assert result.false_discoveries == 1
        assert result.total_rejections == 2
        assert result.fdp == "1/2"

    def test_no_rejections(self) -> None:
        req = FDPRequest(rejected_ids=(), true_null_ids=("h1",))
        result = false_discovery_proportion(req.rejected_ids, req.true_null_ids)
        assert result.fdp == "0"

    def test_rejects_out_of_range_p_value_and_level(self) -> None:
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as p_value_error:
            HypothesisSpec(
                hypothesis_id="h1", p_value=CanonicalRational(num="-1", den="1")
            )
        assert (
            p_value_error.value.errors()[0]["type"]
            == "multiple_testing.p_value_out_of_range"
        )
        with pytest.raises(ValidationError) as level_error:
            BHStepUpRequest(
                hypotheses=(
                    HypothesisSpec(
                        hypothesis_id="h1",
                        p_value=CanonicalRational(num="1", den="2"),
                    ),
                ),
                level=CanonicalRational(num="2", den="1"),
            )
        assert (
            level_error.value.errors()[0]["type"]
            == "multiple_testing.level_out_of_range"
        )
