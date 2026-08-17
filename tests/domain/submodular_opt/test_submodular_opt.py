"""Tests for submodular optimization operations."""

from jacobian.contracts.submodular_opt import (
    MonotonicityCheckRequest,
    SetFunction,
    SetFunctionEntry,
    SetFunctionEvalRequest,
    SubmodularityCheckRequest,
)
from jacobian.domains.submodular_opt.operations import (
    check_monotonicity,
    check_submodularity,
    evaluate_set_function,
)


def _make_uniform_function(n: int) -> SetFunction:
    """f(S) = |S| for ground set {0, ..., n-1}."""
    entries = []
    for mask in range(1 << n):
        subset = tuple(i for i in range(n) if mask & (1 << i))
        entries.append(
            SetFunctionEntry(subset=subset, value={"num": str(len(subset)), "den": "1"})
        )
    return SetFunction(ground_set_size=n, entries=tuple(entries))


class TestSetFunctionEval:
    def test_simple(self):
        fn = _make_uniform_function(2)
        req = SetFunctionEvalRequest(function=fn, subset=(0, 1))
        result = evaluate_set_function(req)
        assert result.found is True
        assert result.value == "2"

    def test_rejects_duplicate_eval_subset(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="unique"):
            SetFunctionEvalRequest(function=_make_uniform_function(1), subset=(0, 0))


class TestMonotonicity:
    def test_monotone(self):
        fn = _make_uniform_function(2)
        req = MonotonicityCheckRequest(function=fn)
        result = check_monotonicity(req)
        assert result.is_monotone is True

    def test_rejects_incomplete_table(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="exactly one value per subset"):
            SetFunction(
                ground_set_size=2,
                entries=(
                    SetFunctionEntry(subset=(), value={"num": "0", "den": "1"}),
                    SetFunctionEntry(
                        subset=(0, 1),
                        value={"num": "-1", "den": "1"},
                    ),
                ),
            )


class TestSubmodularity:
    def test_modular_is_submodular(self):
        """f(S) = |S| is modular (hence submodular)."""
        fn = _make_uniform_function(2)
        req = SubmodularityCheckRequest(function=fn)
        result = check_submodularity(req)
        assert result.is_submodular is True
