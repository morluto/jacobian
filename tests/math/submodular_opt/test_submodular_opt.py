"""Tests for submodular optimization operations."""

import pytest

from jacobian.math.submodular_opt._models import (
    MonotonicityCheckRequest,
    SetFunction,
    SetFunctionEntry,
    SetFunctionEvalRequest,
    SubmodularityCheckRequest,
)
from jacobian.math.submodular_opt._operations import (
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


class TestKernelEquivalence:
    """Cross-check the local characterizations against brute-force oracles."""

    @staticmethod
    def _random_function(n: int, seed: int, fractional: bool = False) -> SetFunction:
        import random
        from fractions import Fraction

        generator = random.Random(seed)
        entries = []
        for mask in range(1 << n):
            subset = tuple(i for i in range(n) if mask & (1 << i))
            if fractional and generator.random() < 0.5:
                fraction = Fraction(generator.randint(-20, 20), generator.randint(1, 7))
            else:
                fraction = Fraction(generator.randint(-50, 50), 1)
            entries.append(
                SetFunctionEntry(
                    subset=subset,
                    value={
                        "num": str(fraction.numerator),
                        "den": str(fraction.denominator),
                    },
                )
            )
        return SetFunction(ground_set_size=n, entries=tuple(entries))

    @staticmethod
    def _bruteforce_monotone(function: SetFunction) -> bool:
        table = {
            tuple(sorted(entry.subset)): entry.value.as_fraction()
            for entry in function.entries
        }
        n = function.ground_set_size
        for mask in range(1 << n):
            for j in range(n):
                bit = 1 << j
                if mask & bit:
                    continue
                sup = tuple(i for i in range(n) if (mask | bit) & (1 << i))
                sub = tuple(i for i in range(n) if mask & (1 << i))
                if table[sub] > table[sup]:
                    return False
        return True

    @staticmethod
    def _bruteforce_submodular(function: SetFunction) -> bool:
        table = {
            tuple(sorted(entry.subset)): entry.value.as_fraction()
            for entry in function.entries
        }
        keys = list(table)
        n = function.ground_set_size
        masks = list(range(1 << n))

        def to_mask(subset):
            value = 0
            for element in subset:
                value |= 1 << element
            return value

        for left in range(len(keys)):
            s_set = set(keys[left])
            for right in range(left + 1, len(keys)):
                t_key = keys[right]
                t_set = set(t_key)
                union = tuple(sorted(s_set | t_set))
                inter = tuple(sorted(s_set & t_set))
                if table[keys[left]] + table[t_key] < table[union] + table[inter]:
                    return False
        del masks, n
        return True

    def test_local_matches_bruteforce_integer_tables(self):
        for seed in range(6):
            function = self._random_function(6, seed=seed)
            assert check_monotonicity(
                MonotonicityCheckRequest(function=function)
            ).is_monotone == self._bruteforce_monotone(function)
            assert check_submodularity(
                SubmodularityCheckRequest(function=function)
            ).is_submodular == self._bruteforce_submodular(function)

    def test_local_matches_bruteforce_fractional_tables(self):
        for seed in range(4):
            function = self._random_function(5, seed=100 + seed, fractional=True)
            assert check_submodularity(
                SubmodularityCheckRequest(function=function)
            ).is_submodular == self._bruteforce_submodular(function)

    def test_violation_message_names_witnesses(self):
        # f({}) = 0 but f({0}) = -1 violates monotonicity at a covering edge.
        entries = [
            SetFunctionEntry(subset=(), value={"num": "0", "den": "1"}),
            SetFunctionEntry(subset=(0,), value={"num": "-1", "den": "1"}),
        ]
        result = check_monotonicity(
            MonotonicityCheckRequest(
                function=SetFunction(ground_set_size=1, entries=tuple(entries))
            )
        )
        assert result.is_monotone is False
        assert result.violation == "f(()) > f((0,))"

    def test_transport_preflight_rejects_unwritable_tables(self):
        """A complete 2^16 table with wide values exceeds the byte envelope."""
        from pydantic import ValidationError

        n = 16
        entries = []
        for mask in range(1 << n):
            subset = tuple(i for i in range(n) if mask & (1 << i))
            entries.append(
                SetFunctionEntry(
                    subset=subset,
                    value={"num": "9" * 90, "den": "1"},
                )
            )
        with pytest.raises(ValidationError, match="transport envelope"):
            SetFunction(ground_set_size=16, entries=tuple(entries))


def test_value_height_bound_keeps_scan_work_small() -> None:
    """Scan requests reject 129-digit values so the ~8M-inequality scan stays
    on small big-ints; the shared entry type keeps admitting them so the
    single-lookup evaluator can return any exact representable height."""
    from pydantic import ValidationError

    wide_empty = SetFunctionEntry(subset=(), value={"num": "9" * 129, "den": "1"})
    wide_full = SetFunctionEntry(subset=(0,), value={"num": "9" * 129, "den": "1"})
    with pytest.raises(ValidationError, match="128-digit"):
        MonotonicityCheckRequest(
            function=SetFunction(ground_set_size=1, entries=(wide_empty, wide_full))
        )
    with pytest.raises(ValidationError, match="128-digit"):
        SubmodularityCheckRequest(
            function=SetFunction(ground_set_size=1, entries=(wide_empty, wide_full))
        )

    narrow = SetFunctionEntry(subset=(0,), value={"num": "9" * 128, "den": "1"})
    # Exactly-128-digit values are admitted and the scan completes normally.
    assert (
        check_monotonicity(
            MonotonicityCheckRequest(
                function=SetFunction(
                    ground_set_size=1,
                    entries=(
                        SetFunctionEntry(subset=(), value={"num": "0", "den": "1"}),
                        narrow,
                    ),
                )
            )
        ).is_monotone
        is True
    )
