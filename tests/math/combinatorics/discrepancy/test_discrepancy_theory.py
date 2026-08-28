"""Tests for discrepancy theory operations."""

from __future__ import annotations

import itertools
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction

import pytest
from pydantic import ValidationError

import jacobian.math.combinatorics.discrepancy._models as discrepancy_models
import jacobian.math.combinatorics.discrepancy._operations as discrepancy_operations
import jacobian.math.combinatorics.discrepancy._optimum_process as optimum_process
from jacobian.math.combinatorics.discrepancy._models import (
    MAX_COLUMN_INCIDENCES,
    MAX_MONITORED_COLUMNS,
    MAX_ROUNDING_COORDINATES,
    MAX_ROUNDING_INTERMEDIATE_DIGITS,
    MAX_ROUNDING_RATIONAL_DIGITS,
    MAX_ROUNDING_RESULT_RATIONAL_DIGITS,
    MAX_ROUNDING_WORK,
    DiscrepancyEvalRequest,
    DiscrepancyOptimumRequest,
    DiscrepancyOptimumResult,
    FiniteSetSystem,
    HardConstraintRoundingRequest,
    HardConstraintRoundingResult,
)
from jacobian.math.combinatorics.discrepancy._operations import (
    compute_discrepancy,
    compute_hard_constraint_rounding,
    compute_optimal_discrepancy,
)
from jacobian.process import BoundedProcessResult


@contextmanager
def _validation_code(code: str) -> Iterator[None]:
    with pytest.raises(ValidationError) as caught:
        yield
    assert caught.value.errors()[0]["type"] == code


def _rational(value: Fraction | int) -> dict[str, str]:
    fraction = Fraction(value)
    return {"num": str(fraction.numerator), "den": str(fraction.denominator)}


def _rounding_request(
    *,
    values: tuple[Fraction | int, ...] = (
        Fraction(1, 2),
        Fraction(1, 2),
        Fraction(1, 2),
        Fraction(1, 2),
    ),
    rows: tuple[object, ...] = (
        {"label": "left", "coordinates": (0, 1)},
        {"label": "right", "coordinates": (2, 3)},
    ),
    columns: tuple[object, ...] = (
        {"label": "diagonal", "coordinates": (0, 2)},
        {"label": "off_diagonal", "coordinates": (1, 3)},
    ),
    coordinate_labels: tuple[str, ...] | None = None,
) -> HardConstraintRoundingRequest:
    labels = coordinate_labels or tuple(f"a{index}" for index in range(len(values)))
    return HardConstraintRoundingRequest.model_validate(
        {
            "source": {
                "coordinate_labels": labels,
                "values": tuple(_rational(value) for value in values),
                "rows": rows,
                "columns": columns,
            }
        }
    )


class TestHardConstraintRounding:
    def test_known_answer_preserves_rows_and_returns_complete_ledgers(self) -> None:
        request = _rounding_request()

        first = compute_hard_constraint_rounding(request)
        second = compute_hard_constraint_rounding(request)

        assert first == second
        assert first.rounded_values == (1, 0, 1, 0)
        assert tuple(item.rounded_sum for item in first.row_ledger) == (1, 1)
        assert first.maximum_column_incidence == 1
        assert first.column_error_bound == 4
        assert tuple(
            item.signed_error.as_fraction() for item in first.column_ledger
        ) == (
            Fraction(1),
            Fraction(-1),
        )
        assert HardConstraintRoundingResult.model_validate(first.model_dump()) == first

    def test_large_active_column_distinguishes_row_only_rounding(self) -> None:
        request = _rounding_request(
            values=(Fraction(1, 2),) * 20,
            rows=({"label": "quota", "coordinates": tuple(range(20))},),
            columns=({"label": "first_half", "coordinates": tuple(range(10))},),
        )

        row_only_rounding = (1,) * 10 + (0,) * 10
        row_only_error = abs(
            sum(row_only_rounding[:10])
            - sum(value.as_fraction() for value in request.source.values[:10])
        )
        assert sum(row_only_rounding) == 10
        assert row_only_error == 5
        assert row_only_error > 4

        result = compute_hard_constraint_rounding(request)

        assert result.row_ledger[0].rounded_sum == 10
        assert result.column_ledger[0].source_sum.as_fraction() == 5
        assert result.column_ledger[0].signed_error.as_fraction() == 0
        assert result.column_ledger[0].absolute_error.as_fraction() <= 4

    def test_integral_singletons_empty_columns_and_unmonitored_coordinates(
        self,
    ) -> None:
        request = _rounding_request(
            values=(0, 1, 0),
            rows=(
                {"label": "zero", "coordinates": (0,)},
                {"label": "one", "coordinates": (1,)},
                {"label": "other", "coordinates": (2,)},
            ),
            columns=(),
        )

        result = compute_hard_constraint_rounding(request)

        assert result.rounded_values == (0, 1, 0)
        assert result.maximum_column_incidence == 0
        assert result.column_error_bound == 0
        assert result.column_ledger == ()

    def test_duplicate_indexed_columns_remain_distinct_ledger_entries(self) -> None:
        request = _rounding_request(
            columns=(
                {"label": "first", "coordinates": (0, 2)},
                {"label": "duplicate", "coordinates": (0, 2)},
            )
        )

        result = compute_hard_constraint_rounding(request)

        assert tuple(item.column_label for item in result.column_ledger) == (
            "first",
            "duplicate",
        )
        assert (
            result.column_ledger[0].signed_error == result.column_ledger[1].signed_error
        )
        assert result.maximum_column_incidence == 2

    def test_coherent_axis_and_family_relabeling_preserves_guarantees(self) -> None:
        original = compute_hard_constraint_rounding(_rounding_request())
        relabeled = compute_hard_constraint_rounding(
            _rounding_request(
                coordinate_labels=("w", "x", "y", "z"),
                rows=(
                    {"label": "r0", "coordinates": (0, 1)},
                    {"label": "r1", "coordinates": (2, 3)},
                ),
                columns=(
                    {"label": "c0", "coordinates": (0, 2)},
                    {"label": "c1", "coordinates": (1, 3)},
                ),
            )
        )

        assert relabeled.rounded_values == original.rounded_values
        assert tuple(item.source_sum for item in relabeled.row_ledger) == tuple(
            item.source_sum for item in original.row_ledger
        )
        assert tuple(item.signed_error for item in relabeled.column_ledger) == tuple(
            item.signed_error for item in original.column_ledger
        )

    @pytest.mark.parametrize(
        ("field", "replacement", "message"),
        [
            (
                "values",
                (_rational(Fraction(-1, 2)),) * 4,
                "discrepancy_theory.source_values_out_of_range",
            ),
            (
                "values",
                (_rational(Fraction(1, 3)),) + (_rational(Fraction(1, 2)),) * 3,
                "discrepancy_theory.row_sum_not_integral",
            ),
            (
                "rows",
                ({"label": "only", "coordinates": (0, 1)},),
                "discrepancy_theory.rows_not_a_partition",
            ),
            (
                "rows",
                (
                    {"label": "left", "coordinates": (0, 1)},
                    {"label": "right", "coordinates": (0, 1, 2, 3)},
                ),
                "discrepancy_theory.rows_not_a_partition",
            ),
            (
                "columns",
                ({"label": "bad", "coordinates": (2, 0)},),
                "discrepancy_theory.coordinate_indices_not_strictly_increasing",
            ),
            (
                "columns",
                ({"label": "bad", "coordinates": (4,)},),
                "discrepancy_theory.coordinate_indices_out_of_range",
            ),
        ],
    )
    def test_malformed_sources_are_rejected_before_the_kernel(
        self, field: str, replacement: object, message: str
    ) -> None:
        payload = _rounding_request().model_dump()
        payload["source"][field] = replacement
        with _validation_code(message):
            HardConstraintRoundingRequest.model_validate(payload)

    @pytest.mark.parametrize("mutation", ["bit", "row", "column", "source"])
    def test_result_parsing_checks_shape_without_replaying_source_claims(
        self, mutation: str
    ) -> None:
        payload = compute_hard_constraint_rounding(_rounding_request()).model_dump()
        if mutation == "bit":
            payload["rounded_values"] = (0, 0, 1, 0)
        elif mutation == "row":
            payload["row_ledger"][0]["rounded_sum"] = 0
        elif mutation == "column":
            payload["column_ledger"][0]["signed_error"] = _rational(0)
        else:
            payload["source"]["values"] = (
                _rational(Fraction(1, 3)),
                _rational(Fraction(2, 3)),
                *payload["source"]["values"][2:],
            )
        parsed = HardConstraintRoundingResult.model_validate(payload)
        assert parsed.source.coordinate_labels == ("a0", "a1", "a2", "a3")

    def test_exhaustive_small_half_integral_sources_satisfy_defining_invariants(
        self,
    ) -> None:
        choices = (Fraction(0), Fraction(1, 2), Fraction(1))
        for coordinate_count in range(6):
            columns = tuple(
                {
                    "label": f"c{mask}",
                    "coordinates": tuple(
                        index
                        for index in range(coordinate_count)
                        if mask & (1 << index)
                    ),
                }
                for mask in range(1 << coordinate_count)
            )
            rows = (
                ({"label": "all", "coordinates": tuple(range(coordinate_count))},)
                if coordinate_count
                else ()
            )
            for values in itertools.product(choices, repeat=coordinate_count):
                if sum(values).denominator != 1:
                    continue
                result = compute_hard_constraint_rounding(
                    _rounding_request(values=values, rows=rows, columns=columns)
                )
                assert all(value in (0, 1) for value in result.rounded_values)
                if coordinate_count:
                    assert result.row_ledger[0].rounded_sum == sum(values)
                else:
                    assert result.row_ledger == ()
                assert all(
                    item.absolute_error.as_fraction() <= result.column_error_bound
                    for item in result.column_ledger
                )

    def test_coordinate_column_and_incidence_boundaries(self) -> None:
        coordinate_values = (0,) * MAX_ROUNDING_COORDINATES
        _rounding_request(
            values=coordinate_values,
            rows=(
                {
                    "label": "all",
                    "coordinates": tuple(range(MAX_ROUNDING_COORDINATES)),
                },
            ),
            columns=(),
        )
        with _validation_code("too_long"):
            _rounding_request(
                values=(*coordinate_values, 0),
                rows=(
                    {
                        "label": "all",
                        "coordinates": tuple(range(MAX_ROUNDING_COORDINATES + 1)),
                    },
                ),
                columns=(),
            )

        empty_columns = tuple(
            {"label": f"c{index}", "coordinates": ()}
            for index in range(MAX_MONITORED_COLUMNS)
        )
        _rounding_request(
            values=(0,),
            rows=({"label": "r", "coordinates": (0,)},),
            columns=empty_columns,
        )
        with _validation_code("too_long"):
            _rounding_request(
                values=(0,),
                rows=({"label": "r", "coordinates": (0,)},),
                columns=(*empty_columns, {"label": "above", "coordinates": ()}),
            )

        support = tuple(range(MAX_COLUMN_INCIDENCES // MAX_MONITORED_COLUMNS))
        incidence_columns = tuple(
            {"label": f"i{index}", "coordinates": support}
            for index in range(MAX_MONITORED_COLUMNS)
        )
        values = (0,) * (len(support) + 1)
        rows = ({"label": "r", "coordinates": tuple(range(len(values)))},)
        _rounding_request(values=values, rows=rows, columns=incidence_columns)
        above = list(incidence_columns)
        above[-1] = {"label": "i511", "coordinates": (*support, len(support))}
        with _validation_code("discrepancy_theory.column_incidences_over_budget"):
            _rounding_request(values=values, rows=rows, columns=tuple(above))

    def test_over_incidence_rejects_before_exact_aggregation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail_if_aggregated(
            _values: list[Fraction], _coordinates: tuple[int, ...]
        ) -> Fraction:
            raise AssertionError("exact aggregation reached before cheap preflight")

        monkeypatch.setattr(
            discrepancy_models,
            "_sum_selected_fractions",
            fail_if_aggregated,
        )
        denominator = 10 ** (MAX_ROUNDING_RATIONAL_DIGITS - 1) + 1
        coordinate_count = MAX_ROUNDING_COORDINATES
        support = tuple(range(coordinate_count))
        column_count = MAX_COLUMN_INCIDENCES // coordinate_count + 1
        columns = tuple(
            {"label": f"c{index}", "coordinates": support}
            for index in range(column_count)
        )

        with _validation_code("discrepancy_theory.column_incidences_over_budget"):
            _rounding_request(
                values=(
                    Fraction(1, denominator),
                    Fraction(denominator - 1, denominator),
                )
                * (coordinate_count // 2),
                rows=({"label": "all", "coordinates": support},),
                columns=columns,
            )

    def test_rational_digit_and_work_boundaries(self) -> None:
        denominator = 10 ** (MAX_ROUNDING_RATIONAL_DIGITS - 1) + 1
        _rounding_request(
            values=(Fraction(1, denominator), Fraction(denominator - 1, denominator)),
            rows=({"label": "r", "coordinates": (0, 1)},),
            columns=(),
        )
        above_denominator = 10**MAX_ROUNDING_RATIONAL_DIGITS + 1
        with _validation_code("value_error"):
            _rounding_request(
                values=(
                    Fraction(1, above_denominator),
                    Fraction(above_denominator - 1, above_denominator),
                ),
                rows=({"label": "r", "coordinates": (0, 1)},),
                columns=(),
            )

        accepted_fractional = max(
            count
            for count in range(2, MAX_ROUNDING_COORDINATES + 1, 2)
            if (count * (count + 1) // 2) ** 2 + count**2 <= MAX_ROUNDING_WORK
        )
        _rounding_request(
            values=(Fraction(1, 2),) * accepted_fractional,
            rows=({"label": "r", "coordinates": tuple(range(accepted_fractional))},),
            columns=(),
        )
        rejected_fractional = accepted_fractional + 2
        with _validation_code("discrepancy_theory.rounding_work_over_budget"):
            _rounding_request(
                values=(Fraction(1, 2),) * rejected_fractional,
                rows=(
                    {
                        "label": "r",
                        "coordinates": tuple(range(rejected_fractional)),
                    },
                ),
                columns=(),
            )

    def test_intermediate_height_boundary(self) -> None:
        fractional_count = 74
        direction_growth = sum(
            support * len(str(support)) + 1
            for support in range(2, fractional_count + 1)
        )
        required_denominator_digits = (
            MAX_ROUNDING_INTERMEDIATE_DIGITS - direction_growth - 1
        )
        pair_extra_digits = required_denominator_digits // 2 - fractional_count // 2
        denominator_lengths = [1] * (fractional_count // 2)
        for index in range(pair_extra_digits):
            denominator_lengths[index % len(denominator_lengths)] += 1

        def source(
            lengths: list[int],
        ) -> tuple[tuple[Fraction, ...], tuple[dict[str, object], ...]]:
            values: list[Fraction] = []
            rows: list[dict[str, object]] = []
            for index, length in enumerate(lengths):
                denominator = 10 ** (length - 1) + 1
                values.extend(
                    (
                        Fraction(1, denominator),
                        Fraction(denominator - 1, denominator),
                    )
                )
                rows.append(
                    {"label": f"r{index}", "coordinates": (2 * index, 2 * index + 1)}
                )
            return tuple(values), tuple(rows)

        values, rows = source(denominator_lengths)
        _rounding_request(values=values, rows=rows, columns=())
        denominator_lengths[-1] += 1
        values, rows = source(denominator_lengths)
        with _validation_code(
            "discrepancy_theory.intermediate_rational_height_over_budget"
        ):
            _rounding_request(values=values, rows=rows, columns=())

    def test_exact_result_size_boundary(self) -> None:
        denominator = 10 ** (MAX_ROUNDING_RATIONAL_DIGITS - 1) + 1
        values = (Fraction(1, denominator), Fraction(denominator - 1, denominator))
        base_digits = (
            sum(
                len(str(abs(value.numerator))) + len(str(value.denominator))
                for value in values
            )
            + 2
        )
        per_column_digits = 3 * (
            len(str(1)) + len(str(denominator)) + len(str(len(values))) + 1
        )
        accepted_columns = (
            MAX_ROUNDING_RESULT_RATIONAL_DIGITS - base_digits
        ) // per_column_digits

        def columns(count: int) -> tuple[dict[str, object], ...]:
            return tuple(
                {"label": f"c{index}", "coordinates": (0,)} for index in range(count)
            )

        _rounding_request(
            values=values,
            rows=({"label": "r", "coordinates": (0, 1)},),
            columns=columns(accepted_columns),
        )
        with _validation_code("discrepancy_theory.result_rational_height_over_budget"):
            _rounding_request(
                values=values,
                rows=({"label": "r", "coordinates": (0, 1)},),
                columns=columns(accepted_columns + 1),
            )


class TestDiscrepancyEval:
    def test_simple_two_element(self) -> None:
        req = DiscrepancyEvalRequest(
            set_system=FiniteSetSystem(ground_set_size=2, sets=((0,), (1,))),
            coloring=(1, -1),
        )
        result = compute_discrepancy(req)
        assert result.signed_sums == (1, -1)
        assert result.max_absolute_imbalance == 1

    def test_empty_family(self) -> None:
        req = DiscrepancyEvalRequest(
            set_system=FiniteSetSystem(ground_set_size=3, sets=()),
            coloring=(1, 1, 1),
        )
        result = compute_discrepancy(req)
        assert result.signed_sums == ()
        assert result.max_absolute_imbalance == 0

    def test_balanced_coloring(self) -> None:
        req = DiscrepancyEvalRequest(
            set_system=FiniteSetSystem(ground_set_size=4, sets=((0, 1, 2, 3),)),
            coloring=(1, 1, -1, -1),
        )
        result = compute_discrepancy(req)
        assert result.signed_sums == (0,)
        assert result.max_absolute_imbalance == 0


class TestDiscrepancyOptimum:
    def test_triangle_system(self) -> None:
        req = DiscrepancyOptimumRequest(
            set_system=FiniteSetSystem(
                ground_set_size=3,
                sets=((0, 1), (1, 2), (0, 2)),
            ),
        )
        result = compute_optimal_discrepancy(req)
        assert result.status == "OPTIMAL"
        assert result.optimal_discrepancy == 2
        assert DiscrepancyOptimumResult.model_validate(result.model_dump()) == result

    def test_empty_ground_set(self) -> None:
        req = DiscrepancyOptimumRequest(
            set_system=FiniteSetSystem(ground_set_size=0, sets=()),
        )
        result = compute_optimal_discrepancy(req)
        assert result.status == "OPTIMAL"
        assert result.optimal_discrepancy == 0
        assert result.optimal_coloring == ()

    def test_single_set_optimum(self) -> None:
        req = DiscrepancyOptimumRequest(
            set_system=FiniteSetSystem(
                ground_set_size=2,
                sets=((0, 1),),
            ),
        )
        result = compute_optimal_discrepancy(req)
        assert result.status == "OPTIMAL"
        assert result.optimal_discrepancy == 0

    def test_matches_bruteforce_on_small_instances(self) -> None:
        import itertools
        import random

        generator = random.Random(20260824)
        for _ in range(12):
            n = generator.randint(1, 10)
            set_count = generator.randint(0, 8)
            sets = tuple(
                tuple(sorted(generator.sample(range(n), generator.randint(1, n))))
                for _ in range(set_count)
            )
            system = FiniteSetSystem(ground_set_size=n, sets=sets)
            brute = min(
                max(
                    (
                        abs(sum(coloring[element] for element in subset))
                        for subset in sets
                    ),
                    default=0,
                )
                for coloring in itertools.product((-1, 1), repeat=n)
            )
            result = compute_optimal_discrepancy(
                DiscrepancyOptimumRequest(set_system=system)
            )
            assert result.status == "OPTIMAL"
            assert result.optimal_discrepancy == brute

    def test_solver_scale_beyond_bruteforce(self) -> None:
        """A 40-element instance solves in seconds; 2^40 scanning cannot.

        Pair sets {2i, 2i+1} admit the alternating coloring with discrepancy
        zero, and D >= 0 makes zero the proven optimum as soon as the
        feasible coloring is found.
        """
        import time

        n = 40
        sets = tuple((2 * index, 2 * index + 1) for index in range(20))
        system = FiniteSetSystem(ground_set_size=n, sets=sets)
        start = time.monotonic()
        result = compute_optimal_discrepancy(
            DiscrepancyOptimumRequest(set_system=system)
        )
        elapsed = time.monotonic() - start
        assert result.status == "OPTIMAL"
        assert result.optimal_discrepancy == 0
        assert len(result.optimal_coloring) == n
        # Replay: the returned coloring attains the reported optimum.
        replayed = max(
            abs(sum(result.optimal_coloring[element] for element in subset))
            for subset in sets
        )
        assert replayed == result.optimal_discrepancy
        assert elapsed < 60

    @pytest.mark.parametrize(
        ("milp_status", "expected_status"),
        [
            (1, "BUDGET_EXCEEDED"),
            (2, "EXECUTION_FAILED"),
            (3, "EXECUTION_FAILED"),
            (4, "EXECUTION_FAILED"),
        ],
    )
    def test_milp_statuses_map_to_distinct_typed_outcomes(
        self,
        milp_status: int,
        expected_status: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from types import SimpleNamespace

        def fake_milp(**_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(status=milp_status, x=None)

        monkeypatch.setattr("scipy.optimize.milp", fake_milp)
        req = DiscrepancyOptimumRequest(
            set_system=FiniteSetSystem(ground_set_size=2, sets=((0, 1),)),
        )

        result = compute_optimal_discrepancy(req)

        assert result.status == expected_status
        assert result.optimal_coloring == ()
        assert result.optimal_discrepancy is None
        assert result.set_system == req.set_system
        assert DiscrepancyOptimumResult.model_validate(result.model_dump()) == result

    @pytest.mark.parametrize(
        "malformed_x",
        [
            "nan_vector",
            "partially_nan_vector",
            "inf_bound",
            "short_vector",
            "long_vector",
            "non_array_vector",
            "outside_binary_bounds",
            "integral_outside_binary_bounds",
        ],
    )
    def test_malformed_status_zero_solution_is_execution_failed(
        self,
        malformed_x: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A status-zero backend response with NaN, inf, a wrong-shape
        vector, or assignments outside the binary domain must become typed
        EXECUTION_FAILED; NaN comparisons evaluate false and thresholding
        would otherwise coerce out-of-domain values into a coloring."""
        from types import SimpleNamespace

        import numpy as np

        n = 2
        fixtures = {
            "nan_vector": np.array([np.nan] * (n + 1)),
            "partially_nan_vector": np.array([0.0, np.nan, 1.0]),
            "inf_bound": np.array([0.0, 1.0, np.inf]),
            "short_vector": np.ones(n),
            "long_vector": np.ones(n + 2),
            # A status-zero backend response still has to be an inspectable
            # finite vector.  A tuple has no array shape and previously
            # escaped from the canonical conversion as AttributeError.
            "non_array_vector": (0.0, 1.0, 0.0),
            # Fractional values outside [0, 1]: the old threshold alone
            # coerced these into a coloring with a wrong discrepancy.
            "outside_binary_bounds": np.array([-1.0, -1.0, 2.0]),
            # Integral but outside [0, 1]: integrality checking alone
            # passes; only the binary-domain check rejects them.
            "integral_outside_binary_bounds": np.array([0.0, 2.0, -1.0]),
        }

        def fake_milp(**_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(status=0, x=fixtures[malformed_x])

        monkeypatch.setattr("scipy.optimize.milp", fake_milp)
        req = DiscrepancyOptimumRequest(
            set_system=FiniteSetSystem(ground_set_size=n, sets=((0, 1),)),
        )

        result = compute_optimal_discrepancy(req)

        assert result.status == "EXECUTION_FAILED"
        assert result.optimal_coloring == ()
        assert result.optimal_discrepancy is None
        assert result.set_system == req.set_system
        assert DiscrepancyOptimumResult.model_validate(result.model_dump()) == result

    def test_execution_failed_result_carries_no_claim(self) -> None:
        system = FiniteSetSystem(ground_set_size=2, sets=((0, 1),))
        with _validation_code("discrepancy_theory.incomplete_result_carries_claim"):
            DiscrepancyOptimumResult(
                set_system=system,
                status="EXECUTION_FAILED",
                optimal_coloring=(1, -1),
                optimal_discrepancy=0,
            )

    def test_budget_exceeded_result_carries_no_claim(self) -> None:
        system = FiniteSetSystem(ground_set_size=2, sets=((0, 1),))
        with _validation_code("discrepancy_theory.incomplete_result_carries_claim"):
            DiscrepancyOptimumResult(
                set_system=system,
                status="BUDGET_EXCEEDED",
                optimal_coloring=(1, -1),
                optimal_discrepancy=0,
            )

    def test_budget_exceeded_serialization_carries_no_completeness_claim(self) -> None:
        result = DiscrepancyOptimumResult.model_validate(
            {
                "set_system": {"ground_set_size": 2, "sets": [[0, 1]]},
                "status": "BUDGET_EXCEEDED",
            }
        )
        assert "exhaustive" not in result.model_dump()
        assert result.optimal_coloring == ()
        assert result.optimal_discrepancy is None

    def test_optimum_result_deserialization_does_not_run_the_solver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = {
            "set_system": {"ground_set_size": 2, "sets": [[0, 1]]},
            "status": "OPTIMAL",
            "optimal_coloring": [1, 1],
            "optimal_discrepancy": 0,
        }

        def solver_must_not_run(_system: object, _allowed: int) -> str:
            raise AssertionError("result deserialization must remain structural")

        monkeypatch.setattr(
            discrepancy_models, "_feasibility_outcome", solver_must_not_run
        )

        result = DiscrepancyOptimumResult.model_validate(payload)

        assert result.optimal_discrepancy == 0

    def test_zero_optimum_validates_without_a_lower_bound_solve(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail_if_asked(_system: object, _allowed: int) -> str:
            raise AssertionError("zero lower bound is definitional, not solved")

        monkeypatch.setattr(discrepancy_models, "_feasibility_outcome", fail_if_asked)
        result = compute_optimal_discrepancy(
            DiscrepancyOptimumRequest(
                set_system=FiniteSetSystem(ground_set_size=2, sets=((0, 1),))
            )
        )
        assert result.status == "OPTIMAL"
        assert result.optimal_discrepancy == 0
        assert DiscrepancyOptimumResult.model_validate(result.model_dump()) == result

    @pytest.mark.parametrize(
        ("proof_outcome", "expected_status"),
        [
            ("unsat", "OPTIMAL"),
            ("sat", "EXECUTION_FAILED"),
            ("unknown", "BUDGET_EXCEEDED"),
        ],
    )
    def test_producing_optimal_claims_require_the_exact_proof_outcome(
        self,
        proof_outcome: str,
        expected_status: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A positive incumbent may only become OPTIMAL after the exact Z3
        feasibility check re-establishes the lower bound; any other outcome
        downgrades the result to a claim-free status."""

        def fixed_outcome(_system: object, allowed: int) -> str:
            assert allowed == 1
            return proof_outcome

        monkeypatch.setattr(discrepancy_models, "_feasibility_outcome", fixed_outcome)
        req = DiscrepancyOptimumRequest(
            set_system=FiniteSetSystem(
                ground_set_size=3,
                sets=((0, 1), (1, 2), (0, 2)),
            ),
        )

        result = compute_optimal_discrepancy(req)

        assert result.status == expected_status
        if expected_status == "OPTIMAL":
            assert result.optimal_discrepancy == 2
            assert DiscrepancyOptimumResult.model_validate(result.model_dump()) == (
                result
            )
        else:
            assert result.optimal_coloring == ()
            assert result.optimal_discrepancy is None

    def test_proving_checker_failure_stays_typed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A raising proof checker must translate to the claim-free outcome,
        never escape compute_optimal_discrepancy as a host exception."""

        def broken_checker(_system: object, _allowed: int) -> str:
            raise RuntimeError("z3 backend crashed")

        monkeypatch.setattr(discrepancy_models, "_feasibility_outcome", broken_checker)
        req = DiscrepancyOptimumRequest(
            set_system=FiniteSetSystem(
                ground_set_size=3,
                sets=((0, 1), (1, 2), (0, 2)),
            ),
        )

        result = compute_optimal_discrepancy(req)

        assert result.status == "BUDGET_EXCEEDED"
        assert result.optimal_coloring == ()
        assert result.optimal_discrepancy is None

    def test_milp_exception_becomes_execution_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """scipy.optimize.milp raising instead of returning must surface as
        the typed EXECUTION_FAILED outcome, not a kernel exception."""

        def exploding_milp(**_kwargs: object) -> object:
            raise RuntimeError("HiGHS native invocation failed")

        monkeypatch.setattr("scipy.optimize.milp", exploding_milp)
        req = DiscrepancyOptimumRequest(
            set_system=FiniteSetSystem(ground_set_size=2, sets=((0, 1),)),
        )

        result = compute_optimal_discrepancy(req)

        assert result.status == "EXECUTION_FAILED"
        assert result.optimal_coloring == ()
        assert result.optimal_discrepancy is None
        assert DiscrepancyOptimumResult.model_validate(result.model_dump()) == result

    @pytest.mark.parametrize(
        "blocked_module",
        ["numpy", "scipy.optimize"],
    )
    def test_backend_initialization_failure_is_execution_failed(
        self,
        blocked_module: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A NumPy/SciPy ABI or dynamic-loader failure during backend
        initialization must translate to typed EXECUTION_FAILED instead of
        escaping compute_optimal_discrepancy as an ImportError/OSError."""

        monkeypatch.setitem(sys.modules, blocked_module, None)
        req = DiscrepancyOptimumRequest(
            set_system=FiniteSetSystem(ground_set_size=2, sets=((0, 1),)),
        )

        result = compute_optimal_discrepancy(req)

        assert result.status == "EXECUTION_FAILED"
        assert result.optimal_coloring == ()
        assert result.optimal_discrepancy is None
        assert result.set_system == req.set_system
        assert DiscrepancyOptimumResult.model_validate(result.model_dump()) == result

    def test_solver_options_carry_node_and_time_budgets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Branch-and-bound work is capped by node_limit, not wall time alone."""
        from types import SimpleNamespace

        import numpy as np

        from jacobian.math.combinatorics.discrepancy._models import (
            MAX_OPTIMUM_SOLVER_MILLISECONDS,
            MAX_OPTIMUM_SOLVER_NODES,
        )

        captured: dict[str, object] = {}

        def fake_milp(**kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            # b = (1, 0) encodes the alternating coloring (+1, -1), whose
            # imbalance on the single pair is 0, matching t = 0.
            return SimpleNamespace(status=0, x=np.array([1.0, 0.0, 0.0]))

        monkeypatch.setattr("scipy.optimize.milp", fake_milp)
        req = DiscrepancyOptimumRequest(
            set_system=FiniteSetSystem(ground_set_size=2, sets=((0, 1),)),
        )

        result = compute_optimal_discrepancy(req)

        options = captured["options"]
        assert isinstance(options, dict)
        assert options["node_limit"] == MAX_OPTIMUM_SOLVER_NODES
        assert options["time_limit"] == MAX_OPTIMUM_SOLVER_MILLISECONDS / 1000
        assert result.status == "OPTIMAL"
        assert result.optimal_discrepancy == 0

    def test_easy_instance_still_proves_bounds_after_bound_check(self) -> None:
        """The bound-checking kernel still reports OPTIMAL with its invariant."""
        system = FiniteSetSystem(ground_set_size=4, sets=((0, 1), (2, 3)))
        result = compute_optimal_discrepancy(
            DiscrepancyOptimumRequest(set_system=system)
        )
        assert result.status == "OPTIMAL"
        assert result.optimal_discrepancy == 0
        replayed = max(
            abs(sum(result.optimal_coloring[element] for element in subset))
            for subset in system.sets
        )
        assert replayed == result.optimal_discrepancy

    @pytest.mark.parametrize(
        ("timed_out", "expected_status"),
        ((False, "EXECUTION_FAILED"), (True, "BUDGET_EXCEEDED")),
    )
    def test_tool_worker_failure_stays_claim_free(
        self,
        timed_out: bool,
        expected_status: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def failed_worker(*_args: object, **_kwargs: object) -> BoundedProcessResult:
            return BoundedProcessResult(
                returncode=None if timed_out else -11,
                stdout=b"",
                stderr=b"ASSERTION VIOLATION",
                stdout_exceeded=False,
                stderr_exceeded=False,
                timed_out=timed_out,
            )

        monkeypatch.setattr(
            optimum_process,
            "run_bounded_process",
            failed_worker,
        )
        request = DiscrepancyOptimumRequest(
            set_system=FiniteSetSystem(
                ground_set_size=3,
                sets=((0, 1), (1, 2), (0, 2)),
            )
        )

        result = discrepancy_operations._compute_optimal_discrepancy_isolated(request)

        assert result.status == expected_status
        assert result.set_system == request.set_system
        assert result.optimal_coloring == ()
        assert result.optimal_discrepancy is None

    @pytest.mark.scale
    def test_hard_instance_outcome_is_honest(self) -> None:
        """Whatever the budget outcome, an OPTIMAL claim has the defining
        discrepancy invariant and BUDGET_EXCEEDED carries no witness — a timed-out incumbent must not
        be labeled optimal."""
        import random

        generator = random.Random(7)
        n = 36
        sets = tuple(tuple(sorted(generator.sample(range(n), 18))) for _ in range(24))
        request = DiscrepancyOptimumRequest(
            set_system=FiniteSetSystem(ground_set_size=n, sets=sets)
        )
        result = compute_optimal_discrepancy(request)
        assert result.status in {"OPTIMAL", "BUDGET_EXCEEDED"}
        assert result.set_system == request.set_system
        sets = request.set_system.sets
        if result.status == "OPTIMAL":
            replayed = max(
                abs(sum(result.optimal_coloring[element] for element in subset))
                for subset in sets
            )
            assert replayed == result.optimal_discrepancy
            # The claimed optimum is achievable, so it upper-bounds every
            # coloring; optimality itself was established by coinciding
            # solver bounds before this branch could run.
            assert result.optimal_discrepancy >= 0
        else:
            assert result.optimal_coloring == ()
            assert result.optimal_discrepancy is None
