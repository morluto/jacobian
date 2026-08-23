"""Tests for discrepancy theory operations."""

from __future__ import annotations

import itertools
from fractions import Fraction

import pytest
from pydantic import ValidationError

import jacobian.math.discrepancy_theory._models as discrepancy_models
from jacobian.math.discrepancy_theory._models import (
    MAX_COLUMN_INCIDENCES,
    MAX_MONITORED_COLUMNS,
    MAX_ROUNDING_COORDINATES,
    MAX_ROUNDING_INTERMEDIATE_DIGITS,
    MAX_ROUNDING_RATIONAL_DIGITS,
    MAX_ROUNDING_RESULT_RATIONAL_DIGITS,
    MAX_ROUNDING_WORK,
    DiscrepancyEvalRequest,
    DiscrepancyOptimumRequest,
    FiniteSetSystem,
    HardConstraintRoundingRequest,
    HardConstraintRoundingResult,
)
from jacobian.math.discrepancy_theory._operations import (
    compute_discrepancy,
    compute_hard_constraint_rounding,
    compute_optimal_discrepancy,
)


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
    rows: tuple[dict[str, object], ...] = (
        {"label": "left", "coordinates": (0, 1)},
        {"label": "right", "coordinates": (2, 3)},
    ),
    columns: tuple[dict[str, object], ...] = (
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
            ("values", (_rational(Fraction(-1, 2)),) * 4, r"lie in \[0, 1\]"),
            (
                "values",
                (_rational(Fraction(1, 3)),) + (_rational(Fraction(1, 2)),) * 3,
                "integral source sum",
            ),
            (
                "rows",
                ({"label": "only", "coordinates": (0, 1)},),
                "partition every coordinate",
            ),
            (
                "rows",
                (
                    {"label": "left", "coordinates": (0, 1)},
                    {"label": "right", "coordinates": (0, 1, 2, 3)},
                ),
                "partition every coordinate",
            ),
            (
                "columns",
                ({"label": "bad", "coordinates": (2, 0)},),
                "strictly increasing",
            ),
            (
                "columns",
                ({"label": "bad", "coordinates": (4,)},),
                "coordinate_count",
            ),
        ],
    )
    def test_malformed_sources_are_rejected_before_the_kernel(
        self, field: str, replacement: object, message: str
    ) -> None:
        payload = _rounding_request().model_dump()
        payload["source"][field] = replacement
        with pytest.raises(ValidationError, match=message):
            HardConstraintRoundingRequest.model_validate(payload)

    @pytest.mark.parametrize(
        ("mutation", "message"),
        [
            ("bit", "preserve every hard row"),
            ("row", "row ledger"),
            ("column", "column ledger"),
            ("source", "column ledger"),
        ],
    )
    def test_source_bound_result_rejects_authored_mutations(
        self, mutation: str, message: str
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
        with pytest.raises(ValidationError, match=message):
            HardConstraintRoundingResult.model_validate(payload)

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
        with pytest.raises(ValidationError, match="at most 512 items"):
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
        with pytest.raises(ValidationError, match="at most 512 items"):
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
        with pytest.raises(ValidationError, match="incidences exceed"):
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

        with pytest.raises(ValidationError, match="incidences exceed"):
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
        with pytest.raises(ValidationError, match="256-digit bound"):
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
        with pytest.raises(ValidationError, match="work bound exceeded"):
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
        with pytest.raises(ValidationError, match="intermediate rational-height"):
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
        with pytest.raises(ValidationError, match="result-size bound"):
            _rounding_request(
                values=values,
                rows=({"label": "r", "coordinates": (0, 1)},),
                columns=columns(accepted_columns + 1),
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
