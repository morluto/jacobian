"""Exact contract tests for skew Littlewood-Richardson membership."""

from __future__ import annotations

from jacobian.math.combinatorics.algebraic._models import (
    SkewLittlewoodRichardsonCheckRequest,
    SkewLittlewoodRichardsonCheckResult,
)
from jacobian.math.combinatorics.algebraic._tools import (
    TOOLS,
    check_skew_littlewood_richardson,
)
from jacobian.math.combinatorics.algebraic.operations import (
    verify_skew_littlewood_richardson,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    TableauCandidate,
)


def _request(
    outer: tuple[int, ...],
    inner: tuple[int, ...],
    rows: list[list[int]],
    content: tuple[int, ...],
) -> SkewLittlewoodRichardsonCheckRequest:
    return SkewLittlewoodRichardsonCheckRequest(
        outer=IntegerPartition(parts=outer),
        inner=IntegerPartition(parts=inner),
        tableau=TableauCandidate(rows=tuple(tuple(row) for row in rows)),
        content=IntegerPartition(parts=content),
    )


def test_membership_for_small_skew_shape() -> None:
    result = check_skew_littlewood_richardson(
        _request((2, 1), (1,), [[1], [2]], (1, 1))
    )

    assert result.is_member
    assert result.failure_kind == "OK"
    assert result.reading_word == (1, 2)


def test_lattice_word_violation_is_reported_with_prefix() -> None:
    result = check_skew_littlewood_richardson(
        _request((2, 1), (1,), [[2], [1]], (1, 1))
    )

    assert not result.is_member
    assert result.failure_kind == "LATTICE"
    assert result.failed_prefix_length == 1
    assert result.failed_value == 2


def test_non_containment_is_a_cell_coverage_failure() -> None:
    result = check_skew_littlewood_richardson(_request((1,), (2,), [[1]], (1,)))

    assert not result.is_member
    assert result.failure_kind == "CELL_COVERAGE"


def test_row_coverage_mismatch_is_reported() -> None:
    result = check_skew_littlewood_richardson(_request((2, 1), (1,), [[1, 2]], (1, 1)))

    assert not result.is_member
    assert result.failure_kind == "CELL_COVERAGE"


def test_column_strictness_is_replayed_on_vertical_cells() -> None:
    result = check_skew_littlewood_richardson(
        _request((2, 2), (1, 1), [[1], [1]], (2,))
    )

    assert not result.is_member
    assert result.failure_kind == "SEMISTANDARD_COLUMN"


def test_content_mismatch_is_reported_with_value() -> None:
    result = check_skew_littlewood_richardson(_request((2, 1), (1,), [[1], [2]], (2,)))

    assert not result.is_member
    assert result.failure_kind == "CONTENT"
    assert result.failed_value == 1


def test_serialized_claim_round_trips_and_forgery_is_rejected() -> None:
    result = check_skew_littlewood_richardson(
        _request((2, 1), (1,), [[1], [2]], (1, 1))
    )
    decoded = SkewLittlewoodRichardsonCheckResult.model_validate_json(
        result.model_dump_json()
    )

    assert verify_skew_littlewood_richardson(decoded)

    forged = decoded.model_copy(update={"is_member": False, "failure_kind": "LATTICE"})
    assert not verify_skew_littlewood_richardson(forged)


def test_declaration_example_executes() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "combinatorics.tableau.skew_littlewood_richardson.check"
    )

    example_request = SkewLittlewoodRichardsonCheckRequest.model_validate(
        dict(tool.examples[0].input)
    )
    result = tool.run(example_request)

    assert isinstance(result, SkewLittlewoodRichardsonCheckResult)
    assert result.is_member
