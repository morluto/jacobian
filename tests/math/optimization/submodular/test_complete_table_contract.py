"""Complete-table diagnostics preserve unordered-subset semantics."""

from itertools import permutations

import pytest
from pydantic import BaseModel, ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.optimization.submodular._models import (
    MonotonicityCheckRequest,
    MonotonicityCheckResult,
    SetFunction,
    SetFunctionEntry,
    SetFunctionEvalRequest,
    SetFunctionEvalResult,
    SubmodularityCheckRequest,
    SubmodularityCheckResult,
)


def _entry(subset: tuple[int, ...]) -> SetFunctionEntry:
    return SetFunctionEntry(
        subset=subset, value=CanonicalRational(num=len(subset), den=1)
    )


def test_every_row_order_and_reversed_subset_order_remains_valid() -> None:
    rows = (_entry(()), _entry((0,)), _entry((1,)), _entry((1, 0)))
    for entries in permutations(rows):
        value = SetFunction(ground_set_size=2, entries=entries)
        assert value.entries == entries
        assert SetFunction.model_validate_json(value.model_dump_json()) == value


def test_zero_ground_set_requires_its_empty_subset() -> None:
    assert SetFunction(ground_set_size=0, entries=(_entry(()),)).entries == (
        _entry(()),
    )


@pytest.mark.parametrize(
    ("subsets", "code", "detail"),
    [
        (((), ()), "table_subsets_not_unique", "missing subset [0]"),
        (((),), "table_entry_count_mismatch", "expected 2 entries"),
        (((), (0, 0)), "subset_elements_not_unique", "entries[1].subset"),
        (((), (1,)), "subset_element_out_of_range", "entries[1].subset"),
    ],
)
def test_invalid_table_points_to_entries_and_explains_correction(
    subsets: tuple[tuple[int, ...], ...], code: str, detail: str
) -> None:
    with pytest.raises(ValidationError) as raised:
        SetFunction(
            ground_set_size=1, entries=tuple(_entry(subset) for subset in subsets)
        )
    error = raised.value.errors()[0]
    assert error["loc"] == ("entries",)
    assert error["type"] == f"submodular_opt.{code}"
    assert detail in error["msg"]


def test_permuted_duplicate_reports_indices_and_an_actually_missing_subset() -> None:
    entries = tuple(_entry(s) for s in ((), (0,), (0, 1), (1, 0)))
    with pytest.raises(ValidationError) as raised:
        SetFunction(ground_set_size=2, entries=entries)
    message = raised.value.errors()[0]["msg"]
    assert "entries[3]" in message and "entries[2]" in message
    assert "missing subset [1]" in message
    repaired = SetFunction(ground_set_size=2, entries=(*entries[:3], _entry((1,))))
    assert len(repaired.entries) == 4


def test_bad_ground_set_is_not_masked_by_table_validator() -> None:
    with pytest.raises(ValidationError) as raised:
        SetFunction(ground_set_size=-1, entries=(_entry(()),))
    assert raised.value.errors()[0]["loc"] == ("ground_set_size",)


@pytest.mark.parametrize(
    ("carrier", "fields"),
    [
        (SetFunctionEvalRequest, {"subset": ()}),
        (MonotonicityCheckRequest, {}),
        (SubmodularityCheckRequest, {}),
        (
            SetFunctionEvalResult,
            {"subset": (), "value": CanonicalRational(num=0, den=1)},
        ),
        (MonotonicityCheckResult, {"is_monotone": True}),
        (SubmodularityCheckResult, {"is_submodular": True}),
    ],
)
def test_nested_native_source_rechecks_complete_table(
    carrier: type[BaseModel], fields: dict[str, object]
) -> None:
    valid = SetFunction(ground_set_size=1, entries=(_entry(()), _entry((0,))))
    forged = valid.model_copy(update={"entries": (_entry(()), _entry(()))})
    with pytest.raises(ValidationError) as raised:
        carrier(function=forged, **fields)
    error = raised.value.errors()[0]
    assert error["loc"] == ("function", "entries")
    assert error["type"] == "submodular_opt.table_subsets_not_unique"
