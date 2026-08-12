from __future__ import annotations

import pytest

from jacobian.contracts.base import ContractModel
from jacobian.value_references import ValueReferenceError, ValueReferenceStore


class _Value(ContractModel):
    entries: tuple[int, ...]


class _OtherValue(ContractModel):
    entries: tuple[int, ...]


def test_value_reference_binds_exact_type_digest_and_source() -> None:
    store = ValueReferenceStore()
    value = _Value(entries=(1, 2, 3))

    value_ref = store.put(
        value,
        operation_id="synthetic.value.produce",
        operation_version="2",
        output_port="value",
    )
    stored = store.inspect(value_ref)

    assert value_ref.startswith("value://")
    assert stored.digest.startswith("sha256:")
    assert stored.source_operation_id == "synthetic.value.produce"
    assert stored.source_operation_version == "2"
    assert stored.source_port == "value"
    assert store.resolve(value_ref, _Value) == value


def test_value_reference_rejects_wrong_types_other_runtimes_and_closed_lifetimes() -> (
    None
):
    first = ValueReferenceStore()
    second = ValueReferenceStore()
    value_ref = first.put(
        _Value(entries=(1,)),
        operation_id="synthetic.value.produce",
        operation_version="1",
        output_port="value",
    )

    with pytest.raises(ValueReferenceError, match="expected _OtherValue"):
        first.resolve(value_ref, _OtherValue)
    with pytest.raises(ValueReferenceError, match="another runtime"):
        second.resolve(value_ref, _Value)

    first.close()
    with pytest.raises(ValueReferenceError, match="another runtime"):
        first.resolve(value_ref, _Value)


def test_value_reference_store_evicts_least_recently_used_values() -> None:
    store = ValueReferenceStore()
    references = [
        store.put(
            _Value(entries=(index,)),
            operation_id="synthetic.value.produce",
            operation_version="1",
            output_port="value",
        )
        for index in range(256)
    ]
    store.resolve(references[0], _Value)

    newest = store.put(
        _Value(entries=(256,)),
        operation_id="synthetic.value.produce",
        operation_version="1",
        output_port="value",
    )

    with pytest.raises(ValueReferenceError, match="another runtime"):
        store.resolve(references[1], _Value)
    assert store.resolve(references[0], _Value).entries == (0,)
    assert store.resolve(newest, _Value).entries == (256,)
