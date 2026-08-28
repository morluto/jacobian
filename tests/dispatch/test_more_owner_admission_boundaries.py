"""Owner admission remains typed after wire parsing."""

from __future__ import annotations

import copy

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation


def _example_payload(operation_id: str) -> dict[str, object]:
    operation = Catalog.open().operation(operation_id)
    assert operation is not None
    assert operation.examples
    return copy.deepcopy(operation.examples[0].input)


def test_factor_length_admission_is_typed_after_wire_parsing() -> None:
    payload = _example_payload("word.factors.length.compute")
    payload["factor_length"] = 100

    with pytest.raises(OperationDomainValidationError) as caught:
        invoke_operation("word.factors.length.compute", payload, Catalog.open())

    assert caught.value.errors() == (
        {
            "loc": ("factor_length",),
            "type": "word.factor_length_out_of_range",
            "msg": "factor length must be between zero and the word length",
        },
    )


def test_chain_complex_count_admission_is_typed_after_wire_parsing() -> None:
    payload = {
        "coefficient_field": "QQ",
        "basis_sizes": [1, 1, 1],
        "differential_matrices": [[["0"]]],
    }

    with pytest.raises(OperationDomainValidationError) as caught:
        invoke_operation("chain_complex.construct.compute", payload, Catalog.open())

    assert caught.value.errors()[0]["loc"] == ("differential_matrices",)
    assert (
        caught.value.errors()[0]["type"] == "chain_complex.differential_count_mismatch"
    )


def test_chain_complex_differential_admission_is_typed_after_wire_parsing() -> None:
    payload = {
        "coefficient_field": "QQ",
        "basis_sizes": [1, 1, 1],
        "differential_matrices": [[["1"]], [["1"]]],
    }

    with pytest.raises(OperationDomainValidationError) as caught:
        invoke_operation("chain_complex.construct.compute", payload, Catalog.open())

    assert caught.value.errors() == (
        {
            "loc": ("differential_matrices",),
            "type": "chain_complex.differential_not_square_zero",
            "msg": "constructed complex violates d^2=0 at chain degree 1",
        },
    )


def test_comultiplication_index_admission_is_typed_after_wire_parsing() -> None:
    payload = _example_payload("coalgebra.comultiplication.compute")
    payload["element_index"] = 2

    with pytest.raises(OperationDomainValidationError) as caught:
        invoke_operation("coalgebra.comultiplication.compute", payload, Catalog.open())

    assert caught.value.errors() == (
        {
            "loc": ("element_index",),
            "type": "coalgebra.element_index_out_of_range",
            "msg": "element_index must be in 0..dimension-1",
        },
    )


def test_comultiplication_coalgebra_admission_is_typed() -> None:
    payload = _example_payload("coalgebra.comultiplication.compute")
    coalgebra = payload["coalgebra"]
    assert isinstance(coalgebra, dict)
    coalgebra["prime"] = 4

    with pytest.raises(OperationDomainValidationError) as caught:
        invoke_operation("coalgebra.comultiplication.compute", payload, Catalog.open())

    assert caught.value.errors()[0]["loc"] == ("coalgebra",)
    assert caught.value.errors()[0]["type"] == "coalgebra.prime_not_prime"
