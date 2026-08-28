"""Native and public-dispatch parity for number-theory admission."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.number_theory._direct_factorization_models import (
    FactorizationRequest,
)
from jacobian.math.number_theory._discrete_logarithm import (
    DISCRETE_LOGARITHM_OPERATION,
    DiscreteLogarithmRequest,
)
from jacobian.math.number_theory._divisibility_models import ValuationRequest
from jacobian.math.number_theory._divisibility_operations import (
    compute_valuation,
)
from jacobian.math.number_theory._factorization_kernels import enumerate_divisors
from jacobian.math.number_theory._modular_basic_models import (
    ChineseRemainderRequest,
    JacobiSymbolRequest,
    ModularUnitRequest,
)
from jacobian.math.number_theory._modular_models import (
    ModularPolynomialResidueImageRequest,
    ModularPolynomialVariable,
)
from jacobian.math.number_theory._modular_operations import (
    compute_jacobi_symbol,
    compute_modular_inverse,
    compute_modular_polynomial_residue_image,
    solve_chinese_remainder,
)
from jacobian.math.number_theory._ramanujan_sum import (
    RamanujanSumRequest,
    compute_ramanujan_sum,
)
from jacobian.math.number_theory.modular_polynomials import ModularPolynomialTerm


@pytest.mark.parametrize(
    ("operation_id", "payload", "request_model", "native"),
    (
        (
            "integer.compute.valuation",
            {"value": "1", "prime": "4"},
            ValuationRequest(value="1", prime="4"),
            compute_valuation,
        ),
        (
            "modular.compute.inverse",
            {"value": "2", "modulus": 4},
            ModularUnitRequest(value="2", modulus=4),
            compute_modular_inverse,
        ),
        (
            "number_theory.compute.jacobi_symbol",
            {"a": "1", "n": 4},
            JacobiSymbolRequest(a="1", n=4),
            compute_jacobi_symbol,
        ),
        (
            "number_theory.ramanujan_sum.compute",
            {"modulus": "-1", "frequency": "0"},
            RamanujanSumRequest(modulus="-1", frequency="0"),
            compute_ramanujan_sum,
        ),
    ),
)
def test_native_and_dispatch_share_domain_admission(
    operation_id: str,
    payload: dict[str, Any],
    request_model: Any,
    native: Callable[[Any], Any],
) -> None:
    with pytest.raises(OperationDomainValidationError) as native_error:
        native(request_model)
    with pytest.raises(OperationDomainValidationError) as dispatch_error:
        invoke_operation(operation_id, payload, Catalog.open())

    assert dispatch_error.value.errors() == native_error.value.errors()


def test_native_and_dispatch_share_crt_admission() -> None:
    request = ChineseRemainderRequest(residues=(3,), moduli=(3,))
    with pytest.raises(OperationDomainValidationError) as native_error:
        solve_chinese_remainder(request)
    with pytest.raises(OperationDomainValidationError) as dispatch_error:
        invoke_operation(
            "modular.solve.chinese_remainder",
            {"residues": [3], "moduli": [3]},
            Catalog.open(),
        )

    assert dispatch_error.value.errors() == native_error.value.errors()


def test_discrete_log_request_admission_is_native() -> None:
    request = DiscreteLogarithmRequest(base=5, target=1, modulus=5)
    with pytest.raises(OperationDomainValidationError) as native_error:
        DISCRETE_LOGARITHM_OPERATION.run(request)
    with pytest.raises(OperationDomainValidationError) as dispatch_error:
        invoke_operation(
            "modular.compute.discrete_logarithm",
            {"base": 5, "target": 1, "modulus": 5},
            Catalog.open(),
        )

    assert dispatch_error.value.errors() == native_error.value.errors()


def test_direct_factorization_admission_is_native() -> None:
    request = FactorizationRequest(value="0")
    with pytest.raises(OperationDomainValidationError) as native_error:
        enumerate_divisors(request)
    with pytest.raises(OperationDomainValidationError) as dispatch_error:
        invoke_operation(
            "integer.compute.divisors",
            {"value": "0"},
            Catalog.open(),
        )

    assert dispatch_error.value.errors() == native_error.value.errors()


def test_modular_polynomial_admission_is_native() -> None:
    request = ModularPolynomialResidueImageRequest(
        modulus=5,
        variables=(
            ModularPolynomialVariable(name="x", residues=(0, 1)),
            ModularPolynomialVariable(name="x", residues=(0, 1)),
        ),
        terms=(ModularPolynomialTerm(coefficient="1", exponents=(1, 1)),),
    )
    with pytest.raises(OperationDomainValidationError) as native_error:
        compute_modular_polynomial_residue_image(request)
    with pytest.raises(OperationDomainValidationError) as dispatch_error:
        invoke_operation(
            "modular.polynomial_residue_image.compute",
            {
                "modulus": 5,
                "variables": [
                    {"name": "x", "residues": [0, 1]},
                    {"name": "x", "residues": [0, 1]},
                ],
                "terms": [{"coefficient": "1", "exponents": [1, 1]}],
            },
            Catalog.open(),
        )

    assert dispatch_error.value.errors() == native_error.value.errors()
