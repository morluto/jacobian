"""Exact rational polynomial operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["polynomial_operations"]


def polynomial_operations() -> MathTools:
    from jacobian.domains.polynomial.elementary import (
        INTEGER_POLYNOMIAL_OPERATIONS,
        RATIONAL_POLYNOMIAL_OPERATIONS,
    )
    from jacobian.domains.polynomial.invariants import (
        POLYNOMIAL_INVARIANT_OPERATIONS,
    )
    from jacobian.domains.polynomial.jacobian_syzygy import (
        GRADED_JACOBIAN_SYZYGY_OPERATION,
        JACOBIAN_SYZYGY_COEFFICIENT_LEDGER_OPERATION,
    )

    return (
        *POLYNOMIAL_INVARIANT_OPERATIONS,
        GRADED_JACOBIAN_SYZYGY_OPERATION,
        JACOBIAN_SYZYGY_COEFFICIENT_LEDGER_OPERATION,
        *INTEGER_POLYNOMIAL_OPERATIONS,
        *RATIONAL_POLYNOMIAL_OPERATIONS,
    )
