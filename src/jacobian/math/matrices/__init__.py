"""Supported exact matrix API."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math.matrices.operations import (
        adjugate,
        characteristic_polynomial,
        determinant,
        inverse,
        kronecker_product,
        multiply,
        partial_trace,
        permanent,
        rank,
        rref,
        smith_normal_form,
        solve_linear_system,
        trace,
        verify_adjugate,
        verify_inverse,
        verify_kronecker_product,
        verify_nullspace,
        verify_partial_trace,
        verify_product,
    )
    from jacobian.math.matrices.values import (
        EmbeddedRealSimpleNumberFieldMatrix,
        ExactRealMatrix,
        IntegerMatrix,
        RationalMatrix,
        SmithNormalForm,
        SparseRationalMatrix,
        SparseRationalMatrixEntry,
    )

__all__ = [
    "EmbeddedRealSimpleNumberFieldMatrix",
    "ExactRealMatrix",
    "IntegerMatrix",
    "RationalMatrix",
    "SmithNormalForm",
    "SparseRationalMatrix",
    "SparseRationalMatrixEntry",
    "adjugate",
    "characteristic_polynomial",
    "determinant",
    "inverse",
    "kronecker_product",
    "multiply",
    "partial_trace",
    "permanent",
    "rank",
    "rref",
    "smith_normal_form",
    "solve_linear_system",
    "trace",
    "verify_adjugate",
    "verify_inverse",
    "verify_kronecker_product",
    "verify_nullspace",
    "verify_partial_trace",
    "verify_product",
]

_VALUE_NAMES = frozenset(
    (
        "EmbeddedRealSimpleNumberFieldMatrix",
        "ExactRealMatrix",
        "IntegerMatrix",
        "RationalMatrix",
        "SmithNormalForm",
        "SparseRationalMatrix",
        "SparseRationalMatrixEntry",
    )
)


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    owner = "values" if name in _VALUE_NAMES else "operations"
    value = getattr(import_module(f"{__name__}.{owner}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
