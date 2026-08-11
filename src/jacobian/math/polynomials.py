"""Small native exact-polynomial API backed by the domain producer kernels."""

from __future__ import annotations

from typing import Any

from jacobian.domains.polynomial import kernels

__all__ = ["derivative", "gcdex", "resultant"]


def gcdex(left: Any, right: Any) -> tuple[Any, Any, Any]:
    """Return SymPy's exact QQ extended-GCD tuple for two ``Poly`` values."""

    return kernels.polynomial_gcdex(left, right)


def resultant(left: Any, right: Any, generator: Any) -> Any:
    """Return the exact resultant in the supplied SymPy generator."""

    return kernels.polynomial_resultant(left, right, generator)


def derivative(polynomial: Any) -> Any:
    """Return the formal SymPy ``Poly`` derivative."""

    return kernels.polynomial_derivative(polynomial)
