"""Lazy SymPy backend for polynomial-map operations."""

from __future__ import annotations

from typing import Any, NamedTuple

from jacobian.providers import LazyLoader


class _SympyBackend(NamedTuple):
    """Heavy SymPy implementation symbols loaded on first operation invocation."""

    QQ: Any
    Matrix: Any
    Poly: Any
    expand: Any
    solve: Any
    symbols: Any
    sympify: Any
    Rational: Any
    PolynomialError: type


def _load_sympy_backend() -> _SympyBackend:
    """Construct the pinned SymPy implementation bundle on first use."""
    from sympy import QQ, Matrix, Poly, Rational, expand, solve, symbols, sympify
    from sympy.polys.polyerrors import PolynomialError

    return _SympyBackend(
        QQ,
        Matrix,
        Poly,
        expand,
        solve,
        symbols,
        sympify,
        Rational,
        PolynomialError,
    )


_sympy: LazyLoader[_SympyBackend] = LazyLoader(
    _load_sympy_backend, component_id="jacobian.sympy.polynomial-maps"
)
