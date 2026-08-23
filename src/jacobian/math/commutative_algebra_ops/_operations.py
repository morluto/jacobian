"""Domain functions for commutative algebra operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import sympy

from jacobian.math.commutative_algebra_ops._models import (
    EliminationIdealRequest,
    EliminationIdealResult,
    GroebnerBasisRequest,
    GroebnerBasisResult,
    IdealNormalFormRequest,
    IdealNormalFormResult,
    IdealQuotientRequest,
    IdealQuotientResult,
    IdealRadicalMembershipRequest,
    IdealRadicalMembershipResult,
    IdealRadicalRequest,
    IdealRadicalResult,
    IdealSaturationRequest,
    IdealSaturationResult,
)
from jacobian.math.commutative_algebra_ops._singular import (
    run_singular_ideal_operation,
)
from jacobian.math.polynomials._conversions import (
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.values import RationalPolynomialIdeal


class _GroebnerBudgetExceededError(TimeoutError):
    """Internal signal-handler escape for the enforced Gröbner budget."""


class _NormalFormTimeoutError(TimeoutError):
    """Internal signal-handler escape for the enforced normal-form budget."""


class _EliminationTimeoutError(TimeoutError):
    """Internal signal-handler escape for the enforced elimination budget."""


class _ResultLimitExceededError(ValueError):
    """The exact backend result exceeds the declared output limits."""


class _SympyKernelTimeoutError(TimeoutError):
    """The bounded SymPy worker exceeded the declared wall-time budget."""


class _SympyKernelError(RuntimeError):
    """The bounded SymPy worker failed without producing an exact result."""


_SYMPY_WORKER_SCRIPT = r'''
import json
import sys


def main() -> None:
    payload = json.loads(sys.stdin.buffer.read().decode("ascii"))
    try:
        import sympy

        from jacobian.math.polynomials._conversions import (
            rational_polynomial_from_sympy,
            rational_polynomial_to_sympy,
            symbols_for_variables,
        )
        from jacobian.math.polynomials.values import RationalPolynomial

        mode = payload["mode"]
        variables = tuple(payload["variables"])
        symbols = symbols_for_variables(variables)
        gens = [
            RationalPolynomial.model_validate(item)
            for item in payload["generators"]
        ]
        exprs = [rational_polynomial_to_sympy(g).as_expr() for g in gens]

        def dump(expr: object, terms_variable: tuple, maximum_terms: int) -> dict:
            poly = sympy.Poly(
                expr, *symbols_for_variables(terms_variable), domain=sympy.QQ
            )
            converted = rational_polynomial_from_sympy(
                poly, terms_variable, maximum_terms=maximum_terms
            )
            return converted.model_dump(mode="json")

        if mode == "groebner":
            basis = sympy.groebner(
                exprs, *symbols, order=payload["order"], domain=sympy.QQ
            )
            generators = [
                dump(expr, variables, payload["maximum_terms"]) for expr in basis
            ]
            _emit({"status": "ok", "generators": generators})
        elif mode == "normal_form":
            target = RationalPolynomial.model_validate(payload["polynomial"])
            poly_expr = rational_polynomial_to_sympy(target).as_expr()
            basis = sympy.groebner(exprs, *symbols, order="grevlex", domain=sympy.QQ)
            _, remainder = sympy.reduced(
                poly_expr,
                list(basis.exprs),
                *symbols,
                order="grevlex",
                domain=sympy.QQ,
            )
            remainder_poly = sympy.Poly(remainder, *symbols, domain=sympy.QQ)
            converted = rational_polynomial_from_sympy(
                remainder_poly, variables
            )
            _emit(
                {
                    "status": "ok",
                    "remainder": converted.model_dump(mode="json"),
                }
            )
        elif mode == "elimination":
            eliminated = set(payload["eliminated"])
            remaining = [v for v in variables if v not in eliminated]
            ordered = tuple(v for v in variables if v in eliminated) + tuple(
                remaining
            )
            ordered_symbols = symbols_for_variables(ordered)
            basis = sympy.groebner(
                exprs, *ordered_symbols, order="lex", domain=sympy.QQ
            )
            remaining_symbols = symbols_for_variables(tuple(remaining))
            generators: list[dict] = []
            unit_ideal = False
            for expr in basis:
                poly = sympy.Poly(expr, *ordered_symbols, domain=sympy.QQ)
                involved = {str(s) for s in poly.free_symbols}
                if not involved:
                    unit_ideal = True
                    break
                if involved.issubset(set(remaining)):
                    converted = rational_polynomial_from_sympy(
                        sympy.Poly(expr, *remaining_symbols, domain=sympy.QQ),
                        tuple(remaining),
                    )
                    generators.append(converted.model_dump(mode="json"))
            _emit(
                {
                    "status": "ok",
                    "unit_ideal": unit_ideal,
                    "generators": generators,
                    "remaining": remaining,
                }
            )
        else:  # pragma: no cover - parent owns the mode vocabulary
            _emit({"status": "error", "detail": "unknown kernel mode"})
    except Exception as error:  # noqa: BLE001 - classified for the parent
        message = str(error)
        status = "limit" if "operation budget" in message else "error"
        _emit({"status": status, "detail": message})


def _emit(result: dict) -> None:
    json.dump(result, sys.stdout)
    sys.stdout.flush()


main()
'''

_STDOUT_LIMIT = 8 * 1024 * 1024
_STDERR_LIMIT = 64 * 1024


def _run_sympy_kernel(payload: dict[str, Any], wall_seconds: float) -> dict[str, Any]:
    """Run one exact Groebner-kernel computation in a killable worker.

    The child process is terminated on wall-budget expiry, so an admitted
    request cannot leave detached computations running inside the server.
    """
    import json
    import shutil
    import sys
    import tempfile

    from jacobian.process import (
        ProcessPlatformTools,
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    # Deliberately not resolved: following the interpreter symlink would
    # reparent the worker onto the base prefix without the environment's
    # site-packages.
    resolved = shutil.which(sys.executable) or sys.executable
    prlimit = shutil.which("prlimit")
    if prlimit is not None:
        prlimit = str(Path(prlimit).resolve())
    try:
        with tempfile.TemporaryDirectory(prefix="jacobian-sympy-") as directory:
            completed = run_bounded_process(
                [resolved, "-I", "-c", _SYMPY_WORKER_SCRIPT],
                input_bytes=json.dumps(payload).encode("ascii"),
                timeout_seconds=float(wall_seconds),
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_STDOUT_LIMIT,
                stderr_limit=_STDERR_LIMIT,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=int(wall_seconds),
                    address_space_bytes=2 * 1024 * 1024 * 1024,
                    file_size_bytes=1024 * 1024,
                ),
                platform_tools=ProcessPlatformTools(prlimit_executable=prlimit),
                cwd=directory,
            )
    except OSError as error:
        raise _SympyKernelError(str(error)) from None
    if completed.timed_out:
        raise _SympyKernelTimeoutError()
    if completed.cancelled:
        raise _SympyKernelError("kernel execution was cancelled")
    try:
        result = json.loads(completed.stdout.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as error:
        raise _SympyKernelError(str(error)) from None
    if result.get("status") == "limit":
        raise _ResultLimitExceededError(result.get("detail", ""))
    if result.get("status") != "ok":
        raise _SympyKernelError(str(result.get("detail", "kernel failed")))
    typed: dict[str, Any] = result
    return typed


def compute_ideal_radical(request: IdealRadicalRequest) -> IdealRadicalResult:
    """Compute an exact ideal radical through the bounded Singular backend."""

    backend = run_singular_ideal_operation(
        "radical",
        request.ideal,
        None,
        request.resource_budget,
    )
    return IdealRadicalResult(
        outcome=backend.outcome,
        radical=backend.ideal,
        backend_version=backend.backend_version,
        detail=backend.detail,
    )


def compute_ideal_radical_membership(
    request: IdealRadicalMembershipRequest,
) -> IdealRadicalMembershipResult:
    """Decide radical membership by the exact Rabinowitsch criterion."""

    variable_symbols = symbols_for_variables(request.ideal.variables)
    ideal_generators = [
        rational_polynomial_to_sympy(generator).as_expr()
        for generator in request.ideal.generators
    ]
    polynomial = rational_polynomial_to_sympy(request.polynomial).as_expr()
    auxiliary = sympy.Dummy("jacobian_rabinowitsch")
    basis = sympy.groebner(
        [*ideal_generators, 1 - auxiliary * polynomial],
        *variable_symbols,
        auxiliary,
        order="grevlex",
        domain=sympy.QQ,
    )
    return IdealRadicalMembershipResult(in_radical=len(basis) == 1 and basis[0] == 1)


def compute_ideal_quotient(request: IdealQuotientRequest) -> IdealQuotientResult:
    """Compute an exact ideal quotient through the bounded Singular backend."""

    backend = run_singular_ideal_operation(
        "quotient",
        request.dividend,
        request.divisor,
        request.resource_budget,
    )
    return IdealQuotientResult(
        outcome=backend.outcome,
        quotient=backend.ideal,
        backend_version=backend.backend_version,
        detail=backend.detail,
    )


def compute_ideal_saturation(request: IdealSaturationRequest) -> IdealSaturationResult:
    """Compute I : <d>^infinity through the bounded Singular backend."""

    denominator = RationalPolynomialIdeal(
        variables=request.denominator.variables,
        generators=(request.denominator,),
    )
    backend = run_singular_ideal_operation(
        "saturation",
        request.ideal,
        denominator,
        request.resource_budget,
    )
    return IdealSaturationResult(
        outcome=backend.outcome,
        saturation=backend.ideal,
        backend_version=backend.backend_version,
        detail=backend.detail,
    )


__all__ = [
    "compute_ideal_quotient",
    "compute_ideal_radical",
    "compute_ideal_radical_membership",
    "compute_ideal_saturation",
]


def compute_groebner_basis(request: GroebnerBasisRequest) -> GroebnerBasisResult:
    """Compute a reduced Gröbner basis for a bounded ideal over QQ using SymPy."""
    from jacobian.math.commutative_algebra_ops._models import (
        MAX_OUTPUT_TERMS,
        GroebnerBasisResult,
    )
    from jacobian.math.polynomials.values import (
        RationalPolynomial,
        RationalPolynomialIdeal,
    )

    variables = request.ideal.variables
    order_map = {"lex": "lex", "grlex": "grlex", "grevlex": "grevlex"}
    order = order_map.get(request.monomial_order, "grevlex")
    payload = {
        "mode": "groebner",
        "variables": list(variables),
        "order": order,
        "maximum_terms": MAX_OUTPUT_TERMS,
        "generators": [
            generator.model_dump(mode="json")
            for generator in request.ideal.generators
        ],
    }

    # The unbounded search runs in a killable worker under the declared
    # wall-time budget; result assembly and source-bound verification then
    # operate only on the declared output limits.
    try:
        result_payload = _run_sympy_kernel(
            payload, request.resource_budget.wall_seconds
        )
    except _SympyKernelTimeoutError:
        return GroebnerBasisResult(
            request=request,
            outcome="TIMEOUT",
            monomial_order=request.monomial_order,
            detail=(
                "groebner computation exceeded the enforced "
                f"{request.resource_budget.wall_seconds}s budget"
            ),
        )
    except _ResultLimitExceededError as error:
        return GroebnerBasisResult(
            request=request,
            outcome="LIMIT_EXCEEDED",
            monomial_order=request.monomial_order,
            detail=(
                "the exact reduced Gröbner basis exceeds the declared "
                f"exact-result limit: {error}"
            ),
        )
    except _SympyKernelError as error:
        return GroebnerBasisResult(
            request=request,
            outcome="ERROR",
            monomial_order=request.monomial_order,
            detail=(
                "the bounded Groebner kernel failed without producing an "
                f"exact basis: {error}"
            ),
        )

    basis_generators = [
        RationalPolynomial.model_validate(item)
        for item in result_payload["generators"]
    ]
    if not basis_generators:
        from jacobian.math.polynomials.values import SparseRationalPolynomial

        zero = RationalPolynomial(
            variables=variables,
            polynomial=SparseRationalPolynomial(terms=()),
        )
        basis_generators.append(zero)

    ideal = RationalPolynomialIdeal(
        variables=variables,
        generators=tuple(basis_generators),
    )

    return GroebnerBasisResult(
        request=request,
        basis=ideal,
        generator_count=len(basis_generators),
        monomial_order=request.monomial_order,
    )


def compute_ideal_normal_form(request: IdealNormalFormRequest) -> IdealNormalFormResult:
    """Reduce one polynomial modulo an ideal using a Gröbner basis remainder."""
    from jacobian.math.commutative_algebra_ops._models import IdealNormalFormResult
    from jacobian.math.polynomials.values import RationalPolynomial

    payload = {
        "mode": "normal_form",
        "variables": list(request.ideal.variables),
        "generators": [
            generator.model_dump(mode="json")
            for generator in request.ideal.generators
        ],
        "polynomial": request.polynomial.model_dump(mode="json"),
    }

    # A conservative 10-second budget bounds the killable kernel that runs
    # the unbounded Gröbner search; the remainder conversion and its
    # source-bound replay then operate on declared output limits.
    try:
        result_payload = _run_sympy_kernel(payload, 10)
    except _SympyKernelTimeoutError:
        return IdealNormalFormResult(
            request=request,
            outcome="TIMEOUT",
            detail=(
                "the Gröbner reduction exceeded the enforced 10s wall-time "
                "bound"
            ),
        )
    except _ResultLimitExceededError as error:
        return IdealNormalFormResult(
            request=request,
            outcome="LIMIT_EXCEEDED",
            detail=(
                "the exact normal form exceeds the declared exact-result "
                f"limit: {error}"
            ),
        )
    except _SympyKernelError as error:
        return IdealNormalFormResult(
            request=request,
            outcome="ERROR",
            detail=(
                "the bounded reduction kernel failed without producing an "
                f"exact remainder: {error}"
            ),
        )

    remainder_poly = RationalPolynomial.model_validate(result_payload["remainder"])
    in_ideal = len(remainder_poly.polynomial.terms) == 0
    return IdealNormalFormResult(
        request=request,
        remainder=remainder_poly,
        in_ideal=in_ideal,
    )


def compute_elimination_ideal(
    request: EliminationIdealRequest,
) -> EliminationIdealResult:
    """Compute the elimination ideal I ∩ QQ[remaining variables] using a lex Gröbner basis."""
    from jacobian._exact import CanonicalRational
    from jacobian.math.commutative_algebra_ops._models import EliminationIdealResult
    from jacobian.math.polynomials.values import (
        RationalPolynomial,
        RationalPolynomialIdeal,
        RationalPolynomialTerm,
        SparseRationalPolynomial,
    )

    variables = list(request.ideal.variables)
    payload = {
        "mode": "elimination",
        "variables": variables,
        "eliminated": list(request.eliminated_variables),
        "generators": [
            generator.model_dump(mode="json")
            for generator in request.ideal.generators
        ],
    }

    # The unbounded lex search runs in a killable worker under the declared
    # wall-time budget; canonicalization and source-bound verification then
    # operate only on the declared output limits.
    try:
        result_payload = _run_sympy_kernel(
            payload, request.resource_budget.wall_seconds
        )
    except _SympyKernelTimeoutError:
        return EliminationIdealResult(
            request=request,
            outcome="TIMEOUT",
            eliminated_variables=tuple(request.eliminated_variables),
            detail=(
                "the lex Gröbner elimination exceeded the enforced "
                f"{request.resource_budget.wall_seconds}s wall-time budget"
            ),
        )
    except _ResultLimitExceededError as error:
        return EliminationIdealResult(
            request=request,
            outcome="LIMIT_EXCEEDED",
            eliminated_variables=tuple(request.eliminated_variables),
            detail=(
                "the exact elimination ideal exceeds the declared "
                f"exact-result limit: {error}"
            ),
        )
    except _SympyKernelError as error:
        return EliminationIdealResult(
            request=request,
            outcome="ERROR",
            eliminated_variables=tuple(request.eliminated_variables),
            detail=(
                "the bounded elimination kernel failed without producing "
                f"an exact ideal: {error}"
            ),
        )

    remaining_tuple = tuple(result_payload["remaining"])
    if result_payload["unit_ideal"]:
        one = RationalPolynomial(
            variables=remaining_tuple,
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational(num="1", den="1"),
                        exponents=(0,) * len(remaining_tuple),
                    ),
                )
            ),
        )
        elimination_generators = [one]
    elif not result_payload["generators"]:
        zero = RationalPolynomial(
            variables=remaining_tuple,
            polynomial=SparseRationalPolynomial(terms=()),
        )
        elimination_generators = [zero]
    else:
        elimination_generators = [
            RationalPolynomial.model_validate(item)
            for item in result_payload["generators"]
        ]

    ideal = RationalPolynomialIdeal(
        variables=remaining_tuple,
        generators=tuple(elimination_generators),
    )

    return EliminationIdealResult(
        request=request,
        elimination_ideal=ideal,
        eliminated_variables=tuple(request.eliminated_variables),
    )



