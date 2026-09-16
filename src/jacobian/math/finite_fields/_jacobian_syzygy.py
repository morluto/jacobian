"""Exact char-p Jacobian syzygy checks over bounded quotient rings (#964).

The operation derives the characteristic-p Jacobian of ``f`` itself; callers
supply only candidate syzygy rows. Reduction is exact and bounded without a
general Groebner engine: the ideal is empty (the plain polynomial ring) or
principal with a generator monic in one declared reduction variable, giving a
unique normal form by rewriting that variable's powers. Presentations outside
this regime are rejected as typed admission errors.
"""

from __future__ import annotations

from collections.abc import Sequence

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_fields._admission import require_field
from jacobian.math.finite_fields._algebraic_sets import (
    AlgebraicMonomial,
    AlgebraicPolynomial,
)
from jacobian.math.finite_fields._jacobian_syzygy_models import (
    MAX_JACOBIAN_SYZYGY_DEGREE,
    MAX_JACOBIAN_SYZYGY_IDEAL_GENERATORS,
    MAX_JACOBIAN_SYZYGY_REDUCTION_STEPS,
    MAX_JACOBIAN_SYZYGY_ROWS,
    MAX_JACOBIAN_SYZYGY_TERMS,
    MAX_JACOBIAN_SYZYGY_VARS,
    MAX_JACOBIAN_SYZYGY_WORK,
    JacobianSyzygyCheckResult,
    JacobianSyzygyRowReduction,
)
from jacobian.math.finite_fields.values import (
    Axis,
    FiniteFieldElement,
    FiniteFieldPresentation,
)

type _Terms = dict[tuple[int, ...], int]


def _domain_error(
    location: tuple[str | int, ...], code: str, message: str
) -> OperationDomainValidationError:
    return OperationDomainValidationError(location=location, code=code, message=message)


def _resource_error(
    location: tuple[str | int, ...], code: str, message: str
) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=location, code=code, message=message
    )


def _require_envelope(polynomial: AlgebraicPolynomial, location: str) -> None:
    if len(polynomial.terms) > MAX_JACOBIAN_SYZYGY_TERMS:
        raise _resource_error(
            (location, "terms"),
            "finite_field.jacobian_syzygy_term_bound",
            "a polynomial exceeds the Jacobian-syzygy term bound",
        )
    for term in polynomial.terms:
        if any(exponent > MAX_JACOBIAN_SYZYGY_DEGREE for exponent in term.exponents):
            raise _resource_error(
                (location, "terms"),
                "finite_field.jacobian_syzygy_degree_bound",
                "a monomial exceeds the Jacobian-syzygy degree bound",
            )


def _require_canonical(polynomial: AlgebraicPolynomial, location: str) -> None:
    """Admit one caller-authored sparse polynomial in canonical form."""

    _require_envelope(polynomial, location)
    nvars = len(polynomial.variable_axis.labels)
    seen: set[tuple[int, ...]] = set()
    for index, term in enumerate(polynomial.terms):
        if len(term.exponents) != nvars:
            raise _domain_error(
                (location, "terms", index),
                "finite_field.jacobian_syzygy_exponent_axis_mismatch",
                "monomial exponents must match the variable axis",
            )
        if any(
            type(exponent) is not int or exponent < 0 for exponent in term.exponents
        ):
            raise _domain_error(
                (location, "terms", index),
                "finite_field.jacobian_syzygy_exponent_nonnegative",
                "monomial exponents must be nonnegative integers",
            )
        if term.exponents in seen:
            raise _domain_error(
                (location, "terms", index),
                "finite_field.jacobian_syzygy_duplicate_exponent",
                "monomial exponents must be unique",
            )
        seen.add(term.exponents)
    if any(term.coefficient.is_zero for term in polynomial.terms) and not (
        len(polynomial.terms) == 1 and polynomial.terms[0].coefficient.is_zero
    ):
        raise _domain_error(
            (location, "terms"),
            "finite_field.jacobian_syzygy_zero_coefficient",
            "zero coefficients are allowed only for the lone zero-polynomial term",
        )


def _require_shared_parent(
    polynomial: AlgebraicPolynomial,
    presentation: FiniteFieldPresentation,
    axis: Axis,
    location: tuple[str | int, ...],
) -> None:
    if polynomial.presentation != presentation or polynomial.variable_axis != axis:
        raise _domain_error(
            location,
            "finite_field.jacobian_syzygy_shared_parent",
            "every polynomial must share the source presentation and variable axis",
        )


def _to_terms(polynomial: AlgebraicPolynomial) -> _Terms:
    return {
        term.exponents: int(term.coefficient.coordinates[0])
        for term in polynomial.terms
        if not term.coefficient.is_zero
    }


def _from_terms(
    terms: _Terms, presentation: FiniteFieldPresentation, axis: Axis
) -> AlgebraicPolynomial:
    if not terms:
        zero = FiniteFieldElement(
            presentation=presentation, coordinates=(0,) * presentation.degree
        )
        return AlgebraicPolynomial._from_kernel(
            presentation=presentation,
            variable_axis=axis,
            terms=(
                AlgebraicMonomial._from_kernel(
                    coefficient=zero, exponents=(0,) * len(axis.labels)
                ),
            ),
        )
    canonical = tuple(
        AlgebraicMonomial._from_kernel(
            coefficient=FiniteFieldElement(
                presentation=presentation, coordinates=(coefficient,)
            ),
            exponents=exponents,
        )
        for exponents, coefficient in sorted(terms.items(), reverse=True)
        if coefficient
    )
    return AlgebraicPolynomial._from_kernel(
        presentation=presentation, variable_axis=axis, terms=canonical
    )


def _add_into(target: _Terms, source: _Terms, prime: int) -> None:
    for exponents, coefficient in source.items():
        accumulated = (target.get(exponents, 0) + coefficient) % prime
        if accumulated:
            target[exponents] = accumulated
        else:
            target.pop(exponents, None)


def _multiply(left: _Terms, right: _Terms, prime: int) -> _Terms:
    product: _Terms = {}
    for left_exponents, left_coefficient in left.items():
        for right_exponents, right_coefficient in right.items():
            exponents = tuple(
                first + second
                for first, second in zip(left_exponents, right_exponents, strict=True)
            )
            accumulated = (
                product.get(exponents, 0) + left_coefficient * right_coefficient
            ) % prime
            if accumulated:
                product[exponents] = accumulated
            else:
                product.pop(exponents, None)
    return product


def _differentiate(terms: _Terms, index: int, prime: int) -> _Terms:
    """Formal characteristic-p derivative with respect to one axis variable."""

    derivative: _Terms = {}
    for exponents, coefficient in terms.items():
        exponent = exponents[index]
        if exponent == 0 or exponent % prime == 0:
            continue
        reduced = list(exponents)
        reduced[index] = exponent - 1
        key = tuple(reduced)
        accumulated = (derivative.get(key, 0) + coefficient * exponent) % prime
        if accumulated:
            derivative[key] = accumulated
        else:
            derivative.pop(key, None)
    return derivative


def _monic_tail(terms: _Terms, reduction_index: int) -> tuple[int, _Terms]:
    """Return (d, h) for a generator g == z^d + h in the reduction variable."""

    degree = max((exponents[reduction_index] for exponents in terms), default=0)
    top = [exponents for exponents in terms if exponents[reduction_index] == degree]
    if (
        degree < 1
        or len(top) != 1
        or any(
            position != reduction_index and exponent
            for position, exponent in enumerate(top[0])
        )
        or terms[top[0]] != 1
    ):
        raise _domain_error(
            ("ideal_generators", 0),
            "finite_field.jacobian_syzygy_monic_generator",
            "the principal generator must be monic in the reduction variable: "
            "its top power must appear exactly once, with unit coefficient and "
            "no other variables",
        )
    tail = dict(terms)
    del tail[top[0]]
    return degree, tail


def _normal_form(
    terms: _Terms,
    *,
    tail: _Terms | None,
    reduction_degree: int,
    reduction_index: int,
    prime: int,
) -> tuple[_Terms, int]:
    """Reduce by z^d == -h; unique normal form, exact step accounting."""

    if tail is None:
        return dict(terms), 0
    working = dict(terms)
    steps = 0
    while True:
        candidates = [
            exponents
            for exponents in working
            if exponents[reduction_index] >= reduction_degree
        ]
        if not candidates:
            return working, steps
        exponents = max(candidates)
        coefficient = working.pop(exponents)
        steps += 1
        if steps > MAX_JACOBIAN_SYZYGY_REDUCTION_STEPS:
            raise _resource_error(
                ("rows",),
                "finite_field.jacobian_syzygy_reduction_step_bound",
                "quotient reduction exceeds the admitted step bound",
            )
        base = list(exponents)
        base[reduction_index] -= reduction_degree
        for tail_exponents, tail_coefficient in tail.items():
            shifted = tuple(
                first + second
                for first, second in zip(base, tail_exponents, strict=True)
            )
            accumulated = (
                working.get(shifted, 0) - coefficient * tail_coefficient
            ) % prime
            if accumulated:
                working[shifted] = accumulated
            else:
                working.pop(shifted, None)


def _admit_sources(
    polynomial: AlgebraicPolynomial,
    rows: Sequence[Sequence[AlgebraicPolynomial]],
    ideal_generators: Sequence[AlgebraicPolynomial],
) -> int:
    """Admit the envelope and canonical shape; return the prime."""

    presentation = polynomial.presentation
    axis = polynomial.variable_axis
    nvars = len(axis.labels)
    if not 1 <= nvars <= MAX_JACOBIAN_SYZYGY_VARS:
        raise _resource_error(
            ("polynomial", "variable_axis"),
            "finite_field.jacobian_syzygy_variable_bound",
            "the variable axis exceeds the Jacobian-syzygy variable bound",
        )
    if len(set(axis.labels)) != nvars:
        raise _domain_error(
            ("polynomial", "variable_axis"),
            "finite_field.jacobian_syzygy_axis_labels_unique",
            "variable axis labels must be unique",
        )
    if not 1 <= len(rows) <= MAX_JACOBIAN_SYZYGY_ROWS:
        raise _resource_error(
            ("rows",),
            "finite_field.jacobian_syzygy_row_bound",
            "candidate rows exceed the Jacobian-syzygy row bound",
        )
    if len(ideal_generators) > MAX_JACOBIAN_SYZYGY_IDEAL_GENERATORS:
        raise _resource_error(
            ("ideal_generators",),
            "finite_field.jacobian_syzygy_ideal_generator_bound",
            "the admitted reduction regime carries at most one principal generator",
        )
    if presentation.degree != 1:
        raise _domain_error(
            ("polynomial", "presentation"),
            "finite_field.jacobian_syzygy_prime_field_only",
            "Jacobian syzygies are bound to a prime field GF(p) presentation",
        )
    require_field(presentation)

    _require_canonical(polynomial, "polynomial")
    for index, row in enumerate(rows):
        if len(row) != nvars:
            raise _domain_error(
                ("rows", index),
                "finite_field.jacobian_syzygy_row_length",
                "every syzygy row must have one entry per declared variable",
            )
        for entry_index, entry in enumerate(row):
            _require_shared_parent(
                entry, presentation, axis, ("rows", index, entry_index)
            )
            _require_canonical(entry, f"rows.{index}.{entry_index}")
    for index, generator in enumerate(ideal_generators):
        _require_shared_parent(
            generator, presentation, axis, ("ideal_generators", index)
        )
        _require_canonical(generator, f"ideal_generators.{index}")
    return presentation.characteristic


def _admit_reduction_regime(
    axis: Axis,
    ideal_generators: Sequence[AlgebraicPolynomial],
    reduction_variable: str | None,
) -> tuple[int, _Terms | None, int]:
    """Admit the bounded reduction regime; return (index, tail, degree)."""

    if not ideal_generators:
        if reduction_variable is not None:
            raise _domain_error(
                ("reduction_variable",),
                "finite_field.jacobian_syzygy_reduction_variable",
                "an empty ideal must not declare a reduction variable",
            )
        return 0, None, 0
    if reduction_variable is None:
        raise _domain_error(
            ("reduction_variable",),
            "finite_field.jacobian_syzygy_reduction_variable",
            "a nonempty principal ideal requires a declared reduction variable",
        )
    if reduction_variable not in axis.labels:
        raise _domain_error(
            ("reduction_variable",),
            "finite_field.jacobian_syzygy_reduction_variable",
            "the reduction variable must be a label of the variable axis",
        )
    reduction_index = axis.labels.index(reduction_variable)
    generator_terms = _to_terms(ideal_generators[0])
    if not generator_terms:
        raise _domain_error(
            ("ideal_generators", 0),
            "finite_field.jacobian_syzygy_monic_generator",
            "the zero polynomial is not a monic principal generator",
        )
    reduction_degree, tail = _monic_tail(generator_terms, reduction_index)
    return reduction_index, tail, reduction_degree


def check_jacobian_syzygy(
    *,
    polynomial: AlgebraicPolynomial,
    rows: Sequence[Sequence[AlgebraicPolynomial]],
    ideal_generators: Sequence[AlgebraicPolynomial] = (),
    reduction_variable: str | None = None,
) -> JacobianSyzygyCheckResult:
    """Check candidate rows against the derived char-p Jacobian modulo the ideal."""

    prime = _admit_sources(polynomial, rows, ideal_generators)
    presentation = polynomial.presentation
    axis = polynomial.variable_axis
    nvars = len(axis.labels)
    reduction_index, tail, reduction_degree = _admit_reduction_regime(
        axis, ideal_generators, reduction_variable
    )

    source_terms = _to_terms(polynomial)
    jacobian_terms = tuple(
        _differentiate(source_terms, index, prime) for index in range(nvars)
    )

    # Preflight: bound every product expansion and reduction pass before any
    # reduction runs. Each product term needs at most one reduction step per
    # reduction-degree multiple, and each step expands by the generator tail.
    tail_terms = len(tail) if tail is not None else 0
    work = 0
    for row in rows:
        for entry, partial in zip(row, jacobian_terms, strict=True):
            product_terms = max(1, len(_to_terms(entry))) * max(1, len(partial))
            work += (
                product_terms
                * (1 + MAX_JACOBIAN_SYZYGY_DEGREE // max(1, reduction_degree or 1))
                * max(1, tail_terms)
            )
    if work > MAX_JACOBIAN_SYZYGY_WORK:
        raise _resource_error(
            ("rows",),
            "finite_field.jacobian_syzygy_work_bound",
            "row-by-Jacobian reduction exceeds the admitted work bound",
        )

    ledger: list[JacobianSyzygyRowReduction] = []
    for row_index, row in enumerate(rows):
        combined: _Terms = {}
        for entry, partial in zip(row, jacobian_terms, strict=True):
            _add_into(combined, _multiply(_to_terms(entry), partial, prime), prime)
        remainder_terms, steps = _normal_form(
            combined,
            tail=tail,
            reduction_degree=reduction_degree,
            reduction_index=reduction_index,
            prime=prime,
        )
        ledger.append(
            JacobianSyzygyRowReduction._from_kernel(
                row_index=row_index,
                remainder=_from_terms(remainder_terms, presentation, axis),
                reduction_steps=steps,
            )
        )

    return JacobianSyzygyCheckResult._from_kernel(
        polynomial=polynomial,
        ideal_generators=tuple(ideal_generators),
        reduction_variable=reduction_variable,
        jacobian=tuple(
            _from_terms(partial, presentation, axis) for partial in jacobian_terms
        ),
        rows=tuple(tuple(row) for row in rows),
        ledger=tuple(ledger),
    )


__all__ = [
    "check_jacobian_syzygy",
]
