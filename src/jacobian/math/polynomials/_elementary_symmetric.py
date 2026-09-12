"""Exact elementary-symmetric polynomial families over ``QQ``.

The public value is an ordinary family of canonical sparse polynomials.  The
operation owns admission for the complete family because its result contains
every requested degree, while the dynamic-product kernel avoids enumerating a
separate subset list for each polynomial.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from re import fullmatch

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._execution import OperationWorkLedger, request_checkpoint
from jacobian._models import StrictModel
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_TERMS,
    MAX_POLYNOMIAL_VARIABLES,
    PolynomialVariable,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

# These are semantic carrier budgets, not transport byte limits.  The largest
# shared polynomial axis has eight variables, so even the complete family has
# only 2**8 monomials.  The explicit limits keep the controlling quantities
# visible and protect this operation if the shared carrier is widened later.
MAX_ELEMENTARY_FAMILY_MONOMIALS = MAX_POLYNOMIAL_TERMS
MAX_ELEMENTARY_FAMILY_EXPONENT_CELLS = MAX_POLYNOMIAL_TERMS * MAX_POLYNOMIAL_VARIABLES
MAX_ELEMENTARY_FAMILY_LABEL_BYTES = 4_096
MAX_ELEMENTARY_FAMILY_RESULT_CELLS = 8_192
MAX_ELEMENTARY_FAMILY_WORK = 65_536


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"polynomial.symmetric.elementary.{reason}", message)


class ElementarySymmetricFamilyRequest(StrictModel):
    """An ordered variable axis and the largest requested elementary degree."""

    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=0,
        max_length=MAX_POLYNOMIAL_VARIABLES,
        description=(
            "Distinct ordered QQ variables. The family preserves this axis; "
            "maximum_degree must not exceed its length."
        ),
        json_schema_extra={"uniqueItems": True},
    )
    maximum_degree: StrictInt = Field(
        ge=0,
        le=MAX_POLYNOMIAL_VARIABLES,
        description=(
            "Largest degree to return, including e_0 = 1; it must be at most "
            "the number of variables."
        ),
        examples=[2],
    )

    @model_validator(mode="after")
    def require_degree_and_axis(self) -> ElementarySymmetricFamilyRequest:
        if len(set(self.variables)) != len(self.variables):
            raise _validation_error(
                "duplicate_variables",
                "elementary symmetric variables must be unique",
            )
        if self.maximum_degree > len(self.variables):
            raise _validation_error(
                "degree_exceeds_axis",
                "maximum_degree cannot exceed the variable count",
            )
        return self


class ElementarySymmetricFamilyResult(StrictModel):
    """The complete canonical family ``(e_0, ..., e_k)`` on one QQ axis."""

    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=0, max_length=MAX_POLYNOMIAL_VARIABLES
    )
    maximum_degree: StrictInt = Field(
        ge=0,
        le=MAX_POLYNOMIAL_VARIABLES,
    )
    polynomials: tuple[RationalPolynomial, ...] = Field(
        min_length=1,
        max_length=MAX_POLYNOMIAL_VARIABLES + 1,
    )

    @model_validator(mode="after")
    def require_axis_and_family_shape(self) -> ElementarySymmetricFamilyResult:
        if len(set(self.variables)) != len(self.variables):
            raise _validation_error(
                "result_duplicate_variables",
                "result variables must be unique",
            )
        if self.maximum_degree > len(self.variables):
            raise _validation_error(
                "result_degree_exceeds_axis",
                "result maximum_degree cannot exceed the variable count",
            )
        if len(self.polynomials) != self.maximum_degree + 1:
            raise _validation_error(
                "result_family_shape",
                "result must contain exactly one polynomial for every degree 0..k",
            )
        if any(
            polynomial.variables != self.variables for polynomial in self.polynomials
        ):
            raise _validation_error(
                "result_axis_mismatch",
                "every family polynomial must preserve the declared variable axis",
            )
        e_zero = self.polynomials[0].polynomial.terms
        if len(e_zero) != 1 or (
            e_zero[0].exponents != (0,) * len(self.variables)
            or e_zero[0].coefficient.as_fraction() != 1
        ):
            raise _validation_error(
                "result_e_zero",
                "the first family polynomial must be the canonical constant one",
            )
        return self


@dataclass(frozen=True, slots=True)
class _ElementaryFamilyBounds:
    """One admitted execution plan, reused by the kernel without recomputation."""

    monomials: int
    exponent_cells: int
    label_bytes: int
    result_cells: int
    recurrence_work: int
    work: int


def _bounds(
    variables_axis: tuple[PolynomialVariable, ...], maximum_degree: int
) -> _ElementaryFamilyBounds:
    variables = len(variables_axis)
    monomials = sum(comb(variables, degree) for degree in range(maximum_degree + 1))
    exponent_cells = monomials * variables
    # Every returned polynomial retains the full authoritative axis.
    label_bytes = (maximum_degree + 1) * sum(
        len(variable.encode("utf-8")) for variable in variables_axis
    )
    # A term has n exponent cells, one coefficient, and one sparse-term record;
    # each polynomial has its axis and family slot.  This is a structural
    # result-size bound, not a serialized JSON/transport byte ceiling.
    result_cells = (
        label_bytes
        + monomials * (variables + 3)
        + (maximum_degree + 1) * (variables + 2)
    )
    recurrence_work = sum(
        comb(index, degree)
        for index in range(variables)
        for degree in range(min(index, maximum_degree) + 1)
    )
    # The recurrence reads each retained prefix term once and copies each
    # retained prefix level once before adding the new-variable terms.
    work = 2 * recurrence_work + exponent_cells + monomials + maximum_degree + 1
    bounds = _ElementaryFamilyBounds(
        monomials=monomials,
        exponent_cells=exponent_cells,
        label_bytes=label_bytes,
        result_cells=result_cells,
        recurrence_work=recurrence_work,
        work=work,
    )
    if bounds.monomials > MAX_ELEMENTARY_FAMILY_MONOMIALS:
        raise OperationResourceAdmissionError(
            location=("maximum_degree",),
            code="polynomial.symmetric.elementary.monomial_bound",
            message="elementary family exceeds the admitted monomial bound",
        )
    if bounds.exponent_cells > MAX_ELEMENTARY_FAMILY_EXPONENT_CELLS:
        raise OperationResourceAdmissionError(
            location=("maximum_degree",),
            code="polynomial.symmetric.elementary.exponent_cells_bound",
            message="elementary family exceeds the admitted exponent-cell bound",
        )
    if bounds.label_bytes > MAX_ELEMENTARY_FAMILY_LABEL_BYTES:
        raise OperationResourceAdmissionError(
            location=("variables",),
            code="polynomial.symmetric.elementary.label_bound",
            message="repeated variable-axis labels exceed the admitted bound",
        )
    if bounds.result_cells > MAX_ELEMENTARY_FAMILY_RESULT_CELLS:
        raise OperationResourceAdmissionError(
            location=("maximum_degree",),
            code="polynomial.symmetric.elementary.result_bound",
            message="elementary family exceeds the admitted canonical result bound",
        )
    if bounds.work > MAX_ELEMENTARY_FAMILY_WORK:
        raise OperationResourceAdmissionError(
            location=("maximum_degree",),
            code="polynomial.symmetric.elementary.work_bound",
            message="elementary family exceeds the admitted work bound",
        )
    return bounds


def _compute(
    variables: tuple[PolynomialVariable, ...],
    maximum_degree: int,
    bounds: _ElementaryFamilyBounds,
) -> ElementarySymmetricFamilyResult:
    """Run the admitted dynamic-product recurrence.

    After processing ``x_1, ..., x_i``, ``levels[j]`` is ``e_j`` on that
    prefix.  Extending one variable applies
    ``e_j(new) = e_j(old) + x_i e_(j-1)(old)``.  No symbolic expression is
    parsed or expanded by a backend, and every emitted polynomial is already
    in the shared canonical sparse representation.
    """

    variable_count = len(variables)
    zero = (0,) * variable_count
    one = CanonicalRational(num=1, den=1)
    levels: list[dict[tuple[int, ...], CanonicalRational]] = [{zero: one}]
    ledger = OperationWorkLedger(bounds.work)

    for index in range(variable_count):
        request_checkpoint("during elementary symmetric recurrence")
        previous = levels
        ledger.charge(sum(len(level) for level in previous))
        levels = [dict(level) for level in previous]
        if len(levels) <= maximum_degree:
            levels.append({})
        for degree in range(1, min(index + 1, maximum_degree) + 1):
            source = previous[degree - 1]
            target = levels[degree]
            for exponents in source:
                ledger.charge()
                shifted = (*exponents[:index], 1, *exponents[index + 1 :])
                target[shifted] = one

    polynomials: list[RationalPolynomial] = []
    for degree in range(maximum_degree + 1):
        entries = levels[degree]
        terms: list[RationalPolynomialTerm] = []
        for exponents in sorted(entries, reverse=True):
            request_checkpoint("during elementary symmetric result construction")
            ledger.charge(len(exponents) + 1)
            terms.append(
                RationalPolynomialTerm(
                    coefficient=entries[exponents],
                    exponents=exponents,
                )
            )
        ledger.charge()
        polynomials.append(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(terms=tuple(terms)),
            )
        )

    return ElementarySymmetricFamilyResult(
        variables=variables,
        maximum_degree=maximum_degree,
        polynomials=tuple(polynomials),
    )


def _elementary_symmetric_family_from_request(
    request: ElementarySymmetricFamilyRequest,
) -> ElementarySymmetricFamilyResult:
    bounds = _bounds(request.variables, request.maximum_degree)
    return _compute(request.variables, request.maximum_degree, bounds)


def _validate_native_arguments(
    variables: tuple[PolynomialVariable, ...], maximum_degree: int
) -> None:
    if type(variables) is not tuple:
        raise TypeError("variables must be a tuple of distinct variable labels")
    if type(maximum_degree) is not int:
        raise TypeError("maximum_degree must be an integer")
    if len(variables) > MAX_POLYNOMIAL_VARIABLES:
        raise ValueError(
            f"variables cannot contain more than {MAX_POLYNOMIAL_VARIABLES} labels"
        )
    for variable in variables:
        if (
            type(variable) is not str
            or fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", variable) is None
        ):
            raise ValueError("variables must use the canonical polynomial label syntax")
    if len(set(variables)) != len(variables):
        raise ValueError("elementary symmetric variables must be unique")
    if not 0 <= maximum_degree <= len(variables):
        raise ValueError("maximum_degree must be between 0 and the variable count")


def elementary_symmetric_family(
    variables: tuple[PolynomialVariable, ...], maximum_degree: int
) -> ElementarySymmetricFamilyResult:
    """Return ``e_0, ..., e_k`` for one ordered QQ variable axis.

    This direct native API accepts mathematical arguments rather than the
    catalog's wire request model.  The private request adapter below is used
    only by catalog dispatch, so both paths share one admission and kernel.
    """

    _validate_native_arguments(variables, maximum_degree)
    bounds = _bounds(variables, maximum_degree)
    return _compute(variables, maximum_degree, bounds)


__all__ = [
    "MAX_ELEMENTARY_FAMILY_EXPONENT_CELLS",
    "MAX_ELEMENTARY_FAMILY_LABEL_BYTES",
    "MAX_ELEMENTARY_FAMILY_MONOMIALS",
    "MAX_ELEMENTARY_FAMILY_RESULT_CELLS",
    "MAX_ELEMENTARY_FAMILY_WORK",
    "ElementarySymmetricFamilyRequest",
    "ElementarySymmetricFamilyResult",
    "elementary_symmetric_family",
]
