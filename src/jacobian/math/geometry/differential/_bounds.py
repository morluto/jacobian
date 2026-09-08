"""Derived admission for exact rational coordinate Lie derivatives."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal, NoReturn

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.differential._execution import (
    begin_lie_derivative_deadline,
    require_lie_derivative_deadline,
)
from jacobian.math.geometry.differential._recognition_process import (
    RationalFunctionRecognitionCandidate,
    canonical_recognition_candidates,
    recognize_canonical_rational_functions,
)
from jacobian.math.geometry.differential.values import (
    MAX_RATIONAL_TENSOR_COEFFICIENT_DIGITS,
    MAX_RATIONAL_TENSOR_EXPONENT,
    MAX_RATIONAL_TENSOR_LOCUS_GUARDS,
    MAX_RATIONAL_TENSOR_POLYNOMIAL_TERMS,
    RationalCoordinateTensor,
    canonical_locus_guards,
)
from jacobian.math.polynomials.rational_functions._bounds import (
    BoundWorkCategory,
    FractionBound,
    RationalFunctionBoundLimits,
    _add_fractions,
    _differentiate_fraction,
    _fraction_bound,
    _multiply_fractions,
    _recognition_work_units,
    _remove_guaranteed_common_monomial,
    _validate_canonical_result_bound,
    _zero_fraction,
)
from jacobian.math.polynomials.values import (
    SparseRationalPolynomial,
)

MAX_LIE_DERIVATIVE_WORK_UNITS = 25_000_000
MAX_LIE_DERIVATIVE_RAW_POLYNOMIAL_TERMS = 4_096
MAX_LIE_DERIVATIVE_RAW_COEFFICIENT_DIGITS = 4_096

type LieWorkCategory = BoundWorkCategory

_LIE_WORK_CATEGORIES: tuple[LieWorkCategory, ...] = (
    "recognition",
    "source_conversion",
    "differentiation",
    "multiplication",
    "addition",
    "normalization",
)


@dataclass(frozen=True)
class FactorReference:
    owner: Literal["VECTOR", "TENSOR"]
    component: int
    derivative_axis: int | None


@dataclass(frozen=True)
class LieProductTerm:
    sign: Literal[-1, 1]
    left: FactorReference
    right: FactorReference


@dataclass(frozen=True)
class LieComponentPlan:
    terms: tuple[LieProductTerm, ...]
    raw_result: FractionBound
    canonical_coefficient_digits: int


@dataclass(frozen=True)
class LieDerivativePlan:
    components: tuple[LieComponentPlan, ...]
    recognition_candidates: tuple[RationalFunctionRecognitionCandidate, ...]
    inherited_locus_guards: tuple[SparseRationalPolynomial, ...]
    work_units_by_category: tuple[tuple[LieWorkCategory, int], ...]

    @property
    def work_units(self) -> int:
        return sum(amount for _, amount in self.work_units_by_category)


class _Ledger:
    def __init__(self, *, deadline: float) -> None:
        self.deadline = deadline
        self.limits = RationalFunctionBoundLimits(
            raw_terms=MAX_LIE_DERIVATIVE_RAW_POLYNOMIAL_TERMS,
            raw_digits=MAX_LIE_DERIVATIVE_RAW_COEFFICIENT_DIGITS,
            result_exponent=MAX_RATIONAL_TENSOR_EXPONENT,
            result_terms=MAX_RATIONAL_TENSOR_POLYNOMIAL_TERMS,
            result_digits=MAX_RATIONAL_TENSOR_COEFFICIENT_DIGITS,
            label="Lie-derivative",
            reject=_reject,
        )
        self.work_units = 0
        self._by_category: dict[LieWorkCategory, int] = dict.fromkeys(
            _LIE_WORK_CATEGORIES, 0
        )

    def charge(self, category: LieWorkCategory, amount: int) -> None:
        if amount < 0:
            raise AssertionError("Lie-derivative work charges must be nonnegative")
        require_lie_derivative_deadline(
            self.deadline, f"while charging {category} work"
        )
        self.work_units += amount
        self._by_category[category] += amount
        if self.work_units > MAX_LIE_DERIVATIVE_WORK_UNITS:
            _reject(
                "work_budget",
                "Lie-derivative exact arithmetic exceeds the "
                f"{MAX_LIE_DERIVATIVE_WORK_UNITS}-unit work budget",
            )

    @property
    def by_category(self) -> tuple[tuple[LieWorkCategory, int], ...]:
        return tuple(
            (category, self._by_category[category]) for category in _LIE_WORK_CATEGORIES
        )


def _reject(
    reason: str,
    message: str,
    *,
    location: tuple[str | int, ...] = (),
) -> NoReturn:
    raise OperationDomainValidationError(
        location=location,
        code=f"differential_geometry.lie_derivative.{reason}",
        message=message,
    )


def _component_offset(index: tuple[int, ...], dimension: int) -> int:
    offset = 0
    for coordinate in index:
        offset = offset * dimension + coordinate
    return offset


def _replace_index(
    index: tuple[int, ...], position: int, replacement: int
) -> tuple[int, ...]:
    return (*index[:position], replacement, *index[position + 1 :])


def build_lie_derivative_plan(
    vector_field: RationalCoordinateTensor,
    tensor: RationalCoordinateTensor,
    *,
    deadline: float | None = None,
) -> LieDerivativePlan:
    """Admit one complete Lie derivative and return its reusable plan."""

    if deadline is None:
        deadline = begin_lie_derivative_deadline()

    if vector_field.coordinate_axis != tensor.coordinate_axis:
        _reject(
            "coordinate_axis_mismatch",
            "vector field and tensor must use the same ordered coordinate axis",
            location=("vector_field", "coordinate_axis"),
        )
    if vector_field.variance != ("CONTRAVARIANT",):
        _reject(
            "vector_signature",
            "vector field must have rank one and CONTRAVARIANT variance",
            location=("vector_field", "variance"),
        )
    dimension = len(tensor.coordinate_axis)
    inherited_guards = canonical_locus_guards(
        vector_field.retained_nonzero_denominators,
        tensor.retained_nonzero_denominators,
        variable_count=dimension,
    )
    if len(inherited_guards) > MAX_RATIONAL_TENSOR_LOCUS_GUARDS:
        _reject(
            "result_locus_guards",
            "Lie-derivative retained locus exceeds the "
            f"{MAX_RATIONAL_TENSOR_LOCUS_GUARDS}-guard representation budget",
        )
    ledger = _Ledger(deadline=deadline)
    vector_bounds = tuple(
        _fraction_bound(value, ledger) for value in vector_field.components
    )
    tensor_bounds = tuple(_fraction_bound(value, ledger) for value in tensor.components)
    vector_derivatives = {
        (component, axis): _zero_fraction(dimension)
        if vector_bounds[component].is_zero
        else _differentiate_fraction(
            vector_field.components[component], vector_bounds[component], axis, ledger
        )
        for component in range(dimension)
        for axis in range(dimension)
    }
    component_plans: list[LieComponentPlan] = []
    for index in product(range(dimension), repeat=len(tensor.variance)):
        component = _component_offset(index, dimension)
        term_plans: list[LieProductTerm] = []
        result_bound = _zero_fraction(dimension)
        for axis in range(dimension):
            if vector_bounds[axis].is_zero:
                continue
            term = LieProductTerm(
                sign=1,
                left=FactorReference("VECTOR", axis, None),
                right=FactorReference("TENSOR", component, axis),
            )
            term_plans.append(term)
            result_bound = _add_fractions(
                result_bound,
                _multiply_fractions(
                    vector_bounds[axis],
                    _differentiate_fraction(
                        tensor.components[component],
                        tensor_bounds[component],
                        axis,
                        ledger,
                    ),
                    ledger,
                ),
                ledger,
            )
        for position, variance in enumerate(tensor.variance):
            component_index = index[position]
            for axis in range(dimension):
                replaced_component = _component_offset(
                    _replace_index(index, position, axis), dimension
                )
                if variance == "CONTRAVARIANT":
                    term = LieProductTerm(
                        sign=-1,
                        left=FactorReference("VECTOR", component_index, axis),
                        right=FactorReference("TENSOR", replaced_component, None),
                    )
                    left_bound = vector_derivatives[(component_index, axis)]
                else:
                    term = LieProductTerm(
                        sign=1,
                        left=FactorReference("VECTOR", axis, component_index),
                        right=FactorReference("TENSOR", replaced_component, None),
                    )
                    left_bound = vector_derivatives[(axis, component_index)]
                if left_bound.is_zero:
                    continue
                term_plans.append(term)
                result_bound = _add_fractions(
                    result_bound,
                    _multiply_fractions(
                        left_bound, tensor_bounds[replaced_component], ledger
                    ),
                    ledger,
                )
        result_bound = _remove_guaranteed_common_monomial(result_bound)
        coefficient_digits = _validate_canonical_result_bound(result_bound, ledger)
        component_plans.append(
            LieComponentPlan(
                terms=tuple(term_plans),
                raw_result=result_bound,
                canonical_coefficient_digits=coefficient_digits,
            )
        )

    plans = tuple(component_plans)
    possible_result_guards = sum(
        not component.raw_result.is_zero
        and any(component.raw_result.denominator.degrees)
        for component in plans
    )
    if (
        len(inherited_guards) + possible_result_guards
        > MAX_RATIONAL_TENSOR_LOCUS_GUARDS
    ):
        _reject(
            "result_locus_guards",
            "Lie-derivative retained locus can exceed the "
            f"{MAX_RATIONAL_TENSOR_LOCUS_GUARDS}-guard representation budget",
        )
    recognition_candidates = canonical_recognition_candidates(vector_field, tensor)
    for candidate in recognition_candidates:
        source_bounds = (
            vector_bounds if candidate.owner == "vector_field" else tensor_bounds
        )
        ledger.charge(
            "recognition",
            _recognition_work_units(source_bounds[candidate.component]),
        )
    plan = LieDerivativePlan(
        components=plans,
        recognition_candidates=recognition_candidates,
        inherited_locus_guards=inherited_guards,
        work_units_by_category=ledger.by_category,
    )
    recognition = recognize_canonical_rational_functions(
        plan.recognition_candidates,
        deadline=deadline,
    )
    if recognition.non_coprime is not None:
        failure = recognition.non_coprime
        _reject(
            "component_not_canonical",
            f"{failure.owner} components must have coprime canonical rational-function parts",
            location=(failure.owner, "components", failure.component),
        )
    require_lie_derivative_deadline(deadline, "after coprimality recognition")
    return plan


__all__ = [
    "MAX_LIE_DERIVATIVE_WORK_UNITS",
    "FactorReference",
    "LieComponentPlan",
    "LieDerivativePlan",
    "LieProductTerm",
    "build_lie_derivative_plan",
]
