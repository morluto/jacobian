"""Canonical rational coordinate tensors and retained chart loci."""

from __future__ import annotations

from typing import Any, Literal, Self, TypeGuard

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import CanonicalLimits
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_VARIABLES,
    PolynomialVariable,
    RationalFunction,
    SparseRationalPolynomial,
    require_sparse_polynomial_budget,
)

type TensorVariance = Literal["CONTRAVARIANT", "COVARIANT"]

MAX_RATIONAL_TENSOR_RANK = 8
MAX_RATIONAL_TENSOR_COMPONENTS = 256
MAX_RATIONAL_TENSOR_LOCUS_GUARDS = 768
MAX_RATIONAL_TENSOR_POLYNOMIAL_TERMS = 256
MAX_RATIONAL_TENSOR_EXPONENT = 64
MAX_RATIONAL_TENSOR_COEFFICIENT_DIGITS = 128
_MAX_RATIONAL_TENSOR_INPUT_DEPTH = CanonicalLimits().max_depth


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"differential_geometry.{reason}", message)


def _is_json_sequence(value: Any) -> TypeGuard[list[Any] | tuple[Any, ...]]:
    return isinstance(value, (list, tuple))


def _require_bounded_tensor_input_depth(value: Any) -> None:
    """Check nested Python input iteratively before immutable conversion."""

    pending: list[tuple[Any, int, bool]] = [(value, 0, False)]
    active: set[int] = set()
    while pending:
        item, depth, exiting = pending.pop()
        if depth > _MAX_RATIONAL_TENSOR_INPUT_DEPTH:
            raise _validation_error(
                "tensor_input_depth",
                "rational coordinate tensor input exceeds the "
                f"{_MAX_RATIONAL_TENSOR_INPUT_DEPTH}-level nesting budget",
            )
        is_container = isinstance(item, dict) or _is_json_sequence(item)
        if not is_container:
            continue
        identity = id(item)
        if exiting:
            active.remove(identity)
            continue
        if identity in active:
            raise _validation_error(
                "tensor_input_cycle",
                "rational coordinate tensor input must be acyclic",
            )
        active.add(identity)
        pending.append((item, depth, True))
        nested_values = item.values() if isinstance(item, dict) else item
        pending.extend((nested, depth + 1, False) for nested in nested_values)


def _polynomial_key(
    polynomial: SparseRationalPolynomial,
) -> tuple[tuple[tuple[int, ...], str, str], ...]:
    return tuple(
        (term.exponents, term.coefficient.num, term.coefficient.den)
        for term in polynomial.terms
    )


def _is_unit_polynomial(
    polynomial: SparseRationalPolynomial, variable_count: int
) -> bool:
    return (
        len(polynomial.terms) == 1
        and polynomial.terms[0].coefficient.as_fraction() == 1
        and polynomial.terms[0].exponents == (0,) * variable_count
    )


def canonical_locus_guards(
    *families: tuple[SparseRationalPolynomial, ...],
    component_denominators: tuple[SparseRationalPolynomial, ...] = (),
    variable_count: int,
) -> tuple[SparseRationalPolynomial, ...]:
    """Return one sorted exact presentation of a retained nonvanishing locus.

    Each family denotes a conjunction of polynomial nonvanishing conditions.
    Unit denominators add no condition. Reducible guards remain intact: this
    presentation deliberately avoids factorization during request parsing.
    """

    guards = {
        _polynomial_key(polynomial): polynomial
        for family in families
        for polynomial in family
        if not _is_unit_polynomial(polynomial, variable_count)
    }
    for polynomial in component_denominators:
        if not _is_unit_polynomial(polynomial, variable_count):
            guards[_polynomial_key(polynomial)] = polynomial
    return tuple(guards[key] for key in sorted(guards))


class RationalCoordinateTensor(StrictModel):
    """One mixed tensor over a retained rational coordinate chart.

    Components are flattened in lexicographic index order, with the final
    tensor index varying fastest. A rank-zero tensor has one component. The
    nonzero-denominator family is an explicit presentation of the retained
    chart locus; it may contain inherited guards that no longer occur in the
    normalized component denominators.
    """

    coordinate_axis: tuple[PolynomialVariable, ...] = Field(
        min_length=1,
        max_length=MAX_POLYNOMIAL_VARIABLES,
        description="Ordered unique rational-chart coordinate axis.",
    )
    variance: tuple[TensorVariance, ...] = Field(
        max_length=MAX_RATIONAL_TENSOR_RANK,
        description=(
            "Ordered tensor-index variance signature. Components use "
            "lexicographic index order with the final index varying fastest."
        ),
    )
    components: tuple[RationalFunction, ...] = Field(
        min_length=1,
        max_length=MAX_RATIONAL_TENSOR_COMPONENTS,
    )
    retained_nonzero_denominators: tuple[SparseRationalPolynomial, ...] = Field(
        default=(),
        max_length=MAX_RATIONAL_TENSOR_LOCUS_GUARDS,
        description=(
            "Sorted unique monic nonconstant polynomials required to remain "
            "nonzero on the retained chart locus. The family includes every "
            "nonunit component denominator and may retain inherited guards."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def preflight_tensor_shape(cls, value: Any) -> Any:
        """Reject impossible dense shapes before nested field values parse."""

        if not isinstance(value, dict):
            _require_bounded_tensor_input_depth(value)
            return canonicalize_json_containers(value)
        axis = value.get("coordinate_axis")
        variance = value.get("variance")
        components = value.get("components")
        guards = value.get("retained_nonzero_denominators", ())
        if (
            _is_json_sequence(components)
            and len(components) > MAX_RATIONAL_TENSOR_COMPONENTS
        ):
            raise _validation_error(
                "tensor_component_budget",
                "rational coordinate tensor exceeds the "
                f"{MAX_RATIONAL_TENSOR_COMPONENTS}-component representation budget",
            )
        if _is_json_sequence(guards) and len(guards) > MAX_RATIONAL_TENSOR_LOCUS_GUARDS:
            raise _validation_error(
                "tensor_locus_guard_budget",
                "rational coordinate tensor exceeds the "
                f"{MAX_RATIONAL_TENSOR_LOCUS_GUARDS}-guard representation budget",
            )
        if _is_json_sequence(axis) and len(axis) > MAX_POLYNOMIAL_VARIABLES:
            raise _validation_error(
                "tensor_coordinate_axis_budget",
                "rational coordinate tensor exceeds the "
                f"{MAX_POLYNOMIAL_VARIABLES}-coordinate representation budget",
            )
        if _is_json_sequence(variance) and len(variance) > MAX_RATIONAL_TENSOR_RANK:
            raise _validation_error(
                "tensor_rank_budget",
                "rational coordinate tensor exceeds the "
                f"rank-{MAX_RATIONAL_TENSOR_RANK} representation budget",
            )
        if _is_json_sequence(axis) and _is_json_sequence(variance):
            dimension = len(axis)
            rank = len(variance)
            if rank <= MAX_RATIONAL_TENSOR_RANK and dimension:
                expected = dimension**rank
                if expected > MAX_RATIONAL_TENSOR_COMPONENTS:
                    raise _validation_error(
                        "tensor_component_budget",
                        "dense tensor shape requires more than "
                        f"{MAX_RATIONAL_TENSOR_COMPONENTS} components",
                    )
                if _is_json_sequence(components) and len(components) != expected:
                    raise _validation_error(
                        "tensor_component_shape",
                        f"rank-{rank} tensor on a {dimension}-coordinate axis "
                        f"requires exactly {expected} components",
                    )
        _require_bounded_tensor_input_depth(value)
        return canonicalize_json_containers(value)

    @model_validator(mode="after")
    def require_one_ordered_chart_and_locus(self) -> Self:
        dimension = len(self.coordinate_axis)
        if len(set(self.coordinate_axis)) != dimension:
            raise _validation_error(
                "tensor_coordinate_axis", "tensor coordinate labels must be unique"
            )
        expected = dimension ** len(self.variance)
        if len(self.components) != expected:
            raise _validation_error(
                "tensor_component_shape",
                f"tensor shape requires exactly {expected} components",
            )
        if any(
            component.variables != self.coordinate_axis for component in self.components
        ):
            raise _validation_error(
                "tensor_component_field",
                "every tensor component must use the complete ordered coordinate field",
            )

        keys: list[tuple[tuple[tuple[int, ...], str, str], ...]] = []
        for guard in self.retained_nonzero_denominators:
            if not guard.terms:
                raise _validation_error(
                    "tensor_locus_zero_guard",
                    "retained-locus guards must be nonzero polynomials",
                )
            if any(len(term.exponents) != dimension for term in guard.terms):
                raise _validation_error(
                    "tensor_locus_guard_axis",
                    "every retained-locus guard must match the coordinate axis",
                )
            try:
                require_sparse_polynomial_budget(
                    guard,
                    maximum_terms=MAX_RATIONAL_TENSOR_POLYNOMIAL_TERMS,
                    maximum_exponent=MAX_RATIONAL_TENSOR_EXPONENT,
                    maximum_coefficient_digits=MAX_RATIONAL_TENSOR_COEFFICIENT_DIGITS,
                    label="retained-locus guard",
                )
            except ValueError as exc:
                raise _validation_error("tensor_locus_guard_budget", str(exc)) from exc
            if guard.terms[0].coefficient.as_fraction() != 1:
                raise _validation_error(
                    "tensor_locus_guard_monic",
                    "retained-locus guards must be monic",
                )
            if _is_unit_polynomial(guard, dimension):
                raise _validation_error(
                    "tensor_locus_unit_guard",
                    "unit polynomials must be omitted from retained-locus guards",
                )
            keys.append(_polynomial_key(guard))
        if tuple(keys) != tuple(sorted(set(keys))):
            raise _validation_error(
                "tensor_locus_guard_order",
                "retained-locus guards must be unique and canonically sorted",
            )

        guard_keys = set(keys)
        for component in self.components:
            denominator = component.denominator
            if (
                not _is_unit_polynomial(denominator, dimension)
                and _polynomial_key(denominator) not in guard_keys
            ):
                raise _validation_error(
                    "tensor_locus_missing_denominator",
                    "every nonunit component denominator must occur in the "
                    "retained-locus guard family",
                )
        return self


__all__ = [
    "MAX_RATIONAL_TENSOR_COMPONENTS",
    "MAX_RATIONAL_TENSOR_LOCUS_GUARDS",
    "MAX_RATIONAL_TENSOR_RANK",
    "RationalCoordinateTensor",
    "TensorVariance",
    "canonical_locus_guards",
]
