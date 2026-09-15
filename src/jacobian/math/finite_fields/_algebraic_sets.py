"""Exact affine/projective algebraic sets over finite fields (#3726).

Closed bounded immutable values with preflight enumeration budgets and
exact finite-field arithmetic only. No solver, search DSL, or CAS session.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Self

from pydantic import Field
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_fields.values import (
    Axis,
    FiniteFieldElement,
    FiniteFieldPresentation,
    ProjectivePoint,
    _encoded_coordinates,
)

MAX_ALGEBRAIC_SET_VARS = 8
MAX_ALGEBRAIC_SET_EQUATIONS = 32
MAX_ALGEBRAIC_SET_TERMS = 256
MAX_ALGEBRAIC_SET_DEGREE = 32
MAX_ALGEBRAIC_SET_POINTS = 65_536
MAX_ALGEBRAIC_SET_WORK = 1_000_000


def _error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


class AlgebraicMonomial(StrictModel):
    """One coefficient times a monomial in the system variable axis."""

    coefficient: FiniteFieldElement
    exponents: tuple[int, ...] = Field(min_length=1, max_length=MAX_ALGEBRAIC_SET_VARS)

    @classmethod
    def _from_kernel(
        cls, *, coefficient: FiniteFieldElement, exponents: tuple[int, ...]
    ) -> Self:
        return cls.model_construct(coefficient=coefficient, exponents=exponents)


class AlgebraicPolynomial(StrictModel):
    """A sparse multivariate polynomial bound to one presentation and axis."""

    presentation: FiniteFieldPresentation
    variable_axis: Axis
    terms: tuple[AlgebraicMonomial, ...] = Field(
        min_length=1, max_length=MAX_ALGEBRAIC_SET_TERMS
    )

    @property
    def nvars(self) -> int:
        return len(self.variable_axis.labels)

    @property
    def total_degrees(self) -> tuple[int, ...]:
        return tuple(sum(term.exponents) for term in self.terms)

    @property
    def is_homogeneous(self) -> bool:
        degrees = self.total_degrees
        return all(degree == degrees[0] for degree in degrees)

    @classmethod
    def _from_kernel(
        cls,
        *,
        presentation: FiniteFieldPresentation,
        variable_axis: Axis,
        terms: tuple[AlgebraicMonomial, ...],
    ) -> Self:
        return cls.model_construct(
            presentation=presentation, variable_axis=variable_axis, terms=terms
        )


class PolynomialSystem(StrictModel):
    """A finite system of multivariate polynomials over one field and axis."""

    presentation: FiniteFieldPresentation
    variable_axis: Axis
    equations: tuple[AlgebraicPolynomial, ...] = Field(
        min_length=1, max_length=MAX_ALGEBRAIC_SET_EQUATIONS
    )

    @property
    def nvars(self) -> int:
        return len(self.variable_axis.labels)

    @classmethod
    def _from_kernel(
        cls,
        *,
        presentation: FiniteFieldPresentation,
        variable_axis: Axis,
        equations: tuple[AlgebraicPolynomial, ...],
    ) -> Self:
        return cls.model_construct(
            presentation=presentation,
            variable_axis=variable_axis,
            equations=equations,
        )


class AffinePoint(StrictModel):
    """One affine point bound to a presentation and variable axis."""

    presentation: FiniteFieldPresentation
    variable_axis: Axis
    coordinates: tuple[FiniteFieldElement, ...]

    @classmethod
    def _from_kernel(
        cls,
        *,
        presentation: FiniteFieldPresentation,
        variable_axis: Axis,
        coordinates: tuple[FiniteFieldElement, ...],
    ) -> Self:
        return cls.model_construct(
            presentation=presentation,
            variable_axis=variable_axis,
            coordinates=coordinates,
        )


class FieldEmbedding(StrictModel):
    """A structural exact field embedding claim (source generator image).

    The root relation ``m_source(image) == 0`` is established by the consuming
    base-change operation, not by structural decoding.
    """

    source: FiniteFieldPresentation
    target: FiniteFieldPresentation
    generator_image: FiniteFieldElement

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: FiniteFieldPresentation,
        target: FiniteFieldPresentation,
        generator_image: FiniteFieldElement,
    ) -> Self:
        return cls.model_construct(
            source=source, target=target, generator_image=generator_image
        )


def _require_canonical_polynomial(
    polynomial: AlgebraicPolynomial,
) -> None:
    nvars = len(polynomial.variable_axis.labels)
    if not 1 <= nvars <= MAX_ALGEBRAIC_SET_VARS:
        raise OperationDomainValidationError(
            location=("variable_axis",),
            code="finite_field.algebraic_set_variable_bound",
            message="algebraic-set variable axis length is outside its bound",
        )
    if len(set(polynomial.variable_axis.labels)) != nvars:
        raise OperationDomainValidationError(
            location=("variable_axis",),
            code="finite_field.algebraic_set_axis_labels_unique",
            message="variable axis labels must be unique",
        )
    seen: set[tuple[int, ...]] = set()
    for index, term in enumerate(polynomial.terms):
        if len(term.exponents) != nvars:
            raise OperationDomainValidationError(
                location=("terms", index),
                code="finite_field.algebraic_set_exponent_axis_mismatch",
                message="monomial exponents must match the variable axis",
            )
        if any(
            type(exponent) is not int or not 0 <= exponent <= MAX_ALGEBRAIC_SET_DEGREE
            for exponent in term.exponents
        ):
            raise OperationDomainValidationError(
                location=("terms", index),
                code="finite_field.algebraic_set_degree_bound",
                message="monomial exponents exceed the degree bound",
            )
        if term.coefficient.presentation != polynomial.presentation:
            raise OperationDomainValidationError(
                location=("terms", index),
                code="finite_field.algebraic_set_coefficient_parent",
                message="coefficients must share the system presentation",
            )
        if term.exponents in seen:
            raise OperationDomainValidationError(
                location=("terms", index),
                code="finite_field.algebraic_set_duplicate_exponent",
                message="monomial exponents must be unique",
            )
        seen.add(term.exponents)
    if any(term.coefficient.is_zero for term in polynomial.terms) and not (
        len(polynomial.terms) == 1 and polynomial.terms[0].coefficient.is_zero
    ):
        raise OperationDomainValidationError(
            location=("terms",),
            code="finite_field.algebraic_set_zero_coefficient",
            message="zero coefficients are allowed only for the lone zero-polynomial term",
        )


def _require_system(system: PolynomialSystem) -> None:
    nvars = len(system.variable_axis.labels)
    if not 1 <= nvars <= MAX_ALGEBRAIC_SET_VARS:
        raise OperationDomainValidationError(
            location=("variable_axis",),
            code="finite_field.algebraic_set_variable_bound",
            message="algebraic-set variable axis length is outside its bound",
        )
    for index, equation in enumerate(system.equations):
        if equation.presentation != system.presentation:
            raise OperationDomainValidationError(
                location=("equations", index),
                code="finite_field.algebraic_set_equation_parent",
                message="every equation must share the system presentation",
            )
        if equation.variable_axis != system.variable_axis:
            raise OperationDomainValidationError(
                location=("equations", index),
                code="finite_field.algebraic_set_equation_axis",
                message="every equation must share the system variable axis",
            )
        _require_canonical_polynomial(equation)


def _admit_enumeration(system: PolynomialSystem, *, projective: bool) -> int:
    _require_system(system)
    order = system.presentation.order
    nvars = system.nvars
    ambient = order**nvars
    if ambient > MAX_ALGEBRAIC_SET_POINTS or ambient < 1:
        raise OperationResourceAdmissionError(
            location=("presentation", "variable_axis"),
            code="finite_field.algebraic_set_ambient_bound",
            message="ambient enumeration exceeds the admitted point bound",
        )
    term_count = sum(len(equation.terms) for equation in system.equations)
    work = ambient * max(1, term_count) * max(1, nvars)
    if work > MAX_ALGEBRAIC_SET_WORK:
        raise OperationResourceAdmissionError(
            location=("equations",),
            code="finite_field.algebraic_set_work_bound",
            message="zero-set evaluation exceeds the admitted work bound",
        )
    if (
        projective
        # Projective enumeration reuses the same affine ambient pass; the
        # canonical-class quotient only reduces retained output.
        and (ambient - 1) // max(1, order - 1) > MAX_ALGEBRAIC_SET_POINTS
    ):
        raise OperationResourceAdmissionError(
            location=("presentation", "variable_axis"),
            code="finite_field.algebraic_set_projective_bound",
            message="projective enumeration exceeds the admitted point bound",
        )
    return int(ambient)


def _evaluate_at(
    equation: AlgebraicPolynomial,
    point: tuple[Any, ...],
    *,
    active_context: Any,
    zero: Any,
) -> Any:
    from jacobian.math.finite_fields import _flint as flint

    total: Any = zero
    for term in equation.terms:
        backend_coeff = flint.to_backend(
            term.coefficient, active_context=active_context
        )
        monomial: Any | None = None
        for coordinate, exponent in zip(point, term.exponents, strict=True):
            if exponent:
                power = coordinate**exponent
                monomial = power if monomial is None else monomial * power
        total = total + (
            backend_coeff if monomial is None else backend_coeff * monomial
        )
    return total


def affine_zero_set(system: PolynomialSystem) -> tuple[AffinePoint, ...]:
    """Return all and only simultaneous zeros in canonical coordinate order."""

    _admit_enumeration(system, projective=False)
    from jacobian.math.finite_fields import _flint as flint
    from jacobian.math.finite_fields.operations import _field_elements

    active_context = flint.context(system.presentation)
    zero = active_context(0)
    field_elements = _field_elements(system.presentation)
    points: list[AffinePoint] = []
    for coordinates in product(field_elements, repeat=system.nvars):
        backends = tuple(
            flint.to_backend(value, active_context=active_context)
            for value in coordinates
        )
        if all(
            _evaluate_at(equation, backends, active_context=active_context, zero=zero)
            == zero
            for equation in system.equations
        ):
            points.append(
                AffinePoint._from_kernel(
                    presentation=system.presentation,
                    variable_axis=system.variable_axis,
                    coordinates=coordinates,
                )
            )
    # Canonical order: lexicographic by encoded coordinates.
    points.sort(key=lambda point: tuple(_encoded_coordinates(c) for c in point.coordinates))
    return tuple(points)


def affine_zero_count(system: PolynomialSystem) -> int:
    """Compact count agreeing exactly with the complete enumeration."""

    return len(affine_zero_set(system))


def _require_homogeneous(system: PolynomialSystem) -> None:
    for index, equation in enumerate(system.equations):
        if not equation.is_homogeneous:
            raise OperationDomainValidationError(
                location=("equations", index),
                code="finite_field.projective_system_must_be_homogeneous",
                message="projective zero sets require a homogeneous system",
            )


def projective_zero_set(system: PolynomialSystem) -> tuple[ProjectivePoint, ...]:
    """Return canonical scalar-class representatives of the projective zeros."""

    _admit_enumeration(system, projective=True)
    _require_homogeneous(system)
    from jacobian.math.finite_fields import _flint as flint
    from jacobian.math.finite_fields import _sympy as sympy
    from jacobian.math.finite_fields.operations import _field_elements

    active_context = flint.context(system.presentation)
    zero = active_context(0)
    field_elements = _field_elements(system.presentation)
    seen: set[tuple[tuple[int, ...], ...]] = set()
    representatives: list[ProjectivePoint] = []
    for coordinates in product(field_elements, repeat=system.nvars):
        if all(value.is_zero for value in coordinates):
            continue
        backends = tuple(
            flint.to_backend(value, active_context=active_context)
            for value in coordinates
        )
        if not all(
            _evaluate_at(equation, backends, active_context=active_context, zero=zero)
            == zero
            for equation in system.equations
        ):
            continue
        normalized = sympy.normalize_projective_coordinates(
            system.presentation, coordinates
        )
        key = tuple(normalized)
        if key in seen:
            continue
        seen.add(key)
        representatives.append(
            ProjectivePoint(
                presentation=system.presentation,
                axis=system.variable_axis,
                coordinates=tuple(
                    FiniteFieldElement(
                        presentation=system.presentation, coordinates=coords
                    )
                    for coords in normalized
                ),
            )
        )
    representatives.sort(key=lambda point: tuple(_encoded_coordinates(c) for c in point.coordinates))
    return tuple(representatives)


def projective_zero_count(system: PolynomialSystem) -> int:
    """Compact projective count agreeing exactly with the enumeration."""

    return len(projective_zero_set(system))


def _require_embedding_shape(embedding: FieldEmbedding) -> None:
    if embedding.generator_image.presentation != embedding.target:
        raise OperationDomainValidationError(
            location=("generator_image",),
            code="finite_field.embedding_image_parent",
            message="generator image must use the target presentation",
        )
    if embedding.source.characteristic != embedding.target.characteristic:
        raise OperationDomainValidationError(
            location=("source", "target"),
            code="finite_field.embedding_characteristic_mismatch",
            message="field embeddings require equal characteristics",
        )
    if embedding.target.degree % embedding.source.degree != 0:
        raise OperationDomainValidationError(
            location=("source", "target"),
            code="finite_field.embedding_degree_divisibility",
            message="source degree must divide target degree",
        )


def _check_embedding_root(embedding: FieldEmbedding) -> None:
    """Establish m_source(image) == 0 in the target (consumer admission)."""

    from jacobian.math.finite_fields import _flint as flint

    _require_embedding_shape(embedding)
    target_context = flint.context(embedding.target)
    image_backend = flint.to_backend(
        embedding.generator_image, active_context=target_context
    )
    # Lift source modulus coefficients (base-prime residues) to target constants.
    result = target_context(0)
    power = target_context(1)
    for coefficient in embedding.source.modulus_coefficients:
        result = result + target_context(int(coefficient)) * power
        power = power * image_backend
    if result != target_context(0):
        raise OperationDomainValidationError(
            location=("generator_image",),
            code="finite_field.embedding_generator_not_root",
            message="generator image must satisfy the source modulus",
        )


def _embed_element(
    element: FiniteFieldElement, embedding: FieldEmbedding, *, target_context: Any
) -> FiniteFieldElement:
    from jacobian.math.finite_fields import _flint as flint

    image_backend = flint.to_backend(
        embedding.generator_image, active_context=target_context
    )
    result = target_context(0)
    power = target_context(1)
    for coordinate in element.coordinates:
        result = result + target_context(int(coordinate)) * power
        power = power * image_backend
    return FiniteFieldElement(
        presentation=embedding.target,
        coordinates=flint.coordinates(result, degree=embedding.target.degree),
    )


def base_change_system(
    system: PolynomialSystem, embedding: FieldEmbedding
) -> PolynomialSystem:
    """Transport a system along an explicit exact field embedding."""

    _require_system(system)
    if system.presentation != embedding.source:
        raise OperationDomainValidationError(
            location=("system", "embedding"),
            code="finite_field.base_change_source_mismatch",
            message="system presentation must equal the embedding source",
        )
    _check_embedding_root(embedding)
    from jacobian.math.finite_fields import _flint as flint

    target_context = flint.context(embedding.target)
    equations: list[AlgebraicPolynomial] = []
    for equation in system.equations:
        terms = tuple(
            AlgebraicMonomial._from_kernel(
                coefficient=_embed_element(
                    term.coefficient, embedding, target_context=target_context
                ),
                exponents=term.exponents,
            )
            for term in equation.terms
        )
        equations.append(
            AlgebraicPolynomial._from_kernel(
                presentation=embedding.target,
                variable_axis=equation.variable_axis,
                terms=terms,
            )
        )
    transported = PolynomialSystem._from_kernel(
        presentation=embedding.target,
        variable_axis=system.variable_axis,
        equations=tuple(equations),
    )
    _require_system(transported)
    return transported


def verify_affine_zero_set(
    system: PolynomialSystem, points: tuple[AffinePoint, ...]
) -> bool:
    try:
        return tuple(affine_zero_set(system)) == tuple(points)
    except (OperationDomainValidationError, OperationResourceAdmissionError):
        raise
    except Exception:
        return False


__all__ = [
    "MAX_ALGEBRAIC_SET_DEGREE",
    "MAX_ALGEBRAIC_SET_EQUATIONS",
    "MAX_ALGEBRAIC_SET_POINTS",
    "MAX_ALGEBRAIC_SET_TERMS",
    "MAX_ALGEBRAIC_SET_VARS",
    "MAX_ALGEBRAIC_SET_WORK",
    "AffinePoint",
    "AlgebraicMonomial",
    "AlgebraicPolynomial",
    "FieldEmbedding",
    "PolynomialSystem",
    "affine_zero_count",
    "affine_zero_set",
    "base_change_system",
    "projective_zero_count",
    "projective_zero_set",
    "verify_affine_zero_set",
]
