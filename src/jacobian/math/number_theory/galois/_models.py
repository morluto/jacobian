"""Typed wire contracts for bounded Galois-theory operations."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.combinatorics.posets.core._models import FinitePoset
from jacobian.math.number_theory.number_fields._field_embedding import (
    SimpleNumberFieldEmbedding,
)
from jacobian.math.number_theory.number_fields.values import (
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
)
from jacobian.math.polynomials.values import MonicPolynomial, RationalPolynomial

MAX_FACTOR_DEGREE = 128
MAX_GALOIS_GROUP_DEGREE = 6
# This exact field carrier currently admits only Q and quadratic extensions.
MAX_SPLITTING_FIELD_DEGREE = 2
MAX_FIELD_ORDER = 251
GaloisCoefficient = Annotated[int, Field(ge=-(10**12), le=10**12, strict=True)]
PositiveFactorDegree = Annotated[
    int,
    Field(ge=1, le=MAX_FACTOR_DEGREE, strict=True),
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"galois_theory.{reason}", message)


def _require_prime(value: int) -> None:
    from sympy import isprime

    if not isprime(value):
        raise _validation_error("field_order_not_prime", "field_order must be prime")


def _supported_galois_polynomial(coefficients: tuple[int, ...]) -> None:
    from sympy import Poly, Symbol

    if not 2 <= len(coefficients) <= MAX_GALOIS_GROUP_DEGREE + 1:
        raise _validation_error(
            "degree_bound",
            "Galois computation requires degree one through six",
        )
    if any(
        type(coefficient) is not int or abs(coefficient) > 10**12
        for coefficient in coefficients
    ):
        raise _validation_error(
            "coefficient_bound",
            "Galois coefficients must be integers of magnitude at most 10^12",
        )
    if coefficients[-1] == 0:
        raise _validation_error(
            "leading_coefficient_zero", "leading coefficient must be nonzero"
        )
    polynomial = Poly.from_list(list(reversed(coefficients)), Symbol("x"), domain="QQ")
    if not polynomial.is_irreducible:
        raise _validation_error(
            "polynomial_not_irreducible",
            "SymPy galois_group requires an irreducible polynomial over QQ",
        )


class GaloisFactorRequest(StrictModel):
    """A nonzero, nonconstant polynomial over the prime field ``GF(p)``."""

    field_order: int = Field(ge=2, le=MAX_FIELD_ORDER, strict=True)
    coefficients: tuple[
        Annotated[int, Field(ge=0, le=MAX_FIELD_ORDER - 1, strict=True)], ...
    ] = Field(
        min_length=2,
        max_length=MAX_FACTOR_DEGREE + 1,
        description="Canonical GF(p) residues in ascending coefficient order, with a nonzero leading coefficient. Degree and modulus jointly determine work admission.",
    )


class FrobeniusCycleRequest(StrictModel):
    """A positive factor-degree partition of a polynomial degree."""

    field_order: int = Field(ge=2, le=MAX_FIELD_ORDER, strict=True)
    polynomial_degree: int = Field(ge=1, le=MAX_FACTOR_DEGREE, strict=True)
    factorization_degrees: tuple[PositiveFactorDegree, ...] = Field(
        min_length=1,
        max_length=MAX_FACTOR_DEGREE,
    )


class _SupportedGaloisPolynomialRequest(StrictModel):
    polynomial: RationalPolynomial = Field(
        description="Univariate polynomial over QQ with integer coefficients of magnitude at most 10^12 and degree one through six."
    )

    @model_validator(mode="after")
    def require_supported_encoding(self) -> Self:
        terms = self.polynomial.polynomial.terms
        if (
            len(self.polynomial.variables) != 1
            or not terms
            or not 1 <= terms[0].exponents[0] <= MAX_GALOIS_GROUP_DEGREE
        ):
            raise _validation_error(
                "degree_bound",
                "Galois computation requires a univariate polynomial of degree one through six",
            )
        if any(
            term.coefficient.den != 1
            or len(format_canonical_integer(abs(term.coefficient.num))) > 13
            or abs(term.coefficient.num) > 10**12
            for term in terms
        ):
            raise _validation_error(
                "coefficient_bound",
                "Galois coefficients must be integers of magnitude at most 10^12",
            )
        return self

    @property
    def coefficients(self) -> tuple[int, ...]:
        coefficients = [0] * (self.polynomial.polynomial.terms[0].exponents[0] + 1)
        for term in self.polynomial.polynomial.terms:
            coefficients[term.exponents[0]] = term.coefficient.num
        return tuple(coefficients)


class GaloisGroupRequest(_SupportedGaloisPolynomialRequest):
    """An irreducible degree-one-through-six polynomial over ``QQ``."""


class SolvableRequest(_SupportedGaloisPolynomialRequest):
    """The supported SymPy domain for deciding radical solvability."""


def _discriminant_coefficients(polynomial: RationalPolynomial) -> tuple[int, ...]:
    """Admit a discriminant-profile polynomial and return ascending coefficients."""
    terms = polynomial.polynomial.terms
    if (
        len(polynomial.variables) != 1
        or not terms
        or not 1 <= terms[0].exponents[0] <= MAX_GALOIS_GROUP_DEGREE
    ):
        raise _validation_error(
            "degree_bound",
            "discriminant profile requires a univariate polynomial of degree one through six",
        )
    if any(
        term.coefficient.den != 1 or abs(term.coefficient.num) > 10**12
        for term in terms
    ):
        raise _validation_error(
            "coefficient_bound",
            "discriminant coefficients must be integers of magnitude at most 10^12",
        )
    degree = terms[0].exponents[0]
    coefficients = [0] * (degree + 1)
    for term in terms:
        coefficients[term.exponents[0]] = term.coefficient.num
    return tuple(coefficients)


class PolynomialDiscriminantRequest(StrictModel):
    """A bounded univariate polynomial over QQ, including reducible inputs."""

    polynomial: RationalPolynomial = Field(
        description="Univariate polynomial over QQ with integer coefficients of magnitude at most 10^12 and degree one through six."
    )

    @model_validator(mode="after")
    def require_supported_encoding(self) -> Self:
        _discriminant_coefficients(self.polynomial)
        return self

    @property
    def coefficients(self) -> tuple[int, ...]:
        return _discriminant_coefficients(self.polynomial)


class PolynomialDiscriminantResult(StrictModel):
    """Exact polynomial discriminant and its rational-square profile."""

    polynomial: RationalPolynomial
    discriminant: ExactInteger = Field(gt=-(10**160), lt=10**160)
    is_rational_square: bool

    @model_validator(mode="after")
    def require_square_profile(self) -> Self:
        from math import isqrt

        value = self.discriminant
        expected = value >= 0 and isqrt(value) ** 2 == value
        if self.is_rational_square is not expected:
            raise _validation_error(
                "discriminant_square_profile",
                "rational-square flag must match the exact integer discriminant",
            )
        return self


class FiniteFieldFactor(StrictModel):
    """One monic irreducible factor with positive multiplicity."""

    coefficients: tuple[int, ...] = Field(
        min_length=2,
        max_length=MAX_FACTOR_DEGREE + 1,
    )
    multiplicity: int = Field(ge=1, le=MAX_FACTOR_DEGREE, strict=True)


class GaloisFactorResult(StrictModel):
    field_order: int = Field(ge=2, le=MAX_FIELD_ORDER, strict=True)
    source_coefficients: tuple[int, ...] = Field(
        min_length=2,
        max_length=MAX_FACTOR_DEGREE + 1,
    )
    unit: int = Field(ge=1, le=MAX_FIELD_ORDER - 1, strict=True)
    factors: tuple[FiniteFieldFactor, ...] = Field(
        min_length=1,
        max_length=MAX_FACTOR_DEGREE,
    )

    @property
    def distinct_factor_count(self) -> int:
        return len(self.factors)

    @property
    def factor_count(self) -> int:
        return sum(factor.multiplicity for factor in self.factors)

    @property
    def is_irreducible(self) -> bool:
        return (
            len(self.factors) == 1
            and self.factors[0].multiplicity == 1
            and len(self.factors[0].coefficients) == len(self.source_coefficients)
        )

    @classmethod
    def _from_kernel(
        cls,
        *,
        field_order: int,
        source_coefficients: tuple[int, ...],
        unit: int,
        factors: tuple[FiniteFieldFactor, ...],
    ) -> Self:
        """Construct output whose complete factorization came from the kernel."""

        return cls.model_construct(
            field_order=field_order,
            source_coefficients=source_coefficients,
            unit=unit,
            factors=factors,
        )

    @model_validator(mode="after")
    def require_structural_consistency(self) -> Self:
        _require_factor_residues(self)
        return self


def _require_factor_residues(result: GaloisFactorResult) -> None:
    prime = result.field_order
    if result.unit >= prime:
        raise _validation_error(
            "unit_not_canonical",
            "factorization unit must be a canonical nonzero residue",
        )
    if any(not 0 <= coefficient < prime for coefficient in result.source_coefficients):
        raise _validation_error(
            "source_coefficients_not_canonical",
            "source coefficients must be canonical field residues",
        )
    for factor in result.factors:
        if any(not 0 <= coefficient < prime for coefficient in factor.coefficients):
            raise _validation_error(
                "factor_coefficients_not_canonical",
                "factor coefficients must be canonical field residues",
            )
        if factor.coefficients[-1] != 1:
            raise _validation_error(
                "factor_not_monic", "finite-field factors must be monic"
            )


class FrobeniusCycleResult(StrictModel):
    cycle_type: tuple[PositiveFactorDegree, ...]

    @property
    def degree(self) -> int:
        return sum(self.cycle_type)

    @property
    def is_irreducible(self) -> bool:
        return self.cycle_type == (self.degree,)

    @model_validator(mode="after")
    def require_canonical_partition(self) -> Self:
        if self.cycle_type != tuple(sorted(self.cycle_type, reverse=True)):
            raise _validation_error(
                "cycle_type_not_canonical",
                "cycle type must be sorted in descending order",
            )
        return self


class QQSplittingField(StrictModel):
    """An actual degree-at-most-two splitting field over QQ.

    ``extension`` is the exact quotient QQ[x]/(f) in the shared number-field
    carrier. Each root is a reduced coordinate vector in its power basis.
    Root order is canonical: rational roots increase; an irreducible
    quadratic lists the quotient generator, then its conjugate.
    """

    source: RationalPolynomial
    extension: SimpleNumberFieldPresentation
    root_values: tuple[SimpleNumberFieldElement, ...] = Field(
        min_length=1, max_length=2
    )
    root_multiplicities: tuple[StrictInt, ...] = Field(min_length=1, max_length=2)

    @property
    def degree(self) -> int:
        return self.extension.degree

    @model_validator(mode="after")
    def require_exact_root_parent_and_axis(self) -> Self:
        if self.degree > MAX_SPLITTING_FIELD_DEGREE:
            raise _validation_error(
                "splitting_field_degree_bound",
                "the exact splitting-field carrier supports extension degree at most two",
            )
        if len(self.root_values) != len(self.root_multiplicities):
            raise _validation_error(
                "root_multiplicity_axis",
                "each exact root value needs one multiplicity",
            )
        if any(root.presentation != self.extension for root in self.root_values):
            raise _validation_error(
                "root_parent",
                "every root coordinate must use the exact extension presentation",
            )
        if any(multiplicity < 1 for multiplicity in self.root_multiplicities):
            raise _validation_error(
                "root_multiplicity",
                "root multiplicities must be positive",
            )
        terms = self.source.polynomial.terms
        if (
            len(self.source.variables) != 1
            or not terms
            or terms[0].exponents[0] not in (1, 2)
        ):
            raise _validation_error(
                "splitting_field_source",
                "the exact splitting-field source must have degree one or two",
            )
        if sum(self.root_multiplicities) != terms[0].exponents[0]:
            raise _validation_error(
                "root_multiplicity_degree",
                "the root multiplicities must sum to the source polynomial degree",
            )
        return self


class QQRoot(StrictModel):
    field: QQSplittingField
    index: StrictInt = Field(ge=0)
    multiplicity: StrictInt = Field(ge=1)
    value: SimpleNumberFieldElement

    @model_validator(mode="after")
    def require_root_axis(self) -> Self:
        if self.index >= len(self.field.root_values):
            raise _validation_error(
                "root_index", "root index exceeds the retained root axis"
            )
        if (
            self.value.presentation != self.field.extension
            or self.value != self.field.root_values[self.index]
            or self.multiplicity != self.field.root_multiplicities[self.index]
        ):
            raise _validation_error(
                "root_value_binding",
                "root index, multiplicity, and exact value must match the field root axis",
            )
        return self


class QQFieldAutomorphism(StrictModel):
    field: QQSplittingField
    root_permutation: tuple[StrictInt, ...] = Field(min_length=1, max_length=2)
    basis_images: tuple[SimpleNumberFieldElement, ...] = Field(
        min_length=1, max_length=2
    )

    @model_validator(mode="after")
    def require_permutation(self) -> Self:
        axis = tuple(range(len(self.field.root_values)))
        if tuple(sorted(self.root_permutation)) != axis:
            raise _validation_error(
                "automorphism_permutation",
                "automorphism must permute the complete root axis",
            )
        if len(self.basis_images) != self.field.degree or any(
            image.presentation != self.field.extension for image in self.basis_images
        ):
            raise _validation_error(
                "automorphism_basis_map",
                "automorphism must provide one exact image per power-basis vector",
            )
        return self


def _supported_splitting_field_polynomial(polynomial: RationalPolynomial) -> None:
    """Admit an integral degree-one-or-two univariate polynomial over QQ."""
    terms = polynomial.polynomial.terms
    if (
        len(polynomial.variables) != 1
        or not terms
        or terms[0].exponents[0] not in (1, 2)
    ):
        raise _validation_error(
            "splitting_field_degree_bound",
            "exact splitting fields currently admit polynomial degree one or two",
        )
    if any(
        term.coefficient.den != 1 or abs(term.coefficient.num) > 10**12
        for term in terms
    ):
        raise _validation_error(
            "splitting_field_coefficient_bound",
            "splitting-field coefficients must be integers of magnitude at most 10^12",
        )


class SplittingFieldRequest(StrictModel):
    """An integral univariate polynomial over QQ of degree at most two."""

    polynomial: RationalPolynomial = Field(
        description="Univariate polynomial over QQ, degree one or two, with integral coefficients of magnitude at most 10^12."
    )

    @model_validator(mode="after")
    def require_supported_encoding(self) -> Self:
        _supported_splitting_field_polynomial(self.polynomial)
        return self

    @property
    def coefficients(self) -> tuple[int, ...]:
        degree = self.polynomial.polynomial.terms[0].exponents[0]
        coefficients = [0] * (degree + 1)
        for term in self.polynomial.polynomial.terms:
            coefficients[term.exponents[0]] = term.coefficient.num
        return tuple(coefficients)


class SplittingFieldResult(StrictModel):
    field: QQSplittingField
    roots: tuple[QQRoot, ...]
    source_coefficients: tuple[StrictInt, ...]
    factor_reconstruction: tuple[SimpleNumberFieldElement, ...]

    @model_validator(mode="after")
    def require_source_and_root_binding(self) -> Self:
        terms = self.field.source.polynomial.terms
        degree = terms[0].exponents[0]
        coefficients = [0] * (degree + 1)
        for term in terms:
            if term.coefficient.den != 1:
                raise _validation_error(
                    "splitting_field_coefficients",
                    "splitting-field source coefficients must be integral",
                )
            coefficients[term.exponents[0]] = term.coefficient.num
        expected = tuple(coefficients)
        if (
            self.source_coefficients != expected
            or len(self.factor_reconstruction) != degree + 1
        ):
            raise _validation_error(
                "splitting_field_reconstruction",
                "source coefficients and exact linear-factor reconstruction must match the source degree",
            )
        if tuple(root.field for root in self.roots) != (self.field,) * len(self.roots):
            raise _validation_error(
                "splitting_field_root_parent",
                "every root must retain the exact splitting-field parent",
            )
        if tuple(root.index for root in self.roots) != tuple(
            range(len(self.field.root_values))
        ):
            raise _validation_error(
                "splitting_field_root_axis",
                "roots must enumerate the complete source root axis",
            )
        if any(
            coefficient.presentation != self.field.extension
            for coefficient in self.factor_reconstruction
        ):
            raise _validation_error(
                "splitting_field_reconstruction_parent",
                "linear-factor reconstruction coefficients must use the exact extension",
            )
        return self


class GaloisCompositumRequest(StrictModel):
    """Two source-bound degree-at-most-two splitting fields over QQ."""

    left: QQSplittingField
    right: QQSplittingField


class GaloisCompositumResult(StrictModel):
    """A simple field with exact inclusions of both source splitting fields."""

    left: QQSplittingField
    right: QQSplittingField
    compositum: SimpleNumberFieldPresentation
    left_embedding: SimpleNumberFieldEmbedding
    right_embedding: SimpleNumberFieldEmbedding

    @model_validator(mode="after")
    def require_bound_embeddings(self) -> Self:
        if self.compositum.degree > 4:
            raise _validation_error(
                "compositum_degree_bound",
                "a compositum of the supported inputs has degree at most four",
            )
        if (
            self.left_embedding.source != self.left.extension
            or self.right_embedding.source != self.right.extension
            or self.left_embedding.target != self.compositum
            or self.right_embedding.target != self.compositum
        ):
            raise _validation_error(
                "compositum_embedding_parent",
                "compositum inclusions must bind both exact source fields and the result field",
            )
        return self


class AutomorphismRequest(StrictModel):
    field: QQSplittingField


class AutomorphismResult(StrictModel):
    field: QQSplittingField
    automorphisms: tuple[QQFieldAutomorphism, ...]


class AutomorphismComposeRequest(StrictModel):
    first: QQFieldAutomorphism
    second: QQFieldAutomorphism


class AutomorphismInverseRequest(StrictModel):
    automorphism: QQFieldAutomorphism


class AutomorphismApplyRequest(StrictModel):
    automorphism: QQFieldAutomorphism
    root: QQRoot


class AutomorphismApplyResult(StrictModel):
    root: QQRoot


class AutomorphismElementApplyRequest(StrictModel):
    """Apply one exact field automorphism to one element of its field."""

    automorphism: QQFieldAutomorphism
    element: SimpleNumberFieldElement


class ElementEmbeddingOrbitRequest(StrictModel):
    """An element of one supported splitting field over QQ."""

    field: QQSplittingField
    element: SimpleNumberFieldElement

    @model_validator(mode="after")
    def require_element_parent(self) -> Self:
        if self.element.presentation != self.field.extension:
            raise _validation_error(
                "element_orbit_parent",
                "element must belong to the exact splitting-field presentation",
            )
        return self


class ElementAutomorphismImage(StrictModel):
    """One exact map and its image of the orbit's source element."""

    automorphism: QQFieldAutomorphism
    image: SimpleNumberFieldElement

    @model_validator(mode="after")
    def require_map_image_parents(self) -> Self:
        if self.image.presentation != self.automorphism.field.extension:
            raise _validation_error(
                "element_orbit_image_parent",
                "automorphism image must lie in its exact target field",
            )
        return self


class GaloisAutomorphismSubgroup(StrictModel):
    """A subgroup of the exact automorphisms of one supported splitting field."""

    field: QQSplittingField
    elements: tuple[QQFieldAutomorphism, ...] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def require_one_parent_and_canonical_axis(self) -> Self:
        if any(element.field != self.field for element in self.elements):
            raise _validation_error(
                "subgroup_parent",
                "every subgroup automorphism must belong to the retained field",
            )
        permutations = tuple(element.root_permutation for element in self.elements)
        if permutations != tuple(sorted(set(permutations))):
            raise _validation_error(
                "subgroup_elements",
                "subgroup automorphisms must be unique and sorted by root action",
            )
        return self


class ElementEmbeddingOrbitResult(StrictModel):
    """Complete exact element action under the supported QQ automorphism group."""

    field: QQSplittingField
    source_element: SimpleNumberFieldElement
    action: tuple[ElementAutomorphismImage, ...] = Field(min_length=1, max_length=2)
    orbit: tuple[SimpleNumberFieldElement, ...] = Field(min_length=1, max_length=2)
    stabilizer: GaloisAutomorphismSubgroup
    orbit_size: StrictInt = Field(ge=1, le=2)
    minimal_polynomial: MonicPolynomial

    @model_validator(mode="after")
    def require_complete_bound_result(self) -> Self:
        if self.source_element.presentation != self.field.extension:
            raise _validation_error(
                "element_orbit_source_parent",
                "source element must belong to the retained splitting field",
            )
        if self.stabilizer.field != self.field:
            raise _validation_error(
                "element_orbit_stabilizer_parent",
                "stabilizer must belong to the retained splitting field",
            )
        if any(
            item.automorphism.field != self.field
            or item.image.presentation != self.field.extension
            for item in self.action
        ):
            raise _validation_error(
                "element_orbit_action_parent",
                "every action map and image must use the retained field",
            )
        if any(value.presentation != self.field.extension for value in self.orbit):
            raise _validation_error(
                "element_orbit_value_parent",
                "every orbit element must use the retained field",
            )
        if self.orbit_size != len(self.orbit) or self.orbit_size * len(
            self.stabilizer.elements
        ) != len(self.action):
            raise _validation_error(
                "element_orbit_cardinality",
                "orbit, action, and stabilizer cardinalities are inconsistent",
            )
        if len(set(self.orbit)) != len(self.orbit):
            raise _validation_error(
                "element_orbit_duplicates",
                "orbit elements must be distinct and canonically ordered",
            )
        return self


class GaloisSubgroupRequest(StrictModel):
    """A bounded candidate subset of a degree-at-most-two automorphism group."""

    field: QQSplittingField
    elements: tuple[QQFieldAutomorphism, ...] = Field(min_length=1, max_length=2)


class GaloisFixedFieldRequest(StrictModel):
    subgroup: GaloisAutomorphismSubgroup


class GaloisFixedFieldResult(StrictModel):
    subgroup: GaloisAutomorphismSubgroup
    fixed_field: SimpleNumberFieldPresentation
    inclusion: SimpleNumberFieldEmbedding

    @model_validator(mode="after")
    def require_embedding_parents(self) -> Self:
        if (
            self.inclusion.source != self.fixed_field
            or self.inclusion.target != self.subgroup.field.extension
        ):
            raise _validation_error(
                "fixed_field_parent",
                "fixed-field inclusion must map the reported field into the subgroup parent",
            )
        return self


class IntermediateFieldStabilizerRequest(StrictModel):
    field: QQSplittingField
    inclusion: SimpleNumberFieldEmbedding


class IntermediateFieldStabilizerResult(StrictModel):
    inclusion: SimpleNumberFieldEmbedding
    subgroup: GaloisAutomorphismSubgroup

    @model_validator(mode="after")
    def require_inclusion_parent(self) -> Self:
        if self.inclusion.target != self.subgroup.field.extension:
            raise _validation_error(
                "stabilizer_parent",
                "intermediate-field inclusion and stabilizer must share the exact extension parent",
            )
        return self


class GaloisCorrespondenceRequest(StrictModel):
    """The complete subgroup/embedded-subfield correspondence for ``L/QQ``."""

    field: QQSplittingField


class GaloisCorrespondencePair(StrictModel):
    """One exact pair ``H <-> L^H`` in a complete correspondence."""

    subgroup_label: Annotated[str, Field(pattern=r"^H[0-1]$", strict=True)]
    field_label: Annotated[str, Field(pattern=r"^F[0-1]$", strict=True)]
    subgroup: GaloisAutomorphismSubgroup
    inclusion: SimpleNumberFieldEmbedding
    stabilizer: GaloisAutomorphismSubgroup
    subgroup_order: ExactInteger = Field(ge=1, le=2)
    subgroup_index: ExactInteger = Field(ge=1, le=2)
    fixed_field_degree: ExactInteger = Field(ge=1, le=2)
    relative_field_degree: ExactInteger = Field(ge=1, le=2)
    normal: bool


class GaloisCorrespondenceResult(StrictModel):
    """Complete degree-at-most-two Galois correspondence, bound to ``L/QQ``.

    Each pair is indexed in both finite posets: its subgroup and embedded fixed
    field are the two objects identified by the Galois correspondence.
    """

    field: QQSplittingField
    pairs: Annotated[
        tuple[GaloisCorrespondencePair, ...], Field(min_length=1, max_length=2)
    ]
    subgroup_inclusion_poset: FinitePoset
    intermediate_field_inclusion_poset: FinitePoset
    normal_subgroup_labels: Annotated[tuple[str, ...], Field(max_length=2)]

    @model_validator(mode="after")
    def require_complete_correspondence_shape(self) -> Self:
        if len(self.pairs) != self.field.degree:
            raise _validation_error(
                "correspondence_cardinality",
                "the bounded correspondence must enumerate every subgroup of the supported Galois group",
            )
        subgroup_labels = tuple(f"H{index}" for index in range(len(self.pairs)))
        field_labels = tuple(f"F{index}" for index in range(len(self.pairs)))
        if (
            tuple(pair.subgroup_label for pair in self.pairs) != subgroup_labels
            or tuple(pair.field_label for pair in self.pairs) != field_labels
        ):
            raise _validation_error(
                "correspondence_pair_axis",
                "subgroup/fixed-field pairs must use their canonical parallel axes",
            )
        if any(
            pair.subgroup.field != self.field
            or pair.stabilizer != pair.subgroup
            or pair.inclusion.target != self.field.extension
            or pair.inclusion.source.degree != pair.fixed_field_degree
            or pair.subgroup_order != len(pair.subgroup.elements)
            or pair.subgroup_index * pair.subgroup_order != self.field.degree
            or pair.fixed_field_degree != pair.subgroup_index
            or pair.relative_field_degree != pair.subgroup_order
            or pair.fixed_field_degree * pair.relative_field_degree != self.field.degree
            or not pair.normal
            for pair in self.pairs
        ):
            raise _validation_error(
                "correspondence_degree_identity",
                "each pair must retain the exact parent, normal subgroup, and Galois degree/index identities",
            )
        if self.normal_subgroup_labels != subgroup_labels:
            raise _validation_error(
                "correspondence_normal_subgroups",
                "all subgroups of a degree-at-most-two Galois group are normal",
            )
        if (
            self.subgroup_inclusion_poset.elements != subgroup_labels
            or self.intermediate_field_inclusion_poset.elements != field_labels
        ):
            raise _validation_error(
                "correspondence_poset_axis",
                "the two inclusion posets must use the exact paired object axes",
            )
        subgroup_relations = {
            (pair.lower, pair.upper)
            for pair in self.subgroup_inclusion_poset.strict_order_pairs
        }
        field_relations = {
            (pair.lower, pair.upper)
            for pair in self.intermediate_field_inclusion_poset.strict_order_pairs
        }
        expected_subgroup_relations = (
            {(subgroup_labels[1], subgroup_labels[0])}
            if self.field.degree == 2
            else set()
        )
        expected_field_relations = (
            {(field_labels[0], field_labels[1])} if self.field.degree == 2 else set()
        )
        if (
            subgroup_relations != expected_subgroup_relations
            or field_relations != expected_field_relations
            or {
                (pair.lower, pair.upper)
                for pair in self.subgroup_inclusion_poset.cover_relations
            }
            != expected_subgroup_relations
            or {
                (pair.lower, pair.upper)
                for pair in self.intermediate_field_inclusion_poset.cover_relations
            }
            != expected_field_relations
        ):
            raise _validation_error(
                "correspondence_poset_order",
                "subgroup and intermediate-field inclusion posets must be complete and inclusion reversing",
            )
        return self


class GaloisRootAxis(StrictModel):
    """The ordered root positions of one retained source polynomial.

    Positions are canonical indices; the polynomial is the source that gives
    those positions their mathematical meaning.  The shape checks here only
    establish a well-formed axis.  Relation verification belongs to the
    public claim verifiers.
    """

    polynomial: RationalPolynomial
    indices: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_GALOIS_GROUP_DEGREE,
    )

    @model_validator(mode="after")
    def require_structural_axis(self) -> Self:
        terms = self.polynomial.polynomial.terms
        if len(self.polynomial.variables) != 1 or not terms:
            raise _validation_error(
                "root_axis_requires_univariate_polynomial",
                "root axis requires a nonempty univariate polynomial",
            )
        degree = terms[0].exponents[0]
        if len(self.indices) != degree or self.indices != tuple(range(degree)):
            raise _validation_error(
                "root_axis_not_canonical",
                "root positions must be the canonical degree-sized index axis",
            )
        return self


class FinitePermutationGroup(StrictModel):
    """A composable permutation group on one polynomial root axis."""

    root_axis: GaloisRootAxis
    generators: tuple[tuple[StrictInt, ...], ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def require_permutations_on_axis(self) -> Self:
        expected = tuple(range(len(self.root_axis.indices)))
        if any(
            len(generator) != len(expected) or tuple(sorted(generator)) != expected
            for generator in self.generators
        ):
            raise _validation_error(
                "generator_not_permutation",
                "every generator must permute the complete root axis",
            )
        return self


class GaloisGroupResult(StrictModel):
    group: FinitePermutationGroup
    group_name: str
    order: int = Field(ge=1)
    degree: int = Field(ge=1, le=MAX_GALOIS_GROUP_DEGREE)
    is_solvable: bool

    @property
    def polynomial(self) -> RationalPolynomial:
        """Return the canonical source retained by the root axis."""

        return self.group.root_axis.polynomial

    @classmethod
    def _from_kernel(
        cls,
        *,
        group: FinitePermutationGroup,
        group_name: str,
        order: int,
        degree: int,
        is_solvable: bool,
    ) -> Self:
        """Construct output whose group properties came from the kernel."""

        return cls.model_construct(
            group=group,
            group_name=group_name,
            order=order,
            degree=degree,
            is_solvable=is_solvable,
        )

    @model_validator(mode="after")
    def require_group_degree(self) -> Self:
        if self.degree != len(self.group.root_axis.indices):
            raise _validation_error(
                "group_degree_mismatch",
                "group root axis must match the polynomial degree",
            )
        return self


class SolvableResult(StrictModel):
    solvable_by_radicals: bool
    group: FinitePermutationGroup

    @property
    def polynomial(self) -> RationalPolynomial:
        """Return the canonical source retained by the root axis."""

        return self.group.root_axis.polynomial

    @classmethod
    def _from_kernel(
        cls,
        *,
        solvable_by_radicals: bool,
        group: FinitePermutationGroup,
    ) -> Self:
        """Construct output whose solvability came from the kernel."""

        return cls.model_construct(
            solvable_by_radicals=solvable_by_radicals,
            group=group,
        )


__all__ = [
    "MAX_SPLITTING_FIELD_DEGREE",
    "AutomorphismApplyRequest",
    "AutomorphismApplyResult",
    "AutomorphismComposeRequest",
    "AutomorphismElementApplyRequest",
    "AutomorphismInverseRequest",
    "AutomorphismRequest",
    "AutomorphismResult",
    "FiniteFieldFactor",
    "FinitePermutationGroup",
    "FrobeniusCycleRequest",
    "FrobeniusCycleResult",
    "GaloisCompositumRequest",
    "GaloisCompositumResult",
    "GaloisCorrespondencePair",
    "GaloisCorrespondenceRequest",
    "GaloisCorrespondenceResult",
    "GaloisFactorRequest",
    "GaloisFactorResult",
    "GaloisGroupRequest",
    "GaloisGroupResult",
    "GaloisRootAxis",
    "QQFieldAutomorphism",
    "QQRoot",
    "QQSplittingField",
    "SolvableRequest",
    "SolvableResult",
    "SplittingFieldRequest",
    "SplittingFieldResult",
]
