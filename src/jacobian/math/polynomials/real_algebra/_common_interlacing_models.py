"""Typed source-bound contracts for exact common polynomial interlacing."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math._labels import OpaqueLabel
from jacobian.math.number_theory.algebraic_numbers.real import (
    RationalIsolatingInterval,
    RealAlgebraicValue,
    _UnrecognizedRealAlgebraicValue,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
)

MAX_COMMON_INTERLACING_FAMILY_SIZE = 8
MAX_COMMON_INTERLACING_SOURCE_DEGREE = 32
MAX_COMMON_INTERLACING_SOURCE_TERMS = MAX_COMMON_INTERLACING_SOURCE_DEGREE + 1
MAX_COMMON_INTERLACING_TOTAL_DEGREE = 128
MAX_COMMON_INTERLACING_TOTAL_TERMS = (
    MAX_COMMON_INTERLACING_TOTAL_DEGREE + MAX_COMMON_INTERLACING_FAMILY_SIZE
)
MAX_COMMON_INTERLACING_INPUT_DIGITS = 64


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"polynomial.common_interlacing_{reason}", message)


def _bound_raw_rational(value: object, *, label: str) -> None:
    if isinstance(value, CanonicalRational):
        components: tuple[object, object] = (value.num, value.den)
    elif isinstance(value, Mapping):
        components = (value.get("num"), value.get("den"))
    else:
        return
    for component in components:
        if isinstance(component, str) and (
            len(component.lstrip("-")) > MAX_COMMON_INTERLACING_INPUT_DIGITS
        ):
            raise _validation_error(
                "coefficient_digits",
                f"{label} exceeds the {MAX_COMMON_INTERLACING_INPUT_DIGITS}-digit input bound",
            )


def _raw_polynomial_parts(value: object) -> tuple[object, object]:
    if isinstance(value, RationalPolynomial):
        return value.variables, value.polynomial.terms
    if not isinstance(value, Mapping):
        return None, None
    sparse = value.get("polynomial")
    terms = sparse.get("terms") if isinstance(sparse, Mapping) else None
    return value.get("variables"), terms


def _bound_raw_polynomial(value: object) -> int:
    variables, terms = _raw_polynomial_parts(value)
    if isinstance(variables, (list, tuple)) and len(variables) > 1:
        raise _validation_error(
            "variable_count",
            "common interlacing sources must be univariate",
        )
    if not isinstance(terms, (list, tuple)):
        return 0
    if len(terms) > MAX_COMMON_INTERLACING_SOURCE_TERMS:
        raise _validation_error(
            "term_count",
            "a common interlacing source exceeds the "
            f"{MAX_COMMON_INTERLACING_SOURCE_TERMS}-term bound",
        )
    for term in terms:
        if isinstance(term, RationalPolynomialTerm):
            coefficient: object = term.coefficient
            exponents: object = term.exponents
        elif isinstance(term, Mapping):
            coefficient = term.get("coefficient")
            exponents = term.get("exponents")
        else:
            continue
        _bound_raw_rational(
            coefficient,
            label="common interlacing source coefficient",
        )
        if isinstance(exponents, (list, tuple)):
            if len(exponents) != 1:
                raise _validation_error(
                    "term_shape",
                    "common interlacing sources require one exponent per term",
                )
            exponent = exponents[0]
            if type(exponent) is int and exponent > (
                MAX_COMMON_INTERLACING_SOURCE_DEGREE
            ):
                raise _validation_error(
                    "source_degree",
                    "a common interlacing source exceeds the "
                    f"degree-{MAX_COMMON_INTERLACING_SOURCE_DEGREE} bound",
                )
    return len(terms)


class LabelledRationalPolynomial(StrictModel):
    """One opaque family label and one canonical polynomial over ``QQ``."""

    label: OpaqueLabel
    polynomial: RationalPolynomial = Field(
        description=(
            "A canonical polynomial over QQ. Every family member must use the "
            "same single variable, be monic, and have the same positive degree."
        )
    )


class CommonInterlacingRequest(StrictModel):
    """A bounded ordered family for a complete common-interlacing profile.

    Labels are unique and preserve the supplied family axis. Every polynomial
    must be monic, univariate in the same named variable, and have the same
    positive degree. Semantic and resource admission occurs once after this
    structural model has parsed the canonical source values.
    """

    family: tuple[LabelledRationalPolynomial, ...] = Field(
        min_length=2,
        max_length=MAX_COMMON_INTERLACING_FAMILY_SIZE,
        description=(
            "Two to eight uniquely labelled canonical QQ polynomials in the "
            "authoritative family order; all must be monic, use one shared "
            "variable, and have one shared positive degree. Each source has "
            "degree at most 32 and 64-digit rational components; the family "
            "has total degree at most 128."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw_family(cls, value: object) -> object:
        value = canonicalize_json_containers(value)
        if not isinstance(value, Mapping):
            return value
        family = value.get("family")
        if not isinstance(family, (list, tuple)):
            return value
        if len(family) > MAX_COMMON_INTERLACING_FAMILY_SIZE:
            raise _validation_error(
                "family_size",
                "common interlacing admits at most "
                f"{MAX_COMMON_INTERLACING_FAMILY_SIZE} family members",
            )
        total_terms = 0
        for member in family:
            polynomial: object
            if isinstance(member, LabelledRationalPolynomial):
                polynomial = member.polynomial
            elif isinstance(member, Mapping):
                polynomial = member.get("polynomial")
            else:
                continue
            total_terms += _bound_raw_polynomial(polynomial)
        if total_terms > MAX_COMMON_INTERLACING_TOTAL_TERMS:
            raise _validation_error(
                "total_terms",
                "common interlacing sources exceed the "
                f"{MAX_COMMON_INTERLACING_TOTAL_TERMS}-term family bound",
            )
        return value


class PolynomialRealRoot(StrictModel):
    """One distinct source root with exact multiplicity and algebraic identity."""

    value: _UnrecognizedRealAlgebraicValue
    multiplicity: StrictInt = Field(
        ge=1,
        le=MAX_COMMON_INTERLACING_SOURCE_DEGREE,
    )
    isolating_interval: RationalIsolatingInterval


class SourceRootProfile(StrictModel):
    """Distinct real roots of one family source in strictly increasing order."""

    source_index: StrictInt = Field(
        ge=0,
        lt=MAX_COMMON_INTERLACING_FAMILY_SIZE,
    )
    roots: tuple[PolynomialRealRoot, ...] = Field(
        default=(),
        max_length=MAX_COMMON_INTERLACING_SOURCE_DEGREE,
    )

    @model_validator(mode="after")
    def require_ordered_isolating_intervals(self) -> Self:
        for left, right in pairwise(self.roots):
            left_upper = left.isolating_interval.upper.as_fraction()
            right_lower = right.isolating_interval.lower.as_fraction()
            both_singletons = (
                left.isolating_interval.interval_type == "SINGLETON"
                and right.isolating_interval.interval_type == "SINGLETON"
            )
            if left_upper > right_lower or (
                left_upper == right_lower and both_singletons
            ):
                raise _validation_error(
                    "root_interval_order",
                    "source root intervals must be strictly ordered and pairwise disjoint",
                )
        return self


class PolynomialRootReference(StrictModel):
    """One root row on the retained family/source/root axes."""

    source_index: StrictInt = Field(
        ge=0,
        lt=MAX_COMMON_INTERLACING_FAMILY_SIZE,
    )
    distinct_root_index: StrictInt = Field(
        ge=0,
        lt=MAX_COMMON_INTERLACING_SOURCE_DEGREE,
    )


class CommonInterlacingGap(StrictModel):
    """One attained closed endpoint interval for an interlacer root."""

    gap_index: StrictInt = Field(
        ge=0,
        lt=MAX_COMMON_INTERLACING_SOURCE_DEGREE,
        description=(
            "Zero-based j for [max_r lambda_j(P_r), min_r lambda_(j+1)(P_r)]."
        ),
    )
    lower: PolynomialRootReference
    upper: PolynomialRootReference


class NonRealRootObstruction(StrictModel):
    """The first source polynomial that is not completely real-rooted."""

    kind: Literal["NON_REAL_ROOT"] = "NON_REAL_ROOT"
    source_index: StrictInt = Field(
        ge=0,
        lt=MAX_COMMON_INTERLACING_FAMILY_SIZE,
    )
    real_root_multiplicity: StrictInt = Field(
        ge=0,
        le=MAX_COMMON_INTERLACING_SOURCE_DEGREE,
    )
    nonreal_root_multiplicity: StrictInt = Field(
        ge=1,
        le=MAX_COMMON_INTERLACING_SOURCE_DEGREE,
    )


class EmptyGapObstruction(StrictModel):
    """The first gap whose attained lower endpoint exceeds its upper endpoint."""

    kind: Literal["EMPTY_GAP"] = "EMPTY_GAP"
    gap_index: StrictInt = Field(
        ge=0,
        lt=MAX_COMMON_INTERLACING_SOURCE_DEGREE,
    )
    maximum_lower: PolynomialRootReference
    minimum_upper: PolynomialRootReference


CommonInterlacingObstruction = Annotated[
    NonRealRootObstruction | EmptyGapObstruction,
    Field(discriminator="kind"),
]


class CommonInterlacingExists(StrictModel):
    """A complete exact common weak-interlacing profile."""

    status: Literal["EXISTS"] = "EXISTS"
    gaps: tuple[CommonInterlacingGap, ...] = Field(
        default=(),
        max_length=MAX_COMMON_INTERLACING_SOURCE_DEGREE - 1,
    )


class CommonInterlacingDoesNotExist(StrictModel):
    """One exact deterministic obstruction to common weak interlacing."""

    status: Literal["DOES_NOT_EXIST"] = "DOES_NOT_EXIST"
    obstruction: CommonInterlacingObstruction


CommonInterlacingOutcome = Annotated[
    CommonInterlacingExists | CommonInterlacingDoesNotExist,
    Field(discriminator="status"),
]


def _source_degree(source: LabelledRationalPolynomial) -> int:
    return max(
        (term.exponents[0] for term in source.polynomial.polynomial.terms),
        default=0,
    )


def _require_family_shape(
    family: tuple[LabelledRationalPolynomial, ...],
) -> int:
    labels = tuple(source.label for source in family)
    if len(set(labels)) != len(labels):
        raise _validation_error(
            "duplicate_label",
            "common interlacing family labels must be unique",
        )
    first = family[0].polynomial
    if len(first.variables) != 1:
        raise _validation_error(
            "variable_count",
            "common interlacing sources must be univariate",
        )
    degree = _source_degree(family[0])
    if degree < 1:
        raise _validation_error(
            "positive_degree",
            "common interlacing sources must have positive degree",
        )
    for source in family:
        polynomial = source.polynomial
        source_degree = _source_degree(source)
        if polynomial.variables != first.variables:
            raise _validation_error(
                "source_ring",
                "common interlacing sources must use the same named variable",
            )
        if source_degree != degree:
            raise _validation_error(
                "common_degree",
                "common interlacing sources must have the same positive degree",
            )
        terms = polynomial.polynomial.terms
        if not terms or terms[0].coefficient.as_fraction() != 1:
            raise _validation_error(
                "monic",
                "common interlacing sources must be monic over QQ",
            )
    return degree


class CommonInterlacingProfile(StrictModel):
    """Retained family, complete ordered real-root data, and exact outcome.

    ``root_profiles[i]`` belongs to ``family[i]``. Distinct roots are ordered
    increasingly and their multiplicities reconstruct the degree-length root
    tuple. The outcome is a discriminated exact-success or exact-obstruction
    branch; backend failure and timeout are not mathematical outcomes.
    """

    family: tuple[LabelledRationalPolynomial, ...] = Field(
        min_length=2,
        max_length=MAX_COMMON_INTERLACING_FAMILY_SIZE,
    )
    root_profiles: tuple[SourceRootProfile, ...] = Field(
        min_length=2,
        max_length=MAX_COMMON_INTERLACING_FAMILY_SIZE,
    )
    outcome: CommonInterlacingOutcome

    @property
    def status(self) -> Literal["EXISTS", "DOES_NOT_EXIST"]:
        """Return the discriminated outcome status for native callers."""

        return self.outcome.status

    def _require_reference(self, reference: PolynomialRootReference) -> None:
        if reference.source_index >= len(self.root_profiles):
            raise _validation_error(
                "root_reference_source",
                "root reference source_index must select a retained source",
            )
        roots = self.root_profiles[reference.source_index].roots
        if reference.distinct_root_index >= len(roots):
            raise _validation_error(
                "root_reference_index",
                "root reference distinct_root_index must select a retained root",
            )

    @model_validator(mode="after")
    def require_structural_profile(self) -> Self:
        if len(self.family) != len(self.root_profiles):
            raise _validation_error(
                "source_axis",
                "root profiles must align one-for-one with the retained family",
            )
        degree = _require_family_shape(self.family)
        if tuple(profile.source_index for profile in self.root_profiles) != tuple(
            range(len(self.family))
        ):
            raise _validation_error(
                "source_axis",
                "root profile source indices must follow the retained family axis",
            )
        multiplicities = tuple(
            sum(root.multiplicity for root in profile.roots)
            for profile in self.root_profiles
        )
        if any(total > degree for total in multiplicities):
            raise _validation_error(
                "root_multiplicity",
                "source real-root multiplicities cannot exceed the common degree",
            )

        if isinstance(self.outcome, CommonInterlacingExists):
            if any(total != degree for total in multiplicities):
                raise _validation_error(
                    "exists_real_rooted",
                    "an EXISTS profile requires every source root multiplicity to equal the degree",
                )
            if tuple(gap.gap_index for gap in self.outcome.gaps) != tuple(
                range(max(degree - 1, 0))
            ):
                raise _validation_error(
                    "gap_axis",
                    "an EXISTS profile requires one ordered interval per root gap",
                )
            references = tuple(
                reference
                for gap in self.outcome.gaps
                for reference in (gap.lower, gap.upper)
            )
        else:
            obstruction = self.outcome.obstruction
            if isinstance(obstruction, NonRealRootObstruction):
                source_index = obstruction.source_index
                if source_index >= len(self.family):
                    raise _validation_error(
                        "nonreal_source",
                        "non-real obstruction source_index must select the retained family",
                    )
                if any(total != degree for total in multiplicities[:source_index]):
                    raise _validation_error(
                        "first_nonreal_source",
                        "a non-real obstruction must identify the first non-real-rooted source",
                    )
                real_multiplicity = multiplicities[source_index]
                if (
                    real_multiplicity != obstruction.real_root_multiplicity
                    or degree - real_multiplicity
                    != obstruction.nonreal_root_multiplicity
                ):
                    raise _validation_error(
                        "nonreal_multiplicity",
                        "non-real obstruction multiplicities must match the retained root profile",
                    )
                references = ()
            else:
                if any(total != degree for total in multiplicities):
                    raise _validation_error(
                        "empty_gap_real_rooted",
                        "an empty-gap obstruction requires every source to be real-rooted",
                    )
                if obstruction.gap_index >= max(degree - 1, 0):
                    raise _validation_error(
                        "gap_axis",
                        "empty-gap obstruction must select a root gap",
                    )
                references = (
                    obstruction.maximum_lower,
                    obstruction.minimum_upper,
                )

        # Validate all references before indexing profiles to avoid
        # raw IndexError escaping the public operation.
        for reference in references:
            self._require_reference(reference)

        # Verify gap endpoints reference the correct expanded root positions.
        if isinstance(self.outcome, CommonInterlacingExists):
            for gap in self.outcome.gaps:
                for ref, pos in (
                    (gap.lower, gap.gap_index),
                    (gap.upper, gap.gap_index + 1),
                ):
                    profile = self.root_profiles[ref.source_index]
                    expanded_start = 0
                    for root_index, root in enumerate(profile.roots):
                        if root_index == ref.distinct_root_index:
                            break
                        expanded_start += root.multiplicity
                    root = profile.roots[ref.distinct_root_index]
                    if not (expanded_start <= pos < expanded_start + root.multiplicity):
                        raise _validation_error(
                            "gap_endpoint_position",
                            "gap endpoint must reference the gap_index-th and "
                            "(gap_index+1)-th expanded root positions",
                        )
        elif isinstance(self.outcome, CommonInterlacingDoesNotExist):
            obstruction = self.outcome.obstruction
            if isinstance(obstruction, EmptyGapObstruction):
                for ref, pos in (
                    (obstruction.maximum_lower, obstruction.gap_index),
                    (obstruction.minimum_upper, obstruction.gap_index + 1),
                ):
                    profile = self.root_profiles[ref.source_index]
                    expanded_start = 0
                    for root_index, root in enumerate(profile.roots):
                        if root_index == ref.distinct_root_index:
                            break
                        expanded_start += root.multiplicity
                    root = profile.roots[ref.distinct_root_index]
                    if not (expanded_start <= pos < expanded_start + root.multiplicity):
                        raise _validation_error(
                            "gap_endpoint_position",
                            "gap endpoint must reference the gap_index-th and "
                            "(gap_index+1)-th expanded root positions",
                        )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        family: tuple[LabelledRationalPolynomial, ...],
        root_profiles: tuple[SourceRootProfile, ...],
        outcome: CommonInterlacingOutcome,
    ) -> Self:
        """Build after the admitted exact kernel established the full relation."""

        return cls.model_construct(
            family=family,
            root_profiles=root_profiles,
            outcome=outcome,
        )


__all__ = [
    "CommonInterlacingDoesNotExist",
    "CommonInterlacingExists",
    "CommonInterlacingGap",
    "CommonInterlacingObstruction",
    "CommonInterlacingOutcome",
    "CommonInterlacingProfile",
    "CommonInterlacingRequest",
    "EmptyGapObstruction",
    "LabelledRationalPolynomial",
    "NonRealRootObstruction",
    "PolynomialRealRoot",
    "PolynomialRootReference",
    "SourceRootProfile",
]
