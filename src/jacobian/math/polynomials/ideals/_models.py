"""Typed wire contracts for commutative algebra operations."""

from __future__ import annotations

import itertools
from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    require_polynomial_budget,
)


def _validation_error(message: str) -> PydanticCustomError:
    return PydanticCustomError("polynomial.ideal_contract", message)


MAX_VARS = 8
MAX_GENERATORS = 32
MAX_INPUT_TERMS = 256
MAX_INPUT_EXPONENT = 20
MAX_COEFFICIENT_DIGITS = 128
MAX_OUTPUT_GENERATORS = 64
MAX_OUTPUT_TERMS = 1024
_RATIONAL_ROOT_PROBES = (0, 1, -1)


class IdealComputationBudget(StrictModel):
    """Enforced wall-time and exact-result limits for one Singular call."""

    wall_seconds: StrictInt = Field(default=10, ge=1, le=60)
    maximum_output_generators: StrictInt = Field(
        default=MAX_OUTPUT_GENERATORS,
        ge=MAX_OUTPUT_GENERATORS,
        le=MAX_OUTPUT_GENERATORS,
    )
    maximum_output_terms: StrictInt = Field(
        default=MAX_OUTPUT_TERMS,
        ge=MAX_OUTPUT_TERMS,
        le=MAX_OUTPUT_TERMS,
    )


def _require_ideal_budget(ideal: RationalPolynomialIdeal, *, label: str) -> None:
    if len(ideal.variables) > MAX_VARS:
        raise _validation_error(
            f"{label} exceeds the {MAX_VARS}-variable operation budget"
        )
    if len(ideal.generators) > MAX_GENERATORS:
        raise _validation_error(
            f"{label} exceeds the {MAX_GENERATORS}-generator operation budget"
        )
    if (
        sum(len(generator.polynomial.terms) for generator in ideal.generators)
        > MAX_INPUT_TERMS
    ):
        raise _validation_error(
            f"{label} exceeds the {MAX_INPUT_TERMS}-term aggregate input budget"
        )
    for generator in ideal.generators:
        require_polynomial_budget(
            generator,
            maximum_terms=MAX_INPUT_TERMS,
            maximum_exponent=MAX_INPUT_EXPONENT,
            maximum_coefficient_digits=MAX_COEFFICIENT_DIGITS,
            label=f"{label} generator",
        )
        if any(
            sum(term.exponents) > MAX_INPUT_EXPONENT
            for term in generator.polynomial.terms
        ):
            raise _validation_error(
                f"{label} generator exceeds total degree {MAX_INPUT_EXPONENT}"
            )


def _single_variable_slot(generator: RationalPolynomial) -> int | None:
    """Return the only variable slot a generator occupies, else ``None``."""

    occupied = {
        index
        for term in generator.polynomial.terms
        for index, exponent in enumerate(term.exponents)
        if exponent
    }
    return occupied.pop() if len(occupied) == 1 else None


def _certified_rational_roots(
    generator: RationalPolynomial,
    slot: int,
) -> list[int]:
    """Return the fixed probe points where the univariate exactly vanishes."""

    terms = [
        (Fraction(*term.coefficient.as_integer_ratio()), term.exponents[slot])
        for term in generator.polynomial.terms
    ]
    return [
        probe
        for probe in _RATIONAL_ROOT_PROBES
        if sum(coefficient * probe**exponent for coefficient, exponent in terms) == 0
    ]


def _vanishes_at_point(
    generator: RationalPolynomial,
    point: tuple[Fraction, ...],
) -> bool:
    """Evaluate one generator exactly at a rational point of the ring."""

    total = Fraction(0)
    for term in generator.polynomial.terms:
        value = Fraction(*term.coefficient.as_integer_ratio())
        for slot, exponent in enumerate(term.exponents):
            if exponent:
                value *= point[slot] ** exponent
        total += value
    return total == 0


def _occupied_slots(generator: RationalPolynomial) -> set[int]:
    """Return every variable slot a generator's terms actually occupy."""

    return {
        index
        for term in generator.polynomial.terms
        for index, exponent in enumerate(term.exponents)
        if exponent
    }


def _sole_owned_generators(
    ideal: RationalPolynomialIdeal,
) -> dict[int, RationalPolynomial]:
    """Map each variable slot to the only generator occupying it, if any."""

    sole: dict[int, RationalPolynomial] = {}
    contested: set[int] = set()
    for generator in ideal.generators:
        for slot in _occupied_slots(generator):
            if slot in sole:
                del sole[slot]
                contested.add(slot)
            elif slot not in contested:
                sole[slot] = generator
    return sole


def _require_provable_family_fit(ideal: RationalPolynomialIdeal) -> None:
    """Reject sources whose complete family provably overflows the envelope.

    Certificate: suppose ``k`` generators each live in one distinct
    variable and each vanishes at two distinct certified rational points.
    Every choice of one certified root per certified generator extends, by
    zero on every other variable, to a rational point of the whole ring;
    the choice is FEASIBLE only when every remaining generator of the
    ideal — not just the certified ones — vanishes at that extended point,
    so coupling or incompatible extra generators remove choices they rule
    out. Distinct feasible choices force distinct minimal primes: each
    holds exactly one irreducible factor per certified generator (two
    coprime factors would generate the unit ideal), and containment in the
    choice's maximal ideal forces that factor through a rational root,
    i.e. to be the corresponding linear form.

    Each further single-variable constraint that SOLELY occupies its own
    uncertified variable adds one forced generator to every counted prime
    without multiplying how many there are. Specializing the ring by the
    choice's roots (and zero elsewhere) leaves each such constraint a
    nonconstant univariate polynomial in its own surviving variable, so
    the images cannot jointly generate 1 and a prime above the specialized
    source exists; that prime contains the ``k`` linear forms plus ``e``
    nonconstant univariate constraints on distinct fresh variables, so it
    has height at least ``k + e`` and, by Krull's height theorem, needs at
    least ``k + e`` generators in any presentation. A slot shared with any
    other generator certifies nothing extra, because specialization could
    then combine constraints into a unit and falsely count dead choices.
    The complete family thus carries at least ``feasible * (k + e)``
    aggregate generators: when that exceeds the exact-result envelope,
    every admitted execution ends in typed LIMIT_EXCEEDED and the source
    is rejected here before any backend launch. A source containing a
    nonzero constant is the unit ideal with an empty family and always
    stays admitted, and a source with no feasible choice certifies nothing
    and stays admitted.
    """

    unit_witness = any(
        len(generator.polynomial.terms) == 1
        and not any(generator.polynomial.terms[0].exponents)
        and generator.polynomial.terms[0].coefficient.num != "0"
        for generator in ideal.generators
    )
    if unit_witness:
        return
    certified_roots: dict[int, list[int]] = {}
    for generator in ideal.generators:
        slot = _single_variable_slot(generator)
        if slot is None or slot in certified_roots:
            continue
        roots = _certified_rational_roots(generator, slot)
        if len(roots) >= 2:
            certified_roots[slot] = roots
    count = len(certified_roots)
    if not count:
        return
    extra_constraints = {
        slot: generator
        for slot, generator in _sole_owned_generators(ideal).items()
        if slot not in certified_roots and _single_variable_slot(generator) == slot
    }
    minimum_component_generators = count + len(extra_constraints)
    constrained = [
        generator
        for generator in ideal.generators
        if _single_variable_slot(generator) not in extra_constraints
    ]
    slots = sorted(certified_roots)
    variables = len(ideal.variables)
    feasible = 0
    for choice in itertools.product(*(certified_roots[slot] for slot in slots)):
        values = dict(zip(slots, choice, strict=True))
        point = tuple(Fraction(values.get(slot, 0)) for slot in range(variables))
        if all(_vanishes_at_point(generator, point) for generator in constrained):
            feasible += 1
            if feasible * minimum_component_generators > MAX_OUTPUT_GENERATORS:
                raise _validation_error(
                    "the source provably forces a complete family above the "
                    f"{MAX_OUTPUT_GENERATORS}-generator exact-result envelope: "
                    f"{count} single-variable generators each certified "
                    f"against two distinct rational roots yield at least "
                    f"{feasible} feasible root choices as minimal primes, "
                    f"and each carries at least "
                    f"{minimum_component_generators} generators including "
                    f"{len(extra_constraints)} further independent "
                    f"single-variable constraints "
                    f"({feasible * minimum_component_generators} aggregate "
                    f"generators)"
                )


class IdealRadicalRequest(StrictModel):
    """Compute ``sqrt(I)`` for a bounded ideal ``I`` in ``QQ[variables]``."""

    ideal: RationalPolynomialIdeal = Field(
        description=(
            "An ideal in at most 6 variables with at most 16 generators and "
            "256 aggregate terms; generator total degree is at most 12 and "
            "coefficient components are at most 128 digits."
        )
    )
    resource_budget: IdealComputationBudget = Field(
        default_factory=IdealComputationBudget
    )


class IdealRadicalMembershipRequest(StrictModel):
    """Check membership of one polynomial in the radical of a bounded ideal."""

    ideal: RationalPolynomialIdeal = Field(
        description=(
            "An ideal in at most 6 variables with at most 16 generators and "
            "256 aggregate terms; generator total degree is at most 12 and "
            "coefficient components are at most 128 digits."
        )
    )
    polynomial: RationalPolynomial = Field(
        description=(
            "A polynomial in the ideal's exact ordered ring, with at most 256 "
            "terms, total degree at most 12, and coefficient components at most "
            "128 digits."
        )
    )

    @model_validator(mode="after")
    def require_backend_domain(self) -> Self:
        if self.polynomial.variables != self.ideal.variables:
            raise _validation_error(
                "membership polynomial must use the ideal's ordered ring"
            )
        return self


class IdealSaturationRequest(StrictModel):
    """Compute ``I : <d>^infinity`` for a bounded ideal and one polynomial."""

    ideal: RationalPolynomialIdeal = Field(
        description=(
            "An ideal in at most 6 variables with at most 16 generators and "
            "256 aggregate terms; generator total degree is at most 12 and "
            "coefficient components are at most 128 digits."
        )
    )
    denominator: RationalPolynomial = Field(
        description=(
            "A single nonzero polynomial d in the dividend's exact ordered "
            "ring, with at most 256 terms, total degree at most 12, and "
            "coefficient components at most 128 digits."
        )
    )
    resource_budget: IdealComputationBudget = Field(
        default_factory=IdealComputationBudget
    )

    @model_validator(mode="after")
    def require_matching_ring(self) -> Self:
        if self.denominator.variables != self.ideal.variables:
            raise _validation_error(
                "saturation operands must use the same ordered ring"
            )
        return self


class IdealQuotientRequest(StrictModel):
    """Compute ``(I : J)`` for bounded ideals in one ``QQ`` ring."""

    dividend: RationalPolynomialIdeal = Field(
        description=(
            "An ideal in at most 6 variables with at most 16 generators and "
            "256 aggregate terms; generator total degree is at most 12 and "
            "coefficient components are at most 128 digits."
        )
    )
    divisor: RationalPolynomialIdeal = Field(
        description=(
            "An ideal in the dividend's exact ordered ring, with the same "
            "6-variable, 16-generator, 256-term, degree-12, and 128-digit bounds."
        )
    )
    resource_budget: IdealComputationBudget = Field(
        default_factory=IdealComputationBudget
    )

    @model_validator(mode="after")
    def require_matching_ring(self) -> Self:
        if self.dividend.variables != self.divisor.variables:
            raise _validation_error(
                "ideal quotient operands must use the same ordered ring"
            )
        return self


IdealExecutionOutcome = Literal[
    "COMPUTED",
    "UNAVAILABLE",
    "TIMEOUT",
    "CANCELLED",
    "LIMIT_EXCEEDED",
    "ERROR",
]


class IdealRadicalResult(StrictModel):
    outcome: IdealExecutionOutcome
    radical: RationalPolynomialIdeal | None = None
    backend_version: str | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def require_outcome_shape(self) -> Self:
        if self.outcome == "COMPUTED":
            if self.radical is None or self.backend_version is None or self.detail:
                raise _validation_error(
                    "computed radical requires a value and backend version"
                )
        elif (
            self.radical is not None
            or self.backend_version is not None
            or not self.detail
        ):
            raise _validation_error(
                "failed radical computation requires only a safe detail"
            )
        return self


class IdealRadicalMembershipResult(StrictModel):
    in_radical: bool


class IdealQuotientResult(StrictModel):
    outcome: IdealExecutionOutcome
    quotient: RationalPolynomialIdeal | None = None
    backend_version: str | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def require_outcome_shape(self) -> Self:
        if self.outcome == "COMPUTED":
            if self.quotient is None or self.backend_version is None or self.detail:
                raise _validation_error(
                    "computed quotient requires a value and backend version"
                )
        elif (
            self.quotient is not None
            or self.backend_version is not None
            or not self.detail
        ):
            raise _validation_error(
                "failed quotient computation requires only a safe detail"
            )
        return self


class IdealSaturationResult(StrictModel):
    outcome: IdealExecutionOutcome
    saturation: RationalPolynomialIdeal | None = None
    backend_version: str | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def require_outcome_shape(self) -> Self:
        if self.outcome == "COMPUTED":
            if self.saturation is None or self.backend_version is None or self.detail:
                raise _validation_error(
                    "computed saturation requires a value and backend version"
                )
        elif (
            self.saturation is not None
            or self.backend_version is not None
            or not self.detail
        ):
            raise _validation_error(
                "failed saturation computation requires only a safe detail"
            )
        return self


class IdealMinimalPrimesRequest(StrictModel):
    """Compute minimal primes of an ideal in the exact ring ``QQ[variables]``."""

    ideal: RationalPolynomialIdeal = Field(
        description=(
            "An ideal in at most 8 variables with at most 32 generators and "
            "256 aggregate terms; generator total degree is at most 20 and "
            "coefficient components are at most 128 digits. A source whose "
            "complete family provably exceeds the aggregate 64-generator "
            "exact-result envelope — k certified two-rational-root "
            "single-variable generators force at least 2^k minimal primes, "
            "and each further independent single-variable constraint raises "
            "every component's forced generator count — is rejected here "
            "before the backend launches; otherwise a family exceeding the "
            "aggregate 64-generator or 1024-term envelope returns a typed "
            "LIMIT_EXCEEDED outcome. The complete serialized result must also "
            "fit the repository's canonical 10 MiB output limit."
        )
    )
    resource_budget: IdealComputationBudget = Field(
        default_factory=IdealComputationBudget
    )


class IdealMinimalPrimesResult(StrictModel):
    """The complete minimal-prime family of a retained rational ideal source.

    Components are the minimal primes of ``ideal`` over ``QQ``—not geometric
    components after scalar extension to an algebraic closure.  The empty
    family represents the unit ideal, whose empty intersection is the unit
    ideal.
    """

    ideal: RationalPolynomialIdeal
    outcome: IdealExecutionOutcome
    components: tuple[RationalPolynomialIdeal, ...] | None = None
    backend_version: str | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def require_outcome_shape(self) -> Self:
        if self.outcome != "COMPUTED":
            if (
                self.components is not None
                or self.backend_version is not None
                or not self.detail
            ):
                raise _validation_error(
                    "incomplete minimal-prime computation requires only a safe detail"
                )
            return self
        if (
            self.components is None
            or self.backend_version is None
            or self.detail is not None
        ):
            raise _validation_error(
                "computed minimal-prime family requires components and backend version"
            )
        _require_computed_minimal_prime_family(self.ideal, self.components)
        return self

    @classmethod
    def _from_kernel(
        cls,
        ideal: RationalPolynomialIdeal,
        components: tuple[RationalPolynomialIdeal, ...] | None,
        backend_version: str | None,
    ) -> Self:
        """Build a computed result from the trusted Singular adapter."""

        return cls.model_construct(
            ideal=ideal,
            outcome="COMPUTED",
            components=components,
            backend_version=backend_version,
        )


def _require_computed_minimal_prime_family(
    ideal: RationalPolynomialIdeal,
    components: tuple[RationalPolynomialIdeal, ...],
) -> None:
    """Gate ring, exact-result envelopes, ordering, and uniqueness."""

    if any(component.variables != ideal.variables for component in components):
        raise _validation_error(
            "every minimal prime must use the source ideal's ordered ring"
        )
    total_generators = 0
    total_terms = 0
    for component in components:
        total_generators += len(component.generators)
        total_terms += sum(
            len(generator.polynomial.terms) for generator in component.generators
        )
    if total_generators > MAX_OUTPUT_GENERATORS:
        raise _validation_error(
            "the complete family must fit the "
            f"{MAX_OUTPUT_GENERATORS}-generator exact-result envelope"
        )
    if total_terms > MAX_OUTPUT_TERMS:
        raise _validation_error(
            "the complete family must fit the "
            f"{MAX_OUTPUT_TERMS}-term exact-result envelope"
        )
    keys = tuple(component.model_dump_json() for component in components)
    if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
        raise _validation_error(
            "minimal-prime components must be unique and canonically ordered"
        )


__all__ = [
    "MAX_OUTPUT_GENERATORS",
    "MAX_OUTPUT_TERMS",
    "EliminationExecutionOutcome",
    "EliminationIdealRequest",
    "EliminationIdealResult",
    "GroebnerBasisRequest",
    "GroebnerBasisResult",
    "IdealComputationBudget",
    "IdealExecutionOutcome",
    "IdealMinimalPrimesRequest",
    "IdealMinimalPrimesResult",
    "IdealNormalFormRequest",
    "IdealNormalFormResult",
    "IdealQuotientRequest",
    "IdealQuotientResult",
    "IdealRadicalMembershipRequest",
    "IdealRadicalMembershipResult",
    "IdealRadicalRequest",
    "IdealRadicalResult",
    "IdealSaturationRequest",
    "IdealSaturationResult",
    "NormalFormExecutionOutcome",
]


# ---------------------------------------------------------------------------
# Gröbner basis computation
# ---------------------------------------------------------------------------


class GroebnerBasisRequest(StrictModel):
    """Compute a reduced Gröbner basis for a bounded ideal in QQ[variables]."""

    ideal: RationalPolynomialIdeal
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex"
    resource_budget: IdealComputationBudget = Field(
        default_factory=IdealComputationBudget
    )


GroebnerExecutionOutcome = Literal["COMPUTED", "ERROR", "LIMIT_EXCEEDED", "TIMEOUT"]


class GroebnerBasisResult(StrictModel):
    """A reduced Gröbner basis, or a typed timeout under the enforced budget."""

    ideal: RationalPolynomialIdeal
    outcome: GroebnerExecutionOutcome = "COMPUTED"
    basis: RationalPolynomialIdeal | None = None
    generator_count: StrictInt = Field(default=0, ge=0, le=MAX_OUTPUT_GENERATORS)
    monomial_order: Literal["lex", "grlex", "grevlex"]
    detail: str | None = None

    @model_validator(mode="after")
    def require_outcome_shape(self) -> Self:
        if self.outcome == "COMPUTED":
            if self.basis is None or self.detail is not None:
                raise _validation_error(
                    "computed basis requires a value and no failure detail"
                )
            if (
                self.generator_count != len(self.basis.generators)
                or self.generator_count < 1
            ):
                raise _validation_error(
                    "generator_count must match the basis generator count"
                )
            _require_basis_shape(self.basis, self.ideal)
        elif self.basis is not None or self.detail is None:
            raise _validation_error("timed-out computation carries only a safe detail")
        return self

    @classmethod
    def _from_kernel(
        cls,
        ideal: RationalPolynomialIdeal,
        basis: RationalPolynomialIdeal,
        monomial_order: Literal["lex", "grlex", "grevlex"],
    ) -> Self:
        """Build a computed result after the bounded kernel has produced it."""

        return cls.model_construct(
            ideal=ideal,
            basis=basis,
            generator_count=len(basis.generators),
            monomial_order=monomial_order,
        )


def _is_zero_polynomial(polynomial: RationalPolynomial) -> bool:
    return not polynomial.polynomial.terms


def _require_basis_shape(
    basis: RationalPolynomialIdeal,
    source_ideal: RationalPolynomialIdeal,
) -> None:
    """Gate result-local ring, zero, and exact-result envelope invariants."""

    if basis.variables != source_ideal.variables:
        raise _validation_error("basis must use the source ideal's ordered ring")
    if any(_is_zero_polynomial(generator) for generator in basis.generators) and not (
        len(basis.generators) == 1
        and all(_is_zero_polynomial(generator) for generator in source_ideal.generators)
    ):
        raise _validation_error(
            "a reduced Gröbner basis contains no zero generator; only "
            "the zero ideal admits the singleton-zero representation"
        )


# ---------------------------------------------------------------------------
# Normal form / ideal membership
# ---------------------------------------------------------------------------


NormalFormMonomialOrder = Literal["lex", "grlex", "grevlex"]


class IdealNormalFormRequest(StrictModel):
    """Reduce one polynomial modulo an ideal's Gröbner basis.

    ``monomial_order`` names the order of the Groebner basis the reduction
    uses; normal forms depend on it, so it is part of the public contract.
    """

    ideal: RationalPolynomialIdeal
    polynomial: RationalPolynomial
    monomial_order: NormalFormMonomialOrder = "grevlex"


NormalFormExecutionOutcome = Literal["COMPUTED", "ERROR", "LIMIT_EXCEEDED", "TIMEOUT"]


class IdealNormalFormResult(StrictModel):
    """The exact remainder modulo an ideal, or a typed incomplete outcome."""

    ideal: RationalPolynomialIdeal
    polynomial: RationalPolynomial
    outcome: NormalFormExecutionOutcome = "COMPUTED"
    remainder: RationalPolynomial | None = None
    in_ideal: bool | None = None
    monomial_order: NormalFormMonomialOrder = "grevlex"
    detail: str | None = None

    @model_validator(mode="after")
    def require_consistent_membership(self) -> Self:
        if self.outcome != "COMPUTED" and self.in_ideal is not None:
            raise _validation_error(
                "an incomplete normal-form outcome states no membership conclusion"
            )
        if self.outcome == "COMPUTED":
            if self.remainder is None or self.detail is not None:
                raise _validation_error(
                    "computed normal form requires a remainder and no failure detail"
                )
            # A computed outcome claims its authoritative membership
            # decision; omitting it would let the result claim success
            # while withholding the conclusion.
            if self.in_ideal is None:
                raise _validation_error(
                    "a computed normal form must state its membership "
                    "conclusion in in_ideal"
                )
            if self.in_ideal and len(self.remainder.polynomial.terms) > 0:
                raise _validation_error(
                    "a polynomial in the ideal must have a zero remainder"
                )
            if not self.in_ideal and len(self.remainder.polynomial.terms) == 0:
                raise _validation_error(
                    "a polynomial not in the ideal must have a nonzero remainder"
                )
            if self.remainder.variables != self.ideal.variables:
                raise _validation_error(
                    "remainder must use the retained ideal's ordered ring"
                )
        elif self.remainder is not None or self.detail is None:
            raise _validation_error("timed-out computation carries only a safe detail")
        return self

    @classmethod
    def _from_kernel(
        cls,
        ideal: RationalPolynomialIdeal,
        polynomial: RationalPolynomial,
        monomial_order: NormalFormMonomialOrder,
        remainder: RationalPolynomial,
    ) -> Self:
        """Build a computed normal form from the bounded kernel output."""

        return cls.model_construct(
            ideal=ideal,
            polynomial=polynomial,
            remainder=remainder,
            in_ideal=_is_zero_polynomial(remainder),
            monomial_order=monomial_order,
        )


# ---------------------------------------------------------------------------
# Elimination ideal
# ---------------------------------------------------------------------------


class EliminationIdealRequest(StrictModel):
    """Compute the elimination ideal I ∩ QQ[remaining variables]."""

    ideal: RationalPolynomialIdeal
    eliminated_variables: tuple[str, ...] = Field(min_length=1, max_length=MAX_VARS)
    resource_budget: IdealComputationBudget = Field(
        default_factory=IdealComputationBudget
    )

    @model_validator(mode="after")
    def require_backend_domain(self) -> Self:
        eliminated = set(self.eliminated_variables)
        for var in eliminated:
            if var not in self.ideal.variables:
                raise _validation_error(
                    "eliminated variables must be a subset of the ideal's variables"
                )
        remaining = tuple(v for v in self.ideal.variables if v not in eliminated)
        if not remaining:
            raise _validation_error(
                "elimination cannot remove every variable; at least one must remain"
            )
        return self


EliminationExecutionOutcome = Literal["COMPUTED", "ERROR", "LIMIT_EXCEEDED", "TIMEOUT"]


class EliminationIdealResult(StrictModel):
    """The elimination ideal I ∩ QQ[remaining variables], or a typed timeout under the enforced budget."""

    ideal: RationalPolynomialIdeal
    outcome: EliminationExecutionOutcome = "COMPUTED"
    elimination_ideal: RationalPolynomialIdeal | None = None
    eliminated_variables: tuple[str, ...] = Field(min_length=1, max_length=MAX_VARS)
    detail: str | None = None

    @model_validator(mode="after")
    def require_consistent_result(self) -> Self:
        if self.outcome == "COMPUTED":
            if self.elimination_ideal is None or self.detail is not None:
                raise _validation_error(
                    "computed elimination requires an ideal and no failure detail"
                )
            for var in self.eliminated_variables:
                if var in self.elimination_ideal.variables:
                    raise _validation_error(
                        "eliminated variables must not appear in the elimination ideal"
                    )
            remaining = tuple(
                variable
                for variable in self.ideal.variables
                if variable not in set(self.eliminated_variables)
            )
            if self.elimination_ideal.variables != remaining:
                raise _validation_error(
                    "elimination ideal must use exactly the remaining ordered ring"
                )
        elif self.elimination_ideal is not None or self.detail is None:
            raise _validation_error("timed-out computation carries only a safe detail")
        return self

    @classmethod
    def _from_kernel(
        cls,
        ideal: RationalPolynomialIdeal,
        eliminated_variables: tuple[str, ...],
        elimination_ideal: RationalPolynomialIdeal,
    ) -> Self:
        """Build a computed elimination result from the bounded kernel output."""

        return cls.model_construct(
            ideal=ideal,
            elimination_ideal=elimination_ideal,
            eliminated_variables=eliminated_variables,
        )
