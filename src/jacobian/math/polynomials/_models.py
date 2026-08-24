"""Contracts for exact polynomial invariants over ``QQ``."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalInteger,
    CanonicalRational,
)
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_TERMS,
    MAX_POLYNOMIAL_VARIABLES,
    PolynomialVariable,
    RationalPolynomial,
    require_polynomial_budget,
)

_MAX_COEFFICIENT_DIGITS = 256
_MAX_GCD_TERMS = 1024
_MAX_INVARIANT_TERMS = 256
# Cumulative monomial-multiplication budget for one factor-replay
# authentication.  Sized from the replay's admitted envelope: the largest
# known authentic canonical decomposition inside the square-free request
# envelope, S_57(x)*(S_31(y)S_31(z))^2*(x+1)^3*((x-1)(y-1)(z-1))^4 with
# per-variable degrees (63, 64, 64), performs exactly 29,892,365 pairwise
# monomial multiplications under multiplicity-ordered reconstruction, so
# the ceiling admits it with headroom while still bounding every forged
# claim's arithmetic before a mismatch can hide inside unbounded work.
_MAX_REPLAY_WORK = 1 << 25
# Conservative ceiling on one replay step's predicted product support,
# checked before the backend multiplication runs.  The admitted trivariate
# square-free envelope keeps every partial-product degree box at or below
# 65^3 = 274,625 terms (the fixture above peaks at 270,400), so this
# ceiling admits that whole envelope; wider claimed degree boxes reject
# before materializing anything rather than trusting backend expansion.
_MAX_REPLAY_INTERMEDIATE_TERMS = 1 << 19
_MAX_GCD_DEGREE = 500
_MAX_ELIMINATION_DEGREE_SUM = 128
_MAX_DISCRIMINANT_DEGREE = 64
_MAX_SQUARE_FREE_EXPONENT = 64
_MAX_ELEMENTARY_DEGREE = 127
_MAX_INTEGER_COEFFICIENT_DIGITS = 256


def _degree(polynomial: RationalPolynomial, variable_index: int) -> int:
    return max(
        (term.exponents[variable_index] for term in polynomial.polynomial.terms),
        default=0,
    )


def _polynomial_total_degree(polynomial: RationalPolynomial) -> int:
    return max(
        (sum(term.exponents) for term in polynomial.polynomial.terms),
        default=0,
    )


class PolynomialPairRequest(StrictModel):
    """Two polynomials in one identical declared rational polynomial ring."""

    left: RationalPolynomial
    right: RationalPolynomial

    @model_validator(mode="after")
    def require_matching_rings(self) -> Self:
        if self.left.variables != self.right.variables:
            raise ValueError("polynomials must use the same ordered variables")
        return self


class PolynomialGcdRequest(PolynomialPairRequest):
    @model_validator(mode="after")
    def require_univariate_budget(self) -> Self:
        if len(self.left.variables) != 1:
            raise ValueError("Bézout GCD currently supports one variable over QQ")
        for polynomial in (self.left, self.right):
            require_polynomial_budget(
                polynomial,
                maximum_terms=_MAX_GCD_TERMS,
                maximum_exponent=_MAX_GCD_DEGREE,
            )
        return self

    @model_validator(mode="after")
    def require_not_both_zero(self) -> Self:
        """Reject gcd(0, 0): zero has no monic normalization."""
        left_zero = len(self.left.polynomial.terms) == 0
        right_zero = len(self.right.polynomial.terms) == 0
        if left_zero and right_zero:
            raise ValueError("gcd(0, 0) is undefined: zero has no monic normalization")
        return self


class PolynomialBezoutIdentity(StrictModel):
    left_multiplier: RationalPolynomial
    right_multiplier: RationalPolynomial


class PolynomialGcdResult(StrictModel):
    gcd: RationalPolynomial
    bezout: PolynomialBezoutIdentity
    normalization: Literal["MONIC"] = "MONIC"


class PolynomialResultantRequest(PolynomialPairRequest):
    elimination_variable: PolynomialVariable

    @model_validator(mode="after")
    def require_elimination_budget(self) -> Self:
        if self.elimination_variable not in self.left.variables:
            raise ValueError("elimination variable must belong to the declared ring")
        for polynomial in (self.left, self.right):
            require_polynomial_budget(
                polynomial,
                maximum_terms=_MAX_INVARIANT_TERMS,
                maximum_exponent=_MAX_ELIMINATION_DEGREE_SUM,
            )
        variable_index = self.left.variables.index(self.elimination_variable)
        degree_sum = _degree(self.left, variable_index) + _degree(
            self.right, variable_index
        )
        if degree_sum > _MAX_ELIMINATION_DEGREE_SUM:
            raise ValueError("Sylvester degree exceeds the resultant budget")
        return self


class PolynomialDiscriminantRequest(StrictModel):
    polynomial: RationalPolynomial
    variable: PolynomialVariable

    @model_validator(mode="after")
    def require_discriminant_budget(self) -> Self:
        if self.variable not in self.polynomial.variables:
            raise ValueError("discriminant variable must belong to the declared ring")
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=_MAX_INVARIANT_TERMS,
            maximum_exponent=_MAX_SQUARE_FREE_EXPONENT,
        )
        variable_index = self.polynomial.variables.index(self.variable)
        if _degree(self.polynomial, variable_index) > _MAX_DISCRIMINANT_DEGREE:
            raise ValueError("main-variable degree exceeds the discriminant budget")
        return self


class PolynomialScalarValue(StrictModel):
    kind: Literal["SCALAR"] = "SCALAR"
    value: CanonicalRational


class PolynomialValue(StrictModel):
    kind: Literal["POLYNOMIAL"] = "POLYNOMIAL"
    value: RationalPolynomial


PolynomialInvariantValue = Annotated[
    PolynomialScalarValue | PolynomialValue,
    Field(discriminator="kind"),
]


class PolynomialResultantResult(StrictModel):
    elimination_variable: PolynomialVariable
    resultant: PolynomialInvariantValue
    convention: Literal["SYLVESTER_DETERMINANT"] = "SYLVESTER_DETERMINANT"


class PolynomialDiscriminantResult(StrictModel):
    variable: PolynomialVariable
    discriminant: PolynomialInvariantValue
    convention: Literal["STANDARD_UNIVARIATE"] = "STANDARD_UNIVARIATE"


class PolynomialSquareFreeRequest(StrictModel):
    polynomial: RationalPolynomial

    @model_validator(mode="after")
    def require_square_free_budget(self) -> Self:
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=_MAX_GCD_TERMS,
            maximum_exponent=_MAX_SQUARE_FREE_EXPONENT,
        )
        return self


class PolynomialSquareFreeFactor(StrictModel):
    factor: RationalPolynomial
    multiplicity: int = Field(ge=1, le=_MAX_SQUARE_FREE_EXPONENT)


class PolynomialSquareFreeDecompositionResult(StrictModel):
    """The square-free decomposition bound to its source polynomial.

    Retains the canonical source polynomial so validation replays
    ``reconstructed = polynomial`` and authenticates the defining relation
    ``polynomial = coefficient * product(factor^multiplicity)`` together
    with everything that makes the records THE monic square-free
    decomposition — pairwise-coprime square-free parts, distinct
    multiplicities, monic records — by exact bounded checks on the claimed
    records alone.  Over ``QQ`` these properties admit exactly one
    decomposition, so no backend recomputation of the source's (possibly
    unrepresentable) true decomposition is ever needed.
    """

    polynomial: RationalPolynomial
    coefficient: CanonicalRational
    factors: tuple[PolynomialSquareFreeFactor, ...] = Field(max_length=64)
    reconstructed: RationalPolynomial
    normalization: Literal["MONIC_FACTORS"] = "MONIC_FACTORS"

    @model_validator(mode="after")
    def require_canonical_factor_records(self) -> Self:
        from jacobian.math.polynomials._conversions import (
            rational_polynomial_to_sympy,
        )

        multiplicities = tuple(factor.multiplicity for factor in self.factors)
        if multiplicities != tuple(sorted(multiplicities)):
            raise ValueError("square-free factors must be ordered by multiplicity")
        if len(set(multiplicities)) != len(multiplicities):
            raise ValueError("each multiplicity must have one square-free factor")
        if any(
            factor.factor.variables != self.reconstructed.variables
            for factor in self.factors
        ):
            raise ValueError("square-free factors must use the source ring")
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=_MAX_GCD_TERMS,
            maximum_exponent=_MAX_SQUARE_FREE_EXPONENT,
            label="retained source polynomial",
        )
        require_polynomial_budget(
            self.reconstructed,
            maximum_terms=_MAX_GCD_TERMS,
            maximum_exponent=_MAX_SQUARE_FREE_EXPONENT,
            label="reconstructed polynomial",
        )
        source = rational_polynomial_to_sympy(self.polynomial)
        if rational_polynomial_to_sympy(self.reconstructed) != source:
            raise ValueError("reconstructed must equal the retained source polynomial")
        _verify_exact_factor_product(
            source,
            self.factors,
            coefficient=self.coefficient,
            mismatch_message=(
                "square-free factors must reconstruct the retained source "
                "polynomial exactly"
            ),
            label="square-free",
            maximum_exponent=_MAX_SQUARE_FREE_EXPONENT,
            require_square_free_parts=True,
        )
        return self


class PolynomialFactorRequest(StrictModel):
    """Univariate factorization request over ``QQ``."""

    polynomial: RationalPolynomial

    @model_validator(mode="after")
    def require_univariate_factor_budget(self) -> Self:
        if len(self.polynomial.variables) != 1:
            raise ValueError("factorization currently supports one variable over QQ")
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=_MAX_GCD_TERMS,
            maximum_exponent=_MAX_GCD_DEGREE,
        )
        return self


class PolynomialIrreducibleFactor(StrictModel):
    factor: RationalPolynomial
    multiplicity: int = Field(ge=1, le=_MAX_GCD_DEGREE)


class PolynomialFactorizationResult(StrictModel):
    """The exact univariate factorization bound to its source polynomial.

    Retains the canonical source polynomial so validation replays the
    defining relations ``reconstructed = polynomial`` and
    ``polynomial = coefficient * product(factor^multiplicity)`` together
    with everything that makes the records THE content-and-monic-
    irreducibles factorization — monic distinct irreducible records under
    canonical ordering — by exact bounded checks on the claimed records
    alone.  Over ``QQ`` these properties admit exactly one factorization,
    so no backend recomputation of the retained source's true
    factorization is ever needed.  The literal
    ``product_reconstruction = EXACT`` label is derived from that replay,
    never accepted as evidence.
    """

    polynomial: RationalPolynomial
    coefficient: CanonicalRational
    factors: tuple[PolynomialIrreducibleFactor, ...] = Field(max_length=64)
    reconstructed: RationalPolynomial
    normalization: Literal["CONTENT_AND_MONIC_IRREDUCIBLES"] = (
        "CONTENT_AND_MONIC_IRREDUCIBLES"
    )
    product_reconstruction: Literal["EXACT"] = "EXACT"

    @model_validator(mode="after")
    def require_canonical_irreducible_records(self) -> Self:
        from jacobian.math.polynomials._conversions import (
            rational_polynomial_to_sympy,
        )

        if len(self.polynomial.variables) != 1:
            raise ValueError("factorization currently supports one variable over QQ")
        if any(
            factor.factor.variables != self.reconstructed.variables
            for factor in self.factors
        ):
            raise ValueError("irreducible factors must use the source ring")
        ordered = tuple(
            sorted(
                self.factors,
                key=lambda record: (
                    record.multiplicity,
                    max(
                        (
                            sum(term.exponents)
                            for term in record.factor.polynomial.terms
                        ),
                        default=0,
                    ),
                    tuple(
                        (
                            term.exponents,
                            term.coefficient.num,
                            term.coefficient.den,
                        )
                        for term in record.factor.polynomial.terms
                    ),
                ),
            )
        )
        if self.factors != ordered:
            raise ValueError(
                "irreducible factors must be ordered by multiplicity, degree, "
                "and sparse term fingerprint"
            )
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=_MAX_GCD_TERMS,
            maximum_exponent=_MAX_GCD_DEGREE,
            label="retained source polynomial",
        )
        require_polynomial_budget(
            self.reconstructed,
            maximum_terms=_MAX_GCD_TERMS,
            maximum_exponent=_MAX_GCD_DEGREE,
            label="reconstructed polynomial",
        )
        source = rational_polynomial_to_sympy(self.polynomial)
        if rational_polynomial_to_sympy(self.reconstructed) != source:
            raise ValueError("reconstructed must equal the retained source polynomial")
        _verify_exact_factor_product(
            source,
            self.factors,
            coefficient=self.coefficient,
            mismatch_message=(
                "factorization must reconstruct the retained source polynomial exactly"
            ),
            label="irreducible",
            maximum_exponent=_MAX_GCD_DEGREE,
            require_square_free_parts=False,
        )
        return self


def _canonical_poly_key(polynomial: Any) -> Any:
    """Return the hashable canonical form of one monic QQ ``Poly``."""

    return tuple(
        sorted(
            (monom, int(coeff.p), int(coeff.q)) for monom, coeff in polynomial.terms()
        )
    )


def _degree_vector(polynomial: Any) -> tuple[int, ...]:
    """Return one backend polynomial's per-generator maximum exponents."""

    monoms = polynomial.monoms()
    return tuple(
        max((monom[index] for monom in monoms), default=0)
        for index in range(len(polynomial.gens))
    )


def _require_bounded_replay_step(
    accumulated: Any, base: Any, work: int, *, label: str
) -> tuple[Any, int]:
    """Multiply one reconstruction step under proven preflight bounds.

    Two independent quantities are derived from the claimed records'
    actual structure and checked before any backend multiplication runs:

    * ``support`` — every output monomial arises from exactly one input
      pair, so ``len(accumulated) * len(base)`` upper-bounds the product's
      support, and so does the per-variable degree box
      ``prod_i(deg_i(accumulated) + deg_i(base) + 1)``.  Their minimum is
      therefore a proven support bound that uses whichever structure the
      claimed records actually exhibit.  Claimed powers whose supports
      collide into a sparse product — geometric sums telescoping under
      multiplication — are admitted where the bare pairwise count would
      reject an authentic decomposition, while claims whose product cannot
      stay inside ``_MAX_REPLAY_INTERMEDIATE_TERMS`` reject before anything
      materializes.
    * ``work`` — ``len(accumulated) * len(base)`` counts this step's exact
      monomial multiplications; the running total must stay inside
      ``_MAX_REPLAY_WORK``.

    Authentic reconstruction prefixes can transiently out-density the
    retained source — later records cancel a dense prefix back into the
    request envelope — so the per-step ceiling is a documented conservative
    replay bound above the serialization envelope instead of the envelope
    itself.  For an authentic claim every partial product divides the
    retained source, so a guard trip proves only that the claim cannot be
    authenticated within bounded work, never a mathematical mismatch.
    """

    step_work = len(accumulated.terms()) * len(base.terms())
    support_bound = step_work
    box_bound = 1
    for accumulated_degree, base_degree in zip(
        _degree_vector(accumulated), _degree_vector(base), strict=True
    ):
        box_bound *= accumulated_degree + base_degree + 1
    if box_bound < support_bound:
        support_bound = box_bound
    if support_bound > _MAX_REPLAY_INTERMEDIATE_TERMS:
        raise ValueError(
            f"{label} factor replay would materialize more than "
            f"{_MAX_REPLAY_INTERMEDIATE_TERMS} intermediate terms"
        )
    work += step_work
    if work > _MAX_REPLAY_WORK:
        raise ValueError(
            f"{label} factor replay exceeds the "
            f"{_MAX_REPLAY_WORK}-operation reconstruction budget"
        )
    return accumulated * base, work


def _validated_replay_records(
    factors: tuple[PolynomialSquareFreeFactor, ...]
    | tuple[PolynomialIrreducibleFactor, ...],
    *,
    mismatch_message: str,
    label: str,
    maximum_exponent: int,
) -> list[tuple[Any, int]]:
    """Budget, convert, and canonically deduplicate the claimed records.

    Every record must sit inside the shared representation envelope, be
    non-constant and monic, and carry a factor polynomial not listed by any
    other record; canonical decompositions never repeat a factor.
    """

    from jacobian.math.polynomials._conversions import (
        rational_polynomial_to_sympy,
    )

    claimed: list[tuple[Any, int]] = []
    seen: set[Any] = set()
    for record in factors:
        require_polynomial_budget(
            record.factor,
            maximum_terms=MAX_POLYNOMIAL_TERMS,
            maximum_exponent=maximum_exponent,
            label=f"{label} factor",
        )
        if _polynomial_total_degree(record.factor) == 0:
            raise ValueError(f"{label} factor must be non-constant")
        base = rational_polynomial_to_sympy(record.factor)
        if base.LC() != 1:
            raise ValueError(mismatch_message)
        key = _canonical_poly_key(base)
        if key in seen:
            raise ValueError(mismatch_message)
        seen.add(key)
        claimed.append((base, record.multiplicity))
    return claimed


def _require_disjoint_square_free_parts(
    claimed: list[tuple[Any, int]], *, mismatch_message: str
) -> None:
    """Reject parts that are not square-free or not pairwise coprime.

    A part fails square-freeness exactly when some irreducible divides it
    together with every nonzero formal partial derivative; two distinct
    parts fail coprimeness exactly when their GCD leaves one variable's
    degree.  Both predicates operate only on the already-admitted claimed
    records.
    """

    for base, _ in claimed:
        derivative_gcd = None
        for generator in base.gens:
            partial = base.diff(generator)
            if partial.is_zero:
                continue
            derivative_gcd = (
                partial if derivative_gcd is None else derivative_gcd.gcd(partial)
            )
        if derivative_gcd is not None and not base.gcd(derivative_gcd).is_one:
            raise ValueError(mismatch_message)
    for index, (base, _) in enumerate(claimed):
        for other, _ in claimed[index + 1 :]:
            if not base.gcd(other).is_one:
                raise ValueError(mismatch_message)


def _verify_exact_factor_product(
    source: Any,
    factors: tuple[PolynomialSquareFreeFactor, ...]
    | tuple[PolynomialIrreducibleFactor, ...],
    *,
    coefficient: CanonicalRational,
    mismatch_message: str,
    label: str,
    maximum_exponent: int,
    require_square_free_parts: bool,
) -> None:
    """Authenticate the claimed records as the unique canonical decomposition.

    Over ``QQ`` (characteristic zero) a decomposition with the following
    exactly-certified properties exists only as THE canonical form of the
    retained source polynomial — the content-and-monic-irreducibles
    factorization when ``require_square_free_parts`` is false, the monic
    square-free decomposition otherwise:

    1. every record is monic, budgeted, non-constant, and distinct;
    2. square-free results carry pairwise-coprime square-free parts;
    3. irreducible-factorization records each satisfy ``is_irreducible``;
    4. ``coefficient * product(record ** multiplicity) == source`` exactly.

    Uniqueness makes these checks equivalent to equality with the recomputed
    canonical decomposition while never invoking a factorization backend:
    recomputing the true decomposition of an admitted source can itself be
    unrepresentable (the multiplicity-one part of
    ``prod((x_i^63 - 1)(x_i - 1))`` holds ``63^5`` terms inside a 1,024-term
    request envelope), so validation must not materialize it to compare.
    Every operation here touches only the claimed records or bounded
    products of them; for an authentic claim every such partial product
    divides the retained source.  Each sparse multiplication's exact work
    count and its structure-derived support bound — the minimum of the
    pairwise-product bound and the per-variable degree box of the claimed
    records — are computed before the backend runs, the support bound per
    step against ``_MAX_REPLAY_INTERMEDIATE_TERMS`` and the cumulative work
    against ``_MAX_REPLAY_WORK``, so nothing larger than those proven bounds
    ever materializes.
    """

    from sympy import Poly, Rational

    if coefficient.as_fraction() == 0 and factors:
        raise ValueError("results with zero content retain no factors")
    claimed = _validated_replay_records(
        factors,
        mismatch_message=mismatch_message,
        label=label,
        maximum_exponent=maximum_exponent,
    )
    accumulated = Poly(
        Rational(*coefficient.as_integer_ratio()),
        *source.gens,
        domain=source.domain,
    )
    work = 0
    for base, multiplicity in claimed:
        if multiplicity == 1:
            accumulated, work = _require_bounded_replay_step(
                accumulated, base, work, label=label
            )
            continue
        power = base
        for _ in range(multiplicity - 1):
            power, work = _require_bounded_replay_step(power, base, work, label=label)
        accumulated, work = _require_bounded_replay_step(
            accumulated, power, work, label=label
        )
    if accumulated != source:
        raise ValueError(mismatch_message)
    if require_square_free_parts:
        _require_disjoint_square_free_parts(claimed, mismatch_message=mismatch_message)
        return
    for base, _ in claimed:
        if not base.is_irreducible:
            raise ValueError(mismatch_message)


class PolynomialGroebnerBudget(StrictModel):
    """Enforced wall and result limits for one isolated Gröbner computation."""

    wall_seconds: StrictInt = Field(default=10, ge=1, le=60)
    maximum_basis_polynomials: StrictInt = Field(default=64, ge=1, le=64)
    maximum_output_terms: StrictInt = Field(default=1024, ge=1, le=1024)


class PolynomialGroebnerBasisRequest(StrictModel):
    generators: tuple[RationalPolynomial, ...] = Field(min_length=1, max_length=16)
    monomial_order: Literal["lex", "grlex", "grevlex"] = "grevlex"
    resource_budget: PolynomialGroebnerBudget = Field(
        default_factory=PolynomialGroebnerBudget
    )

    @model_validator(mode="after")
    def require_groebner_budget(self) -> Self:
        variables = self.generators[0].variables
        if any(generator.variables != variables for generator in self.generators):
            raise ValueError("all ideal generators must use the same ordered ring")
        if sum(len(generator.polynomial.terms) for generator in self.generators) > 256:
            raise ValueError("ideal generators exceed the aggregate term budget")
        for generator in self.generators:
            require_polynomial_budget(
                generator,
                maximum_terms=MAX_POLYNOMIAL_TERMS,
                maximum_exponent=12,
                maximum_coefficient_digits=128,
                label="ideal generator",
            )
            if any(sum(term.exponents) > 12 for term in generator.polynomial.terms):
                raise ValueError("ideal generator exceeds total degree 12")
        return self


class PolynomialGroebnerBasisResult(StrictModel):
    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=1,
        max_length=MAX_POLYNOMIAL_VARIABLES,
    )
    monomial_order: Literal["lex", "grlex", "grevlex"]
    basis: tuple[RationalPolynomial, ...] = Field(max_length=64)
    completion: Literal["COMPLETE"] = "COMPLETE"
    normalization: Literal["REDUCED_MONIC"] = "REDUCED_MONIC"

    @model_validator(mode="after")
    def require_canonical_basis_ring(self) -> Self:
        if any(polynomial.variables != self.variables for polynomial in self.basis):
            raise ValueError("every basis polynomial must use the declared ring")
        if sum(len(polynomial.polynomial.terms) for polynomial in self.basis) > 1024:
            raise ValueError("Gröbner basis exceeds the aggregate output term limit")
        return self


class IntegerPolynomial(StrictModel):
    """Canonical dense polynomial in ``ZZ[x]``, highest degree first."""

    coefficient_order: Literal["DESCENDING_DEGREE"] = "DESCENDING_DEGREE"
    coefficients: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_POLYNOMIAL_TERMS,
    )

    @model_validator(mode="after")
    def require_canonical_coefficients(self) -> Self:
        if len(self.coefficients) > 1 and self.coefficients[0] == "0":
            raise ValueError("leading zero coefficients must be omitted")
        if any(
            len(coefficient.lstrip("-")) > MAX_CANONICAL_RATIONAL_DIGITS
            for coefficient in self.coefficients
        ):
            raise ValueError(
                "integer coefficient exceeds the shared representation limit"
            )
        return self


def _require_integer_polynomial_budget(polynomial: IntegerPolynomial) -> None:
    if len(polynomial.coefficients) > _MAX_ELEMENTARY_DEGREE + 1:
        raise ValueError("integer polynomial exceeds the degree-127 operation budget")
    if any(
        len(coefficient.lstrip("-")) > _MAX_INTEGER_COEFFICIENT_DIGITS
        for coefficient in polynomial.coefficients
    ):
        raise ValueError("integer coefficient exceeds the decimal-digit budget")


class IntegerPolynomialRequest(StrictModel):
    polynomial: IntegerPolynomial

    @model_validator(mode="after")
    def require_operation_budget(self) -> Self:
        _require_integer_polynomial_budget(self.polynomial)
        return self


class IntegerPolynomialShiftRequest(IntegerPolynomialRequest):
    shift: StrictInt = Field(ge=-10_000, le=10_000)


class IntegerPolynomialShiftResult(StrictModel):
    shift: StrictInt = Field(ge=-10_000, le=10_000)
    shifted: IntegerPolynomial
    convention: Literal["SUBSTITUTE_X_PLUS_SHIFT"] = "SUBSTITUTE_X_PLUS_SHIFT"


class IntegerPolynomialPairRequest(StrictModel):
    left: IntegerPolynomial
    right: IntegerPolynomial

    @model_validator(mode="after")
    def require_operation_budget(self) -> Self:
        _require_integer_polynomial_budget(self.left)
        _require_integer_polynomial_budget(self.right)
        return self


class IntegerPolynomialGcdResult(StrictModel):
    gcd: IntegerPolynomial
    left_content: CanonicalInteger
    right_content: CanonicalInteger
    gcd_content: CanonicalInteger
    normalization: Literal["NONNEGATIVE_LEADING_COEFFICIENT"] = (
        "NONNEGATIVE_LEADING_COEFFICIENT"
    )


class IntegerPolynomialContentResult(StrictModel):
    content: CanonicalInteger
    convention: Literal["NONNEGATIVE_COEFFICIENT_GCD"] = "NONNEGATIVE_COEFFICIENT_GCD"


class IntegerPolynomialPrimitivePartResult(StrictModel):
    content: CanonicalInteger
    primitive_part: IntegerPolynomial
    reconstruction: IntegerPolynomial
    convention: Literal["NONNEGATIVE_CONTENT"] = "NONNEGATIVE_CONTENT"


class IntegerPolynomialEvaluationRequest(IntegerPolynomialRequest):
    point: CanonicalInteger

    @model_validator(mode="after")
    def require_bounded_point(self) -> Self:
        if len(self.point.lstrip("-")) > _MAX_INTEGER_COEFFICIENT_DIGITS:
            raise ValueError("evaluation point exceeds the decimal-digit budget")
        return self


class IntegerPolynomialEvaluationResult(StrictModel):
    point: CanonicalInteger
    value: CanonicalInteger


class IntegerPolynomialCompositionRequest(StrictModel):
    outer: IntegerPolynomial
    inner: IntegerPolynomial

    @model_validator(mode="after")
    def require_bounded_output_degree(self) -> Self:
        _require_integer_polynomial_budget(self.outer)
        _require_integer_polynomial_budget(self.inner)
        outer_degree = len(self.outer.coefficients) - 1
        inner_degree = len(self.inner.coefficients) - 1
        if outer_degree * inner_degree > _MAX_ELEMENTARY_DEGREE:
            raise ValueError("composition exceeds the degree-127 output budget")
        return self


class IntegerPolynomialCompositionResult(StrictModel):
    composition: IntegerPolynomial


class RationalPolynomialRequest(StrictModel):
    polynomial: RationalPolynomial

    @model_validator(mode="after")
    def require_univariate_budget(self) -> Self:
        if len(self.polynomial.variables) != 1:
            raise ValueError("elementary polynomial operations require one variable")
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=_MAX_GCD_TERMS,
            maximum_exponent=_MAX_ELEMENTARY_DEGREE,
        )
        return self


class RationalPolynomialDivisionRequest(PolynomialPairRequest):
    @model_validator(mode="after")
    def require_division_budget(self) -> Self:
        if len(self.left.variables) != 1:
            raise ValueError("polynomial division requires one variable")
        if not self.right.polynomial.terms:
            raise ValueError("divisor polynomial must be nonzero")
        for polynomial in (self.left, self.right):
            require_polynomial_budget(
                polynomial,
                maximum_terms=_MAX_GCD_TERMS,
                maximum_exponent=_MAX_ELEMENTARY_DEGREE,
            )
        return self


class RationalPolynomialDivisionResult(StrictModel):
    quotient: RationalPolynomial
    remainder: RationalPolynomial
    reconstruction: RationalPolynomial


class RationalPolynomialEvaluationRequest(RationalPolynomialRequest):
    point: CanonicalRational


class RationalPolynomialEvaluationResult(StrictModel):
    point: CanonicalRational
    value: CanonicalRational


class RationalPolynomialDerivativeResult(StrictModel):
    derivative: RationalPolynomial


class RationalPolynomialIntegralResult(StrictModel):
    antiderivative: RationalPolynomial
    integration_constant: Literal["ZERO"] = "ZERO"


class RationalFunctionRequest(StrictModel):
    numerator: RationalPolynomial
    denominator: RationalPolynomial

    @model_validator(mode="after")
    def require_matching_univariate_ring_and_budget(self) -> Self:
        if self.numerator.variables != self.denominator.variables:
            raise ValueError("numerator and denominator must use the same ring")
        if len(self.numerator.variables) != 1:
            raise ValueError("partial fractions require one variable")
        if not self.denominator.polynomial.terms:
            raise ValueError("denominator polynomial must be nonzero")
        for polynomial in (self.numerator, self.denominator):
            require_polynomial_budget(
                polynomial,
                maximum_terms=_MAX_INVARIANT_TERMS,
                maximum_exponent=_MAX_ELEMENTARY_DEGREE,
            )
        return self


class RationalPartialFractionTerm(StrictModel):
    numerator: RationalPolynomial
    denominator_factor: RationalPolynomial
    denominator_exponent: int = Field(ge=1, le=_MAX_ELEMENTARY_DEGREE)


class RationalPartialFractionResult(StrictModel):
    polynomial_part: RationalPolynomial
    terms: tuple[RationalPartialFractionTerm, ...] = Field(max_length=128)
    reconstruction_numerator: RationalPolynomial
    reconstruction_denominator: RationalPolynomial
    decomposition_field: Literal["QQ"] = "QQ"


__all__ = [
    "IntegerPolynomial",
    "IntegerPolynomialCompositionRequest",
    "IntegerPolynomialCompositionResult",
    "IntegerPolynomialContentResult",
    "IntegerPolynomialEvaluationRequest",
    "IntegerPolynomialEvaluationResult",
    "IntegerPolynomialGcdResult",
    "IntegerPolynomialPairRequest",
    "IntegerPolynomialPrimitivePartResult",
    "IntegerPolynomialRequest",
    "PolynomialBezoutIdentity",
    "PolynomialDiscriminantRequest",
    "PolynomialDiscriminantResult",
    "PolynomialFactorRequest",
    "PolynomialFactorizationResult",
    "PolynomialGcdRequest",
    "PolynomialGcdResult",
    "PolynomialGroebnerBasisRequest",
    "PolynomialGroebnerBasisResult",
    "PolynomialGroebnerBudget",
    "PolynomialInvariantValue",
    "PolynomialIrreducibleFactor",
    "PolynomialPairRequest",
    "PolynomialResultantRequest",
    "PolynomialResultantResult",
    "PolynomialScalarValue",
    "PolynomialSquareFreeDecompositionResult",
    "PolynomialSquareFreeFactor",
    "PolynomialSquareFreeRequest",
    "PolynomialValue",
    "RationalFunctionRequest",
    "RationalPartialFractionResult",
    "RationalPartialFractionTerm",
    "RationalPolynomialDerivativeResult",
    "RationalPolynomialDivisionRequest",
    "RationalPolynomialDivisionResult",
    "RationalPolynomialEvaluationRequest",
    "RationalPolynomialEvaluationResult",
    "RationalPolynomialIntegralResult",
    "RationalPolynomialRequest",
]
