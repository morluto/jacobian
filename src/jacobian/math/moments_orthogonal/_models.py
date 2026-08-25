"""Typed wire contracts for moment-functional operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
)
from jacobian._models import StrictModel
from jacobian.math._rational_height import RationalHeight
from jacobian.math.moments_orthogonal.values import (
    MAX_HANKEL_ORDER,
    MAX_POLYNOMIAL_DEGREE,
    MomentFunctionalPrefix,
    OrthogonalPolynomialFamily,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"moments_orthogonal.{reason}", message)


class MomentsOrthogonalAdmissionError(ValueError):
    """Native admission failure for moments-orthogonal operations."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _require_determinant_representable(
    moments: tuple[CanonicalRational, ...], order: int
) -> None:
    """Bound entry heights so the exact determinant stays canonical.

    The determinant of an (order+1)-square rational matrix carries at most
    roughly (order+1)^2 * H digits for H-digit entries; capping each
    moment's height keeps it inside MAX_CANONICAL_RATIONAL_DIGITS.
    """
    per_entry = MAX_CANONICAL_RATIONAL_DIGITS // ((order + 1) ** 2)
    # A determinant reads 2r+1 consecutive moments; the shifted variant
    # consumes mu_1..mu_(2r+1). Unconsumed moments must not prevent
    # composition.
    for value in moments[: 2 * order + 1]:
        if RationalHeight.from_canonical(value).exceeds(max(per_entry - 2, 8)):
            raise _validation_error(
                "determinant_height",
                f"moment heights exceed the conservative {max(per_entry - 2, 8)}-digit "
                f"bound for an exact order-{order} determinant",
            )


def _require_gram_schmidt_heights_admissible(
    moments: tuple[CanonicalRational, ...], max_degree: int
) -> None:
    """Bound moment heights BEFORE any exact projection runs.

    Monic Gram-Schmidt expresses every derived coefficient and squared
    norm through degree d as a ratio of determinants of at most (d+1)
    -square Hankel matrices over the consumed moments mu_0..mu_2d (the
    classical Cramer-rule form of the elimination). Scaling each row by
    its denominator product bounds an s x s rational determinant's
    numerator and denominator by 10**(s*(s+1)*B + s) when every entry
    carries at most B digits, so bounding each moment by the quotient
    below guarantees every derived value stays canonical. Degree 0
    performs no elimination - its only derived value is mu_0 itself -
    so it needs no input gate beyond canonality.
    """
    if max_degree == 0:
        return
    side = max_degree + 1
    per_entry = (MAX_CANONICAL_RATIONAL_DIGITS - 2 * side) // (2 * side * (side + 1))
    bound = max(per_entry, 8)
    for value in moments[: 2 * max_degree + 1]:
        if RationalHeight.from_canonical(value).exceeds(bound):
            raise MomentsOrthogonalAdmissionError(
                "gram_schmidt_height",
                f"moment heights exceed the conservative {bound}-digit "
                f"bound for exact degree-{max_degree} Gram-Schmidt; supply "
                "a smaller or better-scaled moment prefix",
            )


class HankelRequest(StrictModel):
    """Compute the Hankel matrix H_r from a moment prefix."""

    prefix: MomentFunctionalPrefix
    order: int = Field(ge=0, le=MAX_HANKEL_ORDER)

    @model_validator(mode="after")
    def require_sufficient_moments(self) -> Self:
        needed = 2 * self.order + 1
        if len(self.prefix.moments) < needed:
            raise _validation_error(
                "insufficient_moments",
                f"need at least {needed} moments for order {self.order}, got {len(self.prefix.moments)}",
            )
        _require_determinant_representable(self.prefix.moments, self.order)
        return self


class ShiftedHankelRequest(StrictModel):
    """Compute the shifted Hankel matrix H_r^(1)[i,j] = mu_(i+j+1)."""

    prefix: MomentFunctionalPrefix
    # A shifted matrix of order r consumes mu_1..mu_(2r+1); the canonical
    # prefix holds at most 65 moments, so r = 32 could never validate and
    # must not be advertised as supported.
    order: int = Field(ge=0, le=MAX_HANKEL_ORDER - 1)

    @model_validator(mode="after")
    def require_sufficient_moments(self) -> Self:
        needed = 2 * self.order + 2
        if len(self.prefix.moments) < needed:
            raise _validation_error(
                "insufficient_moments",
                f"need at least {needed} moments for shifted order {self.order}, got {len(self.prefix.moments)}",
            )
        # The shifted determinant consumes mu_1..mu_(2r+1); bound exactly
        # that slice so det H_r^(1) stays canonical.
        _require_determinant_representable(self.prefix.moments[1:], self.order)
        return self


class OrthogonalPolynomialRequest(StrictModel):
    """Compute monic orthogonal polynomials from moments."""

    prefix: MomentFunctionalPrefix
    max_degree: int = Field(ge=0, le=MAX_POLYNOMIAL_DEGREE)

    @model_validator(mode="after")
    def require_sufficient_moments(self) -> Self:
        needed = 2 * self.max_degree + 1
        if len(self.prefix.moments) < needed:
            raise _validation_error(
                "insufficient_moments",
                f"need at least {needed} moments for degree {self.max_degree}, got {len(self.prefix.moments)}",
            )
        return self

    @model_validator(mode="after")
    def require_quasi_definite_prefix(self) -> Self:
        """Replay the exact Gram-Schmidt kernel so a prefix whose orthogonal
        family would hit a zero squared norm is rejected at the boundary
        instead of failing inside execution.

        The conservative height gate runs FIRST: without it, parsing would
        perform every exact projection on unbounded intermediates before
        discovering an over-tall family at wire construction. After the
        gate, both this admission replay and the execution that follows it
        operate on provably bounded intermediates with typed height checks.
        """
        try:
            _require_gram_schmidt_heights_admissible(
                self.prefix.moments, self.max_degree
            )
        except MomentsOrthogonalAdmissionError as exc:
            raise _validation_error(exc.reason, str(exc)) from None
        from jacobian.math.moments_orthogonal.operations import (
            orthogonal_polynomials_from_moments,
        )

        orthogonal_polynomials_from_moments(
            [_m.as_fraction() for _m in self.prefix.moments],
            self.max_degree,
            self.prefix.variable,
        )
        return self


class RecurrenceRequest(StrictModel):
    """Compute three-term recurrence coefficients from a family."""

    family: OrthogonalPolynomialFamily

    @model_validator(mode="after")
    def require_quasi_definite_family(self) -> Self:
        """The kernel divides by every squared norm except the terminal one
        (``beta_k = h_k / h_{k-1}`` for k >= 1); a vanishing interior norm
        would leak ZeroDivisionError instead of a typed result, while a
        vanishing terminal norm leaves ``alpha`` and every ``beta`` exactly
        defined. Admission then replays the exact derivation so every
        emitted ratio is height-checked here — a family such as
        h_0 = 10^-20000 with h_1 = 10^20000 (beta_1 = 10^40000) fails
        parsing, not execution.
        """
        interior = self.family.polynomials[:-1]
        if any(term.squared_norm.as_fraction() == 0 for term in interior):
            raise _validation_error(
                "nonzero_norm",
                "recurrence coefficients require every non-terminal "
                "squared norm to be nonzero",
            )
        from jacobian.math.moments_orthogonal.operations import compute_recurrence

        compute_recurrence(self)
        return self


class ChristoffelDarbouxRequest(StrictModel):
    """Compute the Christoffel-Darboux kernel."""

    family: OrthogonalPolynomialFamily
    degree: int = Field(ge=0)

    @model_validator(mode="after")
    def require_degree_within_family(self) -> Self:
        if self.degree >= len(self.family.polynomials):
            raise _validation_error(
                "degree_out_of_range",
                f"kernel degree {self.degree} exceeds the supplied family "
                f"of {len(self.family.polynomials)} polynomials",
            )
        return self

    @model_validator(mode="after")
    def require_nonzero_norms_through_degree(self) -> Self:
        """The defining sum divides each p_k(x) p_k(y) term by h_k; only
        norms through the requested degree are consumed and gate admission.
        Admission then replays the bounded coefficient construction so an
        over-tall kernel (e.g. p_1 = x + 10^17000 with unit norms at
        degree 1, whose constant coefficient reaches 10^34000 + 1) fails
        parsing instead of raising during execution.
        """
        for term in self.family.polynomials[: self.degree + 1]:
            if term.squared_norm.as_fraction() == 0:
                raise _validation_error(
                    "nonzero_norm",
                    f"Christoffel-Darboux kernel degree {self.degree} "
                    f"requires nonzero squared norms through degree "
                    f"{self.degree}, but p_{term.degree} has a vanishing norm",
                )
        from jacobian.math.moments_orthogonal.operations import (
            compute_christoffel_darboux,
        )

        compute_christoffel_darboux(self)
        return self


class JacobiMatrixRequest(StrictModel):
    """Compute the finite Jacobi matrix."""

    family: OrthogonalPolynomialFamily

    @model_validator(mode="after")
    def require_representable_recurrence(self) -> Self:
        """Pre-computation height bound on the derived recurrence entries.

        Every derived alpha is a difference of admitted family
        coefficients, but representable coefficients do not imply a
        representable difference; each derived beta is an adjacent-norm
        ratio h_k / h_{k-1}: a vanishing norm makes the ratio undefined
        and a representable pair can still exceed the canonical result
        height. Reject such families here so every accepted request can
        return its declared result.
        """
        from fractions import Fraction

        polys = self.family.polynomials
        # A canonical rational carries at most MAX_CANONICAL_RATIONAL_DIGITS
        # digits, i.e. its absolute value stays below 10**that limit.
        digit_limit = 10**MAX_CANONICAL_RATIONAL_DIGITS
        # Derived alphas: alpha_0 = -p_1's constant term; for k >= 1 the
        # residual x*p_k - p_{k+1} carries alpha_k on p_k. Each emitted
        # entry must stay canonical before the operation converts it.
        for k in range(len(polys) - 1):
            p_k = [c.as_fraction() for c in polys[k].coefficients]
            p_next = [c.as_fraction() for c in polys[k + 1].coefficients]
            if k == 0:
                alpha_k = -p_next[0]
            else:
                x_pk = [Fraction(0)] * (len(p_k) + 1)
                for i, coefficient in enumerate(p_k):
                    x_pk[i + 1] = coefficient
                residual = [
                    (x_pk[i] - p_next[i]) if i < len(p_next) else x_pk[i]
                    for i in range(len(x_pk))
                ]
                alpha_k = residual[k] if k < len(residual) else Fraction(0)
            if (
                abs(alpha_k.numerator) >= digit_limit
                or alpha_k.denominator >= digit_limit
            ):
                raise _validation_error(
                    "recurrence_height",
                    f"derived recurrence entry alpha_{k} exceeds the "
                    "canonical rational digit limit; supply a family whose "
                    "coefficient differences stay representable",
                )
        # The operation derives norm ratios h_k/h_{k-1} only for the
        # interior steps that actually appear in the (n-1)-dimensional
        # matrix; terminal ratios are never emitted and must not gate
        # admission.
        for k in range(1, len(polys) - 1):
            h_k = polys[k].squared_norm.as_fraction()
            h_prev = polys[k - 1].squared_norm.as_fraction()
            if h_prev == 0 or h_k == 0:
                raise _validation_error(
                    "norm_ratio",
                    f"adjacent-norm ratio beta_{k} is undefined because "
                    f"squared norm h_{k - 1 if h_prev == 0 else k} vanishes; "
                    "supply a family with nonzero norms for every emitted ratio",
                )
            ratio = h_k / h_prev
            if abs(ratio.numerator) >= digit_limit or ratio.denominator >= digit_limit:
                raise _validation_error(
                    "norm_ratio_height",
                    f"adjacent-norm ratio beta_{k} exceeds the canonical "
                    "rational digit limit; supply a family whose squared "
                    "norm ratios stay representable",
                )
        return self


class GaussianQuadratureRequest(StrictModel):
    """Compute an exact Gaussian quadrature rule."""

    prefix: MomentFunctionalPrefix
    order: int = Field(ge=1, le=16)

    @model_validator(mode="after")
    def require_sufficient_moments(self) -> Self:
        # The conservative Gram-Schmidt height gate runs BEFORE any exact
        # projection: without it, a single schema-valid payload such as
        # mu_0 = 10^-32767 with mu_1 = 10^32767 forces enormous exact
        # backend work during parsing before the derived-node check fires.
        try:
            _require_gram_schmidt_heights_admissible(self.prefix.moments, self.order)
        except MomentsOrthogonalAdmissionError as exc:
            raise _validation_error(exc.reason, str(exc)) from None
        # Building p_order projects only onto earlier polynomials, so the
        # Gram-Schmidt kernel and the Vandermonde weight solve consume
        # moments through mu_(2n-1) exactly; execution verifies exactness
        # through degree 2n-1, so 2n moments are both sufficient and
        # required.
        needed = 2 * self.order
        if len(self.prefix.moments) < needed:
            raise _validation_error(
                "insufficient_moments",
                f"need at least {needed} moments for quadrature order {self.order}, got {len(self.prefix.moments)}",
            )
        return self

    @model_validator(mode="after")
    def require_rational_nodes(self) -> Self:
        """Admit only prefixes whose degree-n orthogonal polynomial splits
        into distinct linear factors over QQ.

        The exact-node contract carries canonical rationals; algebraic
        nodes such as +-sqrt(1/3) cannot be represented, so such prefixes
        are rejected here instead of failing during execution. Gaussian
        construction divides by norms only through p_{n-1}, so a vanishing
        terminal norm (a measure supported on exactly n points) stays
        admissible.
        """
        from fractions import Fraction

        import sympy

        from jacobian.math.moments_orthogonal.operations import (
            _build_quadrature_rule,
            _construct_monic_orthogonal_polynomial,
            _fraction_exceeds_canonical_limit,
        )

        moments = [Fraction(*v.as_integer_ratio()) for v in self.prefix.moments]
        coefficients = _construct_monic_orthogonal_polynomial(moments, self.order)
        x = sympy.Symbol(self.prefix.variable)
        poly = sum(coefficient * x**i for i, coefficient in enumerate(coefficients))
        _, factors = sympy.factor_list(poly)
        if any(
            sympy.degree(factor, x) != 1 or multiplicity != 1
            for factor, multiplicity in factors
        ):
            raise _validation_error(
                "rational_nodes",
                f"quadrature order {self.order} requires p_{self.order} to "
                "split into distinct linear factors over QQ so every node "
                "is an exact rational; this moment prefix yields algebraic "
                "or repeated nodes",
            )
        # Positive weights are part of the declared contract; replay the
        # exact construction so a nonpositive weight is rejected here
        # instead of raising during execution.
        _nodes, weights = _build_quadrature_rule(self.prefix, self.order)
        # Derived nodes and weights can leave the canonical range even when
        # every input moment stays inside it; measure the exact Fractions
        # before execution converts them.
        if any(
            _fraction_exceeds_canonical_limit(value) for value in (*_nodes, *weights)
        ):
            raise _validation_error(
                "quadrature_height",
                "derived quadrature nodes or weights exceed the canonical "
                "rational digit limit; supply a moment prefix whose exact "
                "rule stays representable",
            )
        if any(weight <= 0 for weight in weights):
            raise _validation_error(
                "positive_weights",
                "quadrature admission requires strictly positive weights; "
                "this moment prefix yields a nonpositive weight",
            )
        return self


__all__ = [
    "ChristoffelDarbouxRequest",
    "GaussianQuadratureRequest",
    "HankelRequest",
    "JacobiMatrixRequest",
    "OrthogonalPolynomialRequest",
    "RecurrenceRequest",
    "ShiftedHankelRequest",
]


MomentFunctionalPrefix.model_rebuild()
HankelRequest.model_rebuild()
ShiftedHankelRequest.model_rebuild()
OrthogonalPolynomialRequest.model_rebuild()
ChristoffelDarbouxRequest.model_rebuild()
GaussianQuadratureRequest.model_rebuild()
