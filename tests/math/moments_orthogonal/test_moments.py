"""Tests for moment-functional and orthogonal-polynomial operations (#1900)."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.moments_orthogonal._models import (
    ChristoffelDarbouxRequest,
    GaussianQuadratureRequest,
    HankelRequest,
    JacobiMatrixRequest,
    OrthogonalPolynomialFamily,
    OrthogonalPolynomialRequest,
    RecurrenceRequest,
    ShiftedHankelRequest,
)
from jacobian.math.moments_orthogonal.operations import (
    compute_christoffel_darboux,
    compute_gaussian_quadrature,
    compute_hankel_matrix,
    compute_jacobi_matrix,
    compute_orthogonal_polynomials,
    compute_recurrence,
    compute_shifted_hankel,
)
from jacobian.math.moments_orthogonal.values import (
    GaussianQuadratureRule,
    QuadratureNode,
)


def _prefix(moments):
    from jacobian.math.moments_orthogonal.values import MomentFunctionalPrefix

    return MomentFunctionalPrefix(moments=moments, variable="x")


def _moments_uniform(n: int) -> tuple[CanonicalRational, ...]:
    """Uniform measure on [-1,1]: mu_k = 2/(k+1) for even k, 0 for odd k."""
    return tuple(
        CanonicalRational(num="2", den=str(k + 1))
        if k % 2 == 0
        else CanonicalRational(num="0", den="1")
        for k in range(n)
    )


class TestHankel:
    def test_hankel_order_0(self) -> None:
        result = compute_hankel_matrix(
            HankelRequest(prefix=_prefix(_moments_uniform(3)), order=0)
        )
        assert result.order == 0
        assert len(result.entries) == 1
        assert len(result.entries[0]) == 1
        assert int(result.entries[0][0].num) == 2
        assert int(result.entries[0][0].den) == 1
        assert int(result.determinant.num) == 2
        assert result.rank == 1

    def test_hankel_order_2(self) -> None:
        result = compute_hankel_matrix(
            HankelRequest(prefix=_prefix(_moments_uniform(5)), order=2)
        )
        assert result.order == 2
        assert result.rank == 3
        # det should be positive (positive definite)
        assert int(result.determinant.num) * int(result.determinant.den) > 0
        assert int(result.determinant.den) > 0

    def test_insufficient_moments(self) -> None:
        with pytest.raises(ValidationError):
            HankelRequest(prefix=_prefix(_moments_uniform(3)), order=2)


class TestShiftedHankel:
    def test_shifted_hankel(self) -> None:
        result = compute_shifted_hankel(
            ShiftedHankelRequest(prefix=_prefix(_moments_uniform(6)), order=2)
        )
        assert result.order == 2

    def test_consumed_moment_heights_bound_admission(self) -> None:
        """The shifted determinant consumes mu_1..mu_(2r+1); an extreme
        moment inside that slice must be rejected at admission instead of
        failing when the canonical determinant is constructed."""
        moments = (
            CanonicalRational(num="1", den="1"),
            CanonicalRational.from_fraction(Fraction(10) ** 8000),
            CanonicalRational(num="0", den="1"),
            CanonicalRational.from_fraction(Fraction(10) ** 32767),
        )
        with pytest.raises(ValidationError):
            ShiftedHankelRequest(prefix=_prefix(moments), order=1)

    def test_unconsumed_moments_do_not_gate_admission(self) -> None:
        """Moments beyond mu_(2r+1) are not read by the shifted determinant
        and must not prevent composition."""
        moments = (
            *_moments_uniform(4),
            CanonicalRational.from_fraction(Fraction(10) ** 32000),
        )
        request = ShiftedHankelRequest(prefix=_prefix(moments), order=1)
        assert compute_shifted_hankel(request).order == 1


class TestOrthogonalPolynomials:
    def test_uniform_gives_legendre(self) -> None:
        result = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(
                prefix=_prefix(_moments_uniform(7)), max_degree=3
            )
        )
        assert result.is_quasi_definite
        assert result.is_positive_definite

        # p_0 = 1
        p0 = result.polynomials[0]
        assert len(p0.coefficients) == 1
        assert int(p0.coefficients[0].num) == 1

        # p_1 = x (alpha_0 = 0 for symmetric measure)
        p1 = result.polynomials[1]
        assert int(p1.coefficients[0].num) == 0  # constant term is 0
        assert int(p1.coefficients[1].num) == 1  # leading coefficient is 1 (monic)

        # p_2 = x^2 - 1/3
        p2 = result.polynomials[2]
        assert int(p2.coefficients[2].num) == 1  # monic
        assert int(p2.coefficients[0].num) == -1  # constant term -1/3
        assert int(p2.coefficients[0].den) == 3

        # Norms: h_0=2, h_1=2/3, h_2=8/45
        assert int(result.polynomials[0].squared_norm.num) == 2
        assert int(result.polynomials[0].squared_norm.den) == 1
        assert int(result.polynomials[1].squared_norm.num) == 2
        assert int(result.polynomials[1].squared_norm.den) == 3

    def test_insufficient_moments(self) -> None:
        with pytest.raises(ValidationError):
            OrthogonalPolynomialRequest(
                prefix=_prefix(_moments_uniform(3)), max_degree=2
            )

    def test_zero_norm_prefix_rejected(self) -> None:
        """moments [1,0,0,...]: p_1 has zero norm; no orthogonal family
        through higher degrees exists and the boundary must reject."""
        moments = (
            CanonicalRational(num="1", den="1"),
            CanonicalRational(num="0", den="1"),
            CanonicalRational(num="0", den="1"),
            CanonicalRational(num="0", den="1"),
            CanonicalRational(num="0", den="1"),
        )
        with pytest.raises(ValueError):
            compute_orthogonal_polynomials(
                OrthogonalPolynomialRequest(prefix=_prefix(moments), max_degree=2)
            )
        # The same prefix is rejected at request admission.
        with pytest.raises(ValueError):
            OrthogonalPolynomialRequest(prefix=_prefix(moments), max_degree=2)

    def test_quasi_definite_is_not_positive_definite(self) -> None:
        """Moments (1, 0, -1): norms are 1 and -1 - quasi-definite but not
        positive-definite."""
        moments = (
            CanonicalRational(num="1", den="1"),
            CanonicalRational(num="0", den="1"),
            CanonicalRational(num="-1", den="1"),
        )
        result = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(prefix=_prefix(moments), max_degree=1)
        )
        assert result.is_quasi_definite is True
        assert result.is_positive_definite is False


class TestRecurrence:
    def test_recurrence_from_legendre(self) -> None:
        family = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(
                prefix=_prefix(_moments_uniform(7)), max_degree=3
            )
        )
        rec = compute_recurrence(RecurrenceRequest(family=family))

        # alpha_0..alpha_2 are determined by adjacent polynomials; the
        # terminal alpha_3 is undetermined without p_4 and must be omitted.
        assert len(rec.alpha) == 3
        for a in rec.alpha:
            assert int(a.num) == 0

        # beta[0] is a placeholder; beta_k = h_k / h_{k-1}.
        # beta_1 = 1/3, beta_2 = 4/15, beta_3 = 9/35
        assert len(rec.beta) == 4
        assert int(rec.beta[1].num) == 1
        assert int(rec.beta[1].den) == 3

    def test_singleton_family_has_no_determined_alpha(self) -> None:
        """A family containing only p_0=1 determines no recurrence
        coefficient: mu_1/mu_0 is arbitrary and must not be reported as 0."""
        family = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(
                prefix=_prefix(
                    (
                        CanonicalRational(num="2", den="1"),
                        CanonicalRational(num="5", den="1"),
                        CanonicalRational(num="9", den="1"),
                    )
                ),
                max_degree=0,
            )
        )
        rec = compute_recurrence(RecurrenceRequest(family=family))
        assert len(rec.alpha) == 0


class TestChristoffelDarboux:
    def test_cd_kernel_degree_0(self) -> None:
        family = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(
                prefix=_prefix(_moments_uniform(7)), max_degree=3
            )
        )
        result = compute_christoffel_darboux(
            ChristoffelDarbouxRequest(family=family, degree=0)
        )
        # K_0(x,y) = p_0(x) p_0(y) / h_0 = 1/2
        assert result.coefficients == ((CanonicalRational(num="1", den="2"),),)

    def test_cd_kernel_is_bivariate(self) -> None:
        """Degree-1 Legendre-like kernel is 1/2 + (3/2) x y, not the
        diagonal specialization 1/2 + (3/2) x^2."""
        family = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(
                prefix=_prefix(_moments_uniform(7)), max_degree=1
            )
        )
        result = compute_christoffel_darboux(
            ChristoffelDarbouxRequest(family=family, degree=1)
        )
        half = CanonicalRational(num="1", den="2")
        zero = CanonicalRational(num="0", den="1")
        three_halves = CanonicalRational(num="3", den="2")
        assert result.coefficients == (
            (half, zero),
            (zero, three_halves),
        )
        # Off-diagonal evaluation K_1(2, 3) = 1/2 + (3/2)*6 = 37/4 replays.

        def evaluate(matrix, xv, yv):
            return sum(
                Fraction(int(entry.num), int(entry.den)) * xv**i * yv**j
                for i, row in enumerate(matrix)
                for j, entry in enumerate(row)
            )

        assert evaluate(result.coefficients, Fraction(2), Fraction(3)) == Fraction(
            19, 2
        )

    def test_zero_norm_rejected_at_request_boundary(self) -> None:
        """A singleton family with a vanishing norm is authorable; the
        kernel request must reject it instead of raising mid-execution."""
        from jacobian.math.moments_orthogonal.values import (
            OrthogonalPolynomialFamily,
            OrthogonalPolynomialTerm,
        )

        family = OrthogonalPolynomialFamily(
            polynomials=(
                OrthogonalPolynomialTerm(
                    degree=0,
                    coefficients=(CanonicalRational(num="1", den="1"),),
                    squared_norm=CanonicalRational(num="0", den="1"),
                ),
            ),
            variable="x",
            is_quasi_definite=False,
            is_positive_definite=False,
        )
        with pytest.raises(ValidationError):
            ChristoffelDarbouxRequest(family=family, degree=0)

    def test_zero_norm_beyond_degree_does_not_gate(self) -> None:
        """Only norms through the requested degree divide in the defining
        sum; a degenerate later term must not reject the smaller kernel."""
        from jacobian.math.moments_orthogonal.values import (
            OrthogonalPolynomialFamily,
            OrthogonalPolynomialTerm,
        )

        family = OrthogonalPolynomialFamily(
            polynomials=(
                OrthogonalPolynomialTerm(
                    degree=0,
                    coefficients=(CanonicalRational(num="1", den="1"),),
                    squared_norm=CanonicalRational(num="2", den="1"),
                ),
                OrthogonalPolynomialTerm(
                    degree=1,
                    coefficients=(
                        CanonicalRational(num="0", den="1"),
                        CanonicalRational(num="1", den="1"),
                    ),
                    squared_norm=CanonicalRational(num="0", den="1"),
                ),
            ),
            variable="x",
            is_quasi_definite=False,
            is_positive_definite=False,
        )
        result = compute_christoffel_darboux(
            ChristoffelDarbouxRequest(family=family, degree=0)
        )
        assert result.coefficients == ((CanonicalRational(num="1", den="2"),),)


class TestJacobiMatrix:
    def test_jacobi_matrix(self) -> None:
        family = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(
                prefix=_prefix(_moments_uniform(7)), max_degree=3
            )
        )
        result = compute_jacobi_matrix(JacobiMatrixRequest(family=family))
        # 3x3 matrix: n=4 polynomials, so the multiplication operator acts
        # on p_0..p_2.
        assert len(result.matrix) == 3
        assert len(result.alphas) == 3
        # Diagonal should all be 0 for symmetric measure
        for i in range(3):
            assert int(result.matrix[i][i].num) == 0
        # Monic basis: subdiagonal carries 1, superdiagonal carries beta.
        # beta_1 = 1/3 and beta_2 = 4/15 for Legendre moments; betas keeps
        # the recurrence convention with an unused placeholder first.
        one = CanonicalRational(num="1", den="1")
        third = CanonicalRational(num="1", den="3")
        four_fifteenths = CanonicalRational(num="4", den="15")
        assert result.betas == (
            CanonicalRational(num="0", den="1"),
            third,
            four_fifteenths,
        )
        assert result.matrix[1][0] == one
        assert result.matrix[2][1] == one
        assert result.matrix[0][1] == third
        assert result.matrix[1][2] == four_fifteenths

    def test_singleton_family_yields_empty_jacobi_matrix(self) -> None:
        """A family holding only p_0 admits the empty multiplication
        operator; the value validates without indexing a placeholder beta."""
        from jacobian.math.moments_orthogonal.values import (
            OrthogonalPolynomialFamily,
            OrthogonalPolynomialTerm,
        )

        family = OrthogonalPolynomialFamily(
            polynomials=(
                OrthogonalPolynomialTerm(
                    degree=0,
                    coefficients=(CanonicalRational(num="1", den="1"),),
                    squared_norm=CanonicalRational(num="2", den="1"),
                ),
            ),
            variable="x",
            is_quasi_definite=True,
            is_positive_definite=True,
        )
        result = compute_jacobi_matrix(JacobiMatrixRequest(family=family))
        assert result.alphas == ()
        assert result.betas == ()
        assert result.matrix == ()
        assert result.variable == "x"


class TestGaussianQuadrature:
    def _moments_rational_nodes(self) -> tuple:
        """Measure with weight 7 at +-1 and 5 at +-2: mu_(2j) = 14 + 10*4^j,
        odd moments 0. mu_2/mu_0 = 54/24 = 9/4, so p_2 = x^2 - 9/4 with
        rational nodes +-3/2."""
        values = ("24", "0", "54", "0", "174")
        return tuple(
            CanonicalRational(num=v, den="1")
            if v != "0"
            else CanonicalRational(num="0", den="1")
            for v in values
        )

    def test_exact_rule_for_rational_nodes(self) -> None:
        from jacobian.math.moments_orthogonal.operations import (
            compute_gaussian_quadrature,
        )

        result = compute_gaussian_quadrature(
            GaussianQuadratureRequest(
                prefix=_prefix(self._moments_rational_nodes()), order=2
            )
        )
        assert result.order == 2
        assert len(result.nodes) == 2
        nodes = {(int(n.node.num), int(n.node.den)) for n in result.nodes}
        assert nodes == {(-3, 2), (3, 2)}
        for node in result.nodes:
            assert (int(node.weight.num), int(node.weight.den)) == (12, 1)
        assert result.exactness_degree == 3

    def test_requires_two_n_moments(self) -> None:
        """Construction consumes moments through mu_(2n-1) exactly: 2n
        moments suffice for an exact order-n rule and 2n-1 do not."""

        with pytest.raises(ValueError):
            GaussianQuadratureRequest(
                prefix=_prefix(self._moments_rational_nodes()[:3]), order=2
            )
        request = GaussianQuadratureRequest(
            prefix=_prefix(self._moments_rational_nodes()[:4]), order=2
        )
        result = compute_gaussian_quadrature(request)
        assert {n.node.as_fraction() for n in result.nodes} == {
            Fraction(-3, 2),
            Fraction(3, 2),
        }
        assert result.exactness_degree == 3

    def test_order_one_prefix_needs_only_two_moments(self) -> None:
        """An order-1 prefix (mu_0, mu_1) = (1, 2) determines the exact
        rule with node 2, weight 1, and exactness through degree 1; the
        unused mu_2 must not be required."""
        from jacobian.math.moments_orthogonal.operations import (
            compute_gaussian_quadrature,
        )

        result = compute_gaussian_quadrature(
            GaussianQuadratureRequest(
                prefix=_prefix(
                    (
                        CanonicalRational(num="1", den="1"),
                        CanonicalRational(num="2", den="1"),
                    )
                ),
                order=1,
            )
        )
        assert result.order == 1
        assert len(result.nodes) == 1
        assert (int(result.nodes[0].node.num), int(result.nodes[0].node.den)) == (2, 1)
        assert (int(result.nodes[0].weight.num), int(result.nodes[0].weight.den)) == (
            1,
            1,
        )
        assert result.exactness_degree == 1

    def test_algebraic_nodes_rejected_at_admission(self) -> None:
        """Uniform [0,1] moments give p_2 = x^2 - x + 1/6 with irrational
        roots; the canonical rational node contract cannot carry them, so
        admission rejects instead of crashing on .p/.q access."""

        uniform = tuple(CanonicalRational(num="1", den=str(k + 1)) for k in range(5))
        with pytest.raises(ValueError):
            GaussianQuadratureRequest(prefix=_prefix(uniform), order=2)


class TestQuadratureSourceBinding:
    def _prefix(self):
        from jacobian.math.moments_orthogonal.values import (
            MomentFunctionalPrefix,
        )

        moments = tuple(
            CanonicalRational(num=v, den="1") for v in ("24", "0", "54", "0", "174")
        )
        return MomentFunctionalPrefix(moments=moments, variable="x")

    def test_rule_retains_prefix_and_replays(self) -> None:
        result = compute_gaussian_quadrature(
            GaussianQuadratureRequest(prefix=self._prefix(), order=2)
        )
        assert result.prefix == self._prefix()
        revalidated = GaussianQuadratureRule.model_validate(result.model_dump())
        assert revalidated.exactness_degree == 3

        payload = result.model_dump()
        payload["exactness_degree"] = 999
        with pytest.raises(ValidationError):
            GaussianQuadratureRule.model_validate(payload)

    def test_node_count_matches_order(self) -> None:
        one_node = QuadratureNode(
            node={"num": "0", "den": "1"}, weight={"num": "1", "den": "1"}
        )
        with pytest.raises(ValueError):
            GaussianQuadratureRule(
                order=2,
                nodes=(one_node,),
                variable="x",
                exactness_degree=3,
                prefix=self._prefix(),
            )

    def test_insufficient_source_prefix_rejected(self) -> None:
        """A serialized order-1 rule retaining only mu_0 could never be
        produced by the request boundary, which requires moments through
        mu_(2n-1); the replay must reject it instead of silently reading
        missing moments as zero."""
        from jacobian.math.moments_orthogonal.values import (
            MomentFunctionalPrefix,
        )

        minimal = MomentFunctionalPrefix(
            moments=(CanonicalRational(num="1", den="1"),), variable="x"
        )
        with pytest.raises(ValidationError):
            GaussianQuadratureRule(
                order=1,
                nodes=(
                    QuadratureNode(
                        node=CanonicalRational(num="0", den="1"),
                        weight=CanonicalRational(num="1", den="1"),
                    ),
                ),
                variable="x",
                exactness_degree=1,
                prefix=minimal,
            )


class TestJacobiCrossField:
    def test_contradictory_jacobi_rejected(self) -> None:
        from jacobian.math.moments_orthogonal.values import JacobiMatrix

        with pytest.raises(ValidationError):
            JacobiMatrix(
                alphas=(CanonicalRational(num="0", den="1"),),
                betas=(CanonicalRational(num="0", den="1"),),
                matrix=((CanonicalRational(num="1", den="1"),),),
                variable="x",
            )

    def test_off_band_entry_rejected(self) -> None:
        """A nonzero entry outside the tridiagonal band cannot revalidate
        as a Jacobi matrix, even with every diagonal and off-diagonal
        entry consistent."""
        from jacobian.math.moments_orthogonal.values import JacobiMatrix

        family = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(
                prefix=_prefix(_moments_uniform(7)), max_degree=3
            )
        )
        result = compute_jacobi_matrix(JacobiMatrixRequest(family=family))
        rows = [list(row) for row in result.matrix]
        assert len(rows) == 3
        rows[0][2] = CanonicalRational(num="7", den="5")
        payload = result.model_dump()
        payload["matrix"] = [
            [{"num": c.num, "den": c.den} for c in row] for row in rows
        ]
        with pytest.raises(ValidationError):
            JacobiMatrix.model_validate(payload)


class TestFamilyResidualBasisCheck:
    def _term(self, deg, coeffs, norm):
        from jacobian.math.moments_orthogonal.values import (
            OrthogonalPolynomialTerm,
        )

        return OrthogonalPolynomialTerm(
            degree=deg,
            coefficients=tuple(
                CanonicalRational.from_fraction(Fraction(c)) for c in coeffs
            ),
            squared_norm=CanonicalRational.from_fraction(Fraction(norm)),
        )

    def test_residual_component_below_p_prev_rejected(self) -> None:
        """p_2 = x^2 + 1 with unit norms reconstructs x^2 - 1; the residual
        decomposition in the p_0..p_k basis exposes the forgery."""
        from jacobian.math.moments_orthogonal.values import (
            OrthogonalPolynomialFamily,
        )

        with pytest.raises(ValidationError):
            OrthogonalPolynomialFamily(
                polynomials=(
                    self._term(0, (1,), 1),
                    self._term(1, (0, 1), 1),
                    self._term(2, (1, 0, 1), 1),
                ),
                variable="x",
                is_quasi_definite=True,
                is_positive_definite=True,
            )

    def test_consistent_family_accepted(self) -> None:
        """The Legendre prefix p_0=1, p_1=x, p_2=x^2-1/3 with h=(1,1/3,1/45)
        satisfies the recurrence and every norm ratio."""
        from jacobian.math.moments_orthogonal.values import (
            OrthogonalPolynomialFamily,
        )

        family = OrthogonalPolynomialFamily(
            polynomials=(
                self._term(0, (1,), 1),
                self._term(1, (0, 1), Fraction(1, 3)),
                self._term(2, (Fraction(-1, 3), 0, 1), Fraction(1, 45)),
            ),
            variable="x",
            is_quasi_definite=True,
            is_positive_definite=True,
        )
        assert len(family.polynomials) == 3


class TestDegenerateNormRecurrenceIdentities:
    def _term(self, deg, coeffs, norm):
        from jacobian.math.moments_orthogonal.values import (
            OrthogonalPolynomialTerm,
        )

        return OrthogonalPolynomialTerm(
            degree=deg,
            coefficients=tuple(
                CanonicalRational.from_fraction(Fraction(c)) for c in coeffs
            ),
            squared_norm=CanonicalRational.from_fraction(Fraction(norm)),
        )

    def test_impossible_degenerate_family_rejected(self) -> None:
        """p_0=1,h_0=0; p_1=x,h_1=1; p_2=x^2,h_2=1 claims L(x^2)=1 through
        h_1 while orthogonality of p_0 and p_2 forces L(x^2)=0; no linear
        functional realizes it, so the zero-safe relation h_k = beta_k *
        h_{k-1} must reject it instead of skipping degenerate norms."""
        from jacobian.math.moments_orthogonal.values import (
            OrthogonalPolynomialFamily,
        )

        with pytest.raises(ValidationError):
            OrthogonalPolynomialFamily(
                polynomials=(
                    self._term(0, (1,), 0),
                    self._term(1, (0, 1), 1),
                    self._term(2, (0, 0, 1), 1),
                ),
                variable="x",
                is_quasi_definite=False,
                is_positive_definite=False,
            )

    def test_all_zero_norms_remain_admitted(self) -> None:
        """The zero functional realizes every-vanishing norms, so the fully
        degenerate family stays authorable."""
        from jacobian.math.moments_orthogonal.values import (
            OrthogonalPolynomialFamily,
        )

        family = OrthogonalPolynomialFamily(
            polynomials=(
                self._term(0, (1,), 0),
                self._term(1, (0, 1), 0),
                self._term(2, (0, 0, 1), 0),
            ),
            variable="x",
            is_quasi_definite=False,
            is_positive_definite=False,
        )
        assert len(family.polynomials) == 3

    def test_two_term_degenerate_pair_stays_admitted(self) -> None:
        """A two-polynomial family determines no beta relation (that needs
        three consecutive terms), and any norm pair is realizable - e.g.
        L = (0, 0, 1) realizes h_0 = 0 with h_1 = 1 for p_1 = x + 5 - so
        the degenerate pair stays authorable."""
        from jacobian.math.moments_orthogonal.values import (
            OrthogonalPolynomialFamily,
        )

        admitted = OrthogonalPolynomialFamily(
            polynomials=(
                self._term(0, (1,), 0),
                self._term(1, (5, 1), 1),
            ),
            variable="x",
            is_quasi_definite=False,
            is_positive_definite=False,
        )
        assert len(admitted.polynomials) == 2


class TestGramSchmidtHeightAdmission:
    def test_over_tall_prefix_rejected_before_expansion(self) -> None:
        """mu_0 = 10^-20000 and mu_1 = 10^20000 force derived Gram-Schmidt
        values beyond the canonical range; the conservative pre-expansion
        gate rejects the prefix before any projection runs instead of
        discovering the overflow at wire construction."""
        moments = (
            CanonicalRational.from_fraction(Fraction(1) / Fraction(10) ** 20000),
            CanonicalRational.from_fraction(Fraction(10) ** 20000),
            CanonicalRational(num="0", den="1"),
        )
        with pytest.raises(ValidationError):
            OrthogonalPolynomialRequest(prefix=_prefix(moments), max_degree=1)

    def test_bounded_prefix_still_admits_and_executes(self) -> None:
        """A prefix inside the conservative bound keeps admitting, including
        a quasi-definite but not positive-definite family."""
        moments = (
            CanonicalRational(num="1", den="1"),
            CanonicalRational(num="0", den="1"),
            CanonicalRational(num="-1", den="1"),
        )
        request = OrthogonalPolynomialRequest(prefix=_prefix(moments), max_degree=1)
        result = compute_orthogonal_polynomials(request)
        assert result.is_quasi_definite is True
        assert result.is_positive_definite is False

    def test_unconsumed_moments_do_not_gate(self) -> None:
        """Gram-Schmidt through degree d reads mu_0..mu_2d only; taller
        unconsumed moments must not prevent composition."""
        moments = (
            *_moments_uniform(5),
            CanonicalRational.from_fraction(Fraction(10) ** 32000),
        )
        request = OrthogonalPolynomialRequest(prefix=_prefix(moments), max_degree=2)
        assert len(compute_orthogonal_polynomials(request).polynomials) == 3

    def test_kernel_reports_over_tall_family_with_typed_error(self) -> None:
        """Bypassing admission via unvalidated native construction, the
        kernel itself reports the over-tall derived value as a typed domain
        error instead of failing inside canonical conversion."""
        moments = (
            CanonicalRational.from_fraction(Fraction(1) / Fraction(10) ** 20000),
            CanonicalRational.from_fraction(Fraction(10) ** 20000),
            CanonicalRational(num="0", den="1"),
        )
        request = OrthogonalPolynomialRequest.model_construct(
            prefix=_prefix(moments), max_degree=1
        )
        with pytest.raises(ValueError):
            compute_orthogonal_polynomials(request)


class TestQuadratureMinimalPrefixRoundTrip:
    def test_order_one_two_moment_prefix_round_trips(self) -> None:
        """The exact order-1 rule determined by (mu_0, mu_1) = (1, 2)
        alone survives serialization and replays against its retained
        minimal prefix."""
        from jacobian.math.moments_orthogonal.values import GaussianQuadratureRule

        moments = (
            CanonicalRational(num="1", den="1"),
            CanonicalRational(num="2", den="1"),
        )
        request = GaussianQuadratureRequest(prefix=_prefix(moments), order=1)
        result = compute_gaussian_quadrature(request)
        assert GaussianQuadratureRule.model_validate(result.model_dump()) == result


class TestRecurrenceTupleDimensions:
    def test_contradictory_dimensions_rejected(self) -> None:
        from jacobian.math.moments_orthogonal.values import ThreeTermRecurrence

        with pytest.raises(ValidationError):
            ThreeTermRecurrence(
                alpha=(CanonicalRational(num="0", den="1"),),
                beta=(),
                variable="x",
            )

    def test_nonzero_placeholder_rejected(self) -> None:
        from jacobian.math.moments_orthogonal.values import ThreeTermRecurrence

        with pytest.raises(ValidationError):
            ThreeTermRecurrence(
                alpha=(),
                beta=(CanonicalRational(num="5", den="1"),),
                variable="x",
            )


class TestJacobiNormRatioAdmission:
    def test_unused_terminal_ratio_does_not_gate_admission(self) -> None:
        """For a two-polynomial family the operation returns [[alpha_0]] and
        never emits h_1/h_0, so an extreme terminal norm ratio must not
        reject an otherwise fully representable result."""

        def term(deg, coeffs, norm):
            from jacobian.math.moments_orthogonal.values import (
                OrthogonalPolynomialTerm,
            )

            return OrthogonalPolynomialTerm(
                degree=deg,
                coefficients=tuple(
                    CanonicalRational.from_fraction(Fraction(c)) for c in coeffs
                ),
                squared_norm=CanonicalRational.from_fraction(Fraction(norm)),
            )

        from jacobian.math.moments_orthogonal.values import (
            OrthogonalPolynomialFamily,
        )

        family = OrthogonalPolynomialFamily(
            polynomials=(
                term(0, (1,), Fraction(1, 10**20000)),
                term(1, (0, 1), Fraction(10**20000)),
            ),
            variable="x",
            is_quasi_definite=True,
            is_positive_definite=True,
        )
        request = JacobiMatrixRequest(family=family)
        result = compute_jacobi_matrix(request)
        assert [a.as_fraction() for a in result.alphas] == [Fraction(0)]

    def test_zero_norm_ratio_rejected_at_request_boundary(self) -> None:
        """p_0=1, p_1=x, p_2=x^2 with all-zero norms is authorable; parsing
        the Jacobi request must reject the undefined ratio instead of
        leaking ZeroDivisionError."""
        from jacobian.math.moments_orthogonal.values import (
            OrthogonalPolynomialFamily,
            OrthogonalPolynomialTerm,
        )

        zero = CanonicalRational(num="0", den="1")
        one = CanonicalRational(num="1", den="1")
        family = OrthogonalPolynomialFamily(
            polynomials=(
                OrthogonalPolynomialTerm(
                    degree=0, coefficients=(one,), squared_norm=zero
                ),
                OrthogonalPolynomialTerm(
                    degree=1, coefficients=(zero, one), squared_norm=zero
                ),
                OrthogonalPolynomialTerm(
                    degree=2, coefficients=(zero, zero, one), squared_norm=zero
                ),
            ),
            variable="x",
            is_quasi_definite=False,
            is_positive_definite=False,
        )
        with pytest.raises(ValidationError):
            JacobiMatrixRequest(family=family)

    def test_emitted_ratio_free_family_admits_zero_terminal_norm(self) -> None:
        """A two-polynomial family emits no norm ratio; a vanishing
        terminal norm must not gate admission of [[alpha_0]]."""
        from jacobian.math.moments_orthogonal.values import (
            OrthogonalPolynomialFamily,
            OrthogonalPolynomialTerm,
        )

        family = OrthogonalPolynomialFamily(
            polynomials=(
                OrthogonalPolynomialTerm(
                    degree=0,
                    coefficients=(CanonicalRational(num="1", den="1"),),
                    squared_norm=CanonicalRational(num="1", den="1"),
                ),
                OrthogonalPolynomialTerm(
                    degree=1,
                    coefficients=(
                        CanonicalRational(num="0", den="1"),
                        CanonicalRational(num="1", den="1"),
                    ),
                    squared_norm=CanonicalRational(num="0", den="1"),
                ),
            ),
            variable="x",
            is_quasi_definite=False,
            is_positive_definite=False,
        )
        result = compute_jacobi_matrix(JacobiMatrixRequest(family=family))
        assert [a.as_fraction() for a in result.alphas] == [Fraction(0)]
        assert [b.as_fraction() for b in result.betas] == [Fraction(0)]


class TestFamilyDefinitenessFlags:
    def test_flags_derived_from_norms(self) -> None:
        """A negative squared norm cannot carry positive-definite=true."""
        from jacobian.math.moments_orthogonal.values import (
            OrthogonalPolynomialFamily,
            OrthogonalPolynomialTerm,
        )

        term = OrthogonalPolynomialTerm(
            degree=0,
            coefficients=(CanonicalRational(num="1", den="1"),),
            squared_norm=CanonicalRational(num="-1", den="1"),
        )
        with pytest.raises(ValidationError):
            OrthogonalPolynomialFamily(
                polynomials=(term,),
                variable="x",
                is_quasi_definite=True,
                is_positive_definite=True,
            )


class TestQuadratureVariableBinding:
    def test_variable_must_match_prefix(self) -> None:
        from jacobian.math.moments_orthogonal.values import GaussianQuadratureRule

        prefix = _prefix(_moments_uniform(3))
        rule_payload = None
        try:
            GaussianQuadratureRule(
                order=1,
                nodes=(
                    QuadratureNode(
                        node=CanonicalRational(num="0", den="1"),
                        weight=CanonicalRational(num="2", den="1"),
                    ),
                ),
                variable="y",
                exactness_degree=1,
                prefix=prefix,
            )
            raise AssertionError("mismatched variable accepted")
        except ValidationError as error:
            rule_payload = str(error)
        assert "prefix" in rule_payload


class TestKernelFamilyBinding:
    def test_forged_kernel_rejected_against_family(self) -> None:
        """An asymmetric bivariate payload cannot revalidate as the
        kernel of a retained family."""
        from jacobian.math.moments_orthogonal._models import (
            OrthogonalPolynomialRequest,
        )
        from jacobian.math.moments_orthogonal.values import ChristoffelDarbouxKernel

        family = compute_orthogonal_polynomials(
            OrthogonalPolynomialRequest(
                prefix=_prefix(_moments_uniform(3)), max_degree=1
            )
        )
        payload = {
            "degree": 1,
            "coefficients": (({"num": "1", "den": "2"}, {"num": "7", "den": "5"}),),
            "variable": "x",
            "family": family.model_dump(),
        }
        # Pad to a square matrix so shape passes and the sum replay fails.
        payload["coefficients"] = (
            ({"num": "1", "den": "2"}, {"num": "7", "den": "5"}),
            ({"num": "0", "den": "1"}, {"num": "3", "den": "2"}),
        )
        with pytest.raises(ValidationError):
            ChristoffelDarbouxKernel.model_validate(payload)


class TestDerivedAlphaHeightAdmission:
    def test_nonrepresentable_derived_alpha_rejected_at_admission(self) -> None:
        """Two canonical coefficients can differ by a non-canonical rational;
        admission derives every emitted alpha exactly as execution does and
        rejects an over-tall difference at the boundary, so an accepted
        request cannot die mid-execution."""
        from jacobian.math.moments_orthogonal.values import (
            OrthogonalPolynomialFamily,
            OrthogonalPolynomialTerm,
        )

        q = Fraction(10) ** 10000

        def term(deg, coeffs, norm):
            return OrthogonalPolynomialTerm(
                degree=deg,
                coefficients=tuple(
                    CanonicalRational.from_fraction(Fraction(c)) for c in coeffs
                ),
                squared_norm=CanonicalRational.from_fraction(Fraction(norm)),
            )

        # p_1 = x + s with s = (q^2+1)/q; p_2 = x^2 + u*x + w. The derived
        # alpha_1 = residual coefficient of x*p_1 - p_2 is s - u, whose
        # numerator reaches ~10**40000 digits while both coefficients stay
        # canonical. The recurrence identity fixes h_1 = beta_1 * h_0 with
        # beta_1 the residual's component on p_0, so the authored norms are
        # exactly the consistent ones and the family itself validates.
        s = (q * q + 1) / q
        u = -1 / (q * q + 1)
        w = -(q + 1) / (q * q)
        derived_alpha_1 = s - u
        assert abs(derived_alpha_1.numerator) >= Fraction(10) ** 32768
        consistent_h1 = -w - derived_alpha_1 * s

        family = OrthogonalPolynomialFamily(
            polynomials=(
                term(0, (Fraction(1),), Fraction(1)),
                term(1, (s, Fraction(1)), consistent_h1),
                term(2, (w, u, Fraction(1)), Fraction(1)),
            ),
            variable="x",
            is_quasi_definite=True,
            is_positive_definite=False,
        )
        with pytest.raises(ValidationError):
            JacobiMatrixRequest(family=family)


class TestFiniteSupportQuadratureAdmission:
    def test_finite_support_rule_admitted(self) -> None:
        """A measure supported on exactly n points has a vanishing terminal
        norm h_n; Gaussian construction divides only through p_{n-1}, so the
        exact rule on the retained prefix must stay admitted."""
        moments = tuple(
            CanonicalRational(num=v, den="1") for v in ("2", "0", "2", "0", "2")
        )
        request = GaussianQuadratureRequest(prefix=_prefix(moments), order=2)
        result = compute_gaussian_quadrature(request)
        nodes = {(int(n.node.num), int(n.node.den)) for n in result.nodes}
        assert nodes == {(-1, 1), (1, 1)}
        for node in result.nodes:
            assert (int(node.weight.num), int(node.weight.den)) == (1, 1)
        assert result.exactness_degree == 3

    def test_result_round_trips_through_model_validate(self) -> None:
        """The deserialized finite-support rule replays against its prefix
        without requiring a quasi-definite terminal norm."""
        moments = tuple(
            CanonicalRational(num=v, den="1") for v in ("2", "0", "2", "0", "2")
        )
        result = compute_gaussian_quadrature(
            GaussianQuadratureRequest(prefix=_prefix(moments), order=2)
        )
        revalidated = GaussianQuadratureRule.model_validate(result.model_dump())
        assert revalidated == result


class TestDerivedQuadratureHeightAdmission:
    def test_over_tall_derived_node_rejected_with_typed_error(self) -> None:
        """mu_1/mu_0 far outside the conservative Gram-Schmidt height bound
        is rejected at the typed boundary before any exact projection or
        SymPy factorization runs."""
        moments = (
            CanonicalRational.from_fraction(Fraction(1, 10**16400)),
            CanonicalRational.from_fraction(Fraction(10) ** 16400),
            CanonicalRational(num="0", den="1"),
        )
        with pytest.raises(ValidationError):
            GaussianQuadratureRequest(prefix=_prefix(moments), order=1)

    def test_representable_large_node_admitted_and_round_trips(self) -> None:
        """A 2,000-digit rational node stays inside the conservative bound;
        exact numeric ordering must not stringify roots, which would trip
        CPython's integer-string conversion limit during admission."""
        moments = (
            CanonicalRational(num="1", den="1"),
            CanonicalRational.from_fraction(Fraction(10) ** 2000),
            CanonicalRational(num="0", den="1"),
        )
        request = GaussianQuadratureRequest(prefix=_prefix(moments), order=1)
        result = compute_gaussian_quadrature(request)
        assert result.nodes[0].node.as_fraction() == Fraction(10) ** 2000
        assert result.nodes[0].weight == CanonicalRational(num="1", den="1")
        assert GaussianQuadratureRule.model_validate(result.model_dump()) == result


class TestReplayedRulePositivity:
    def test_negative_weight_rule_rejected_on_revalidation(self) -> None:
        """A serialized rule whose retained prefix replays to a negative
        weight violates the operation's advertised positive-weight contract
        and must not revalidate as the operation's exact result."""
        prefix = _prefix(
            tuple(CanonicalRational(num=v, den="1") for v in ("-1", "0", "0"))
        )
        with pytest.raises(ValidationError):
            GaussianQuadratureRule(
                order=1,
                nodes=(
                    QuadratureNode(
                        node=CanonicalRational(num="0", den="1"),
                        weight=CanonicalRational(num="-1", den="1"),
                    ),
                ),
                variable="x",
                exactness_degree=1,
                prefix=prefix,
            )

    def test_quasi_definite_positive_weight_rule_revalidates(self) -> None:
        """A genuine positive-weight order-1 rule keeps validating against
        its quasi-definite source prefix."""
        prefix = _prefix(
            tuple(CanonicalRational(num=v, den="1") for v in ("1", "0", "-1"))
        )
        rule = GaussianQuadratureRule(
            order=1,
            nodes=(
                QuadratureNode(
                    node=CanonicalRational(num="0", den="1"),
                    weight=CanonicalRational(num="1", den="1"),
                ),
            ),
            variable="x",
            exactness_degree=1,
            prefix=prefix,
        )
        assert GaussianQuadratureRule.model_validate(rule.model_dump()) == rule


class TestShiftedHankelOrderCap:
    def test_order_32_is_not_schema_advertised(self) -> None:
        """A shifted matrix of order 32 would need mu_1..mu_66 but the
        canonical prefix holds at most 65 moments, so order 32 must not be
        advertised as supported."""
        full_prefix = _prefix(
            tuple(CanonicalRational(num="1", den="1") for _ in range(65))
        )
        with pytest.raises(ValidationError):
            ShiftedHankelRequest(prefix=full_prefix, order=32)
        assert ShiftedHankelRequest(prefix=full_prefix, order=31)


class TestCoefficientTupleSchemaBound:
    def test_term_coefficients_cannot_exceed_degree_bound(self) -> None:
        """The schema cap on each term's coefficient tuple matches the
        family's degree bound, so no admitted request can carry an
        unbounded coefficient array inside a low-degree term."""
        from jacobian.math.moments_orthogonal.values import (
            OrthogonalPolynomialTerm,
        )

        one = CanonicalRational(num="1", den="1")
        accepted = OrthogonalPolynomialTerm(
            degree=0,
            coefficients=tuple(one for _ in range(1)),
            squared_norm=one,
        )
        assert len(accepted.coefficients) == 1
        with pytest.raises(ValidationError):
            OrthogonalPolynomialTerm(
                degree=0,
                coefficients=tuple(one for _ in range(34)),
                squared_norm=one,
            )


class TestAdmissionReplaysExecution:
    """Over-height derived values fail parsing, never execution."""

    def test_over_height_recurrence_ratio_rejected_at_admission(self) -> None:
        """p_0=1, p_1=x with h_0=10^-30000 and h_1=10^30000 derives
        beta_1 = 10^60000 (past the 32,768-digit canonical limit);
        RecurrenceRequest parsing must reject it instead of letting
        math.run leak the execution-time ValueError."""
        family = OrthogonalPolynomialFamily.model_validate(
            {
                "polynomials": [
                    {
                        "degree": 0,
                        "coefficients": [{"num": "1", "den": "1"}],
                        "squared_norm": {"num": "1", "den": "1" + "0" * 30000},
                    },
                    {
                        "degree": 1,
                        "coefficients": [
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                        "squared_norm": {"num": "1" + "0" * 30000, "den": "1"},
                    },
                ],
                "variable": "x",
                "is_quasi_definite": True,
                "is_positive_definite": True,
            }
        )
        # h_1/h_0 = 10^20000 exceeds canonical; swap so a ratio overflows.
        with pytest.raises(ValidationError):
            RecurrenceRequest(family=family)

    def test_kernel_coefficient_overflow_rejected_at_admission(self) -> None:
        """p_1 = x + 10^17000 with unit norms: the degree-1 CD kernel's
        constant coefficient reaches ~10^34000 and must fail parsing."""
        big = "1" + "0" * 17000
        family = OrthogonalPolynomialFamily.model_validate(
            {
                "polynomials": [
                    {
                        "degree": 0,
                        "coefficients": [{"num": "1", "den": "1"}],
                        "squared_norm": {"num": "1", "den": "1"},
                    },
                    {
                        "degree": 1,
                        "coefficients": [
                            {"num": big, "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                        "squared_norm": {"num": "1", "den": "1"},
                    },
                ],
                "variable": "x",
                "is_quasi_definite": True,
                "is_positive_definite": True,
            }
        )
        with pytest.raises(ValidationError):
            ChristoffelDarbouxRequest(family=family, degree=1)


class TestNativeAdmission:
    def test_native_quadrature_matches_mcp_adapter(self) -> None:
        """Native quadrature consumes a canonical prefix without a request."""
        from jacobian.math.moments_orthogonal import gaussian_quadrature_rule
        from jacobian.math.moments_orthogonal.values import MomentFunctionalPrefix

        restored = MomentFunctionalPrefix.model_validate_json(
            _prefix(_moments_uniform(3)).model_dump_json()
        )
        assert gaussian_quadrature_rule(restored, 1) == compute_gaussian_quadrature(
            GaussianQuadratureRequest(prefix=restored, order=1)
        )

    def test_native_hankel_surfaces_match_mcp_adapters(self) -> None:
        """Native Hankel calls consume the canonical prefix without a request."""
        from jacobian.math.moments_orthogonal import (
            hankel_matrix,
            shifted_hankel_matrix,
        )
        from jacobian.math.moments_orthogonal.values import MomentFunctionalPrefix

        restored = MomentFunctionalPrefix.model_validate_json(
            _prefix(_moments_uniform(6)).model_dump_json()
        )
        assert hankel_matrix(restored, 2) == compute_hankel_matrix(
            HankelRequest(prefix=restored, order=2)
        )
        assert shifted_hankel_matrix(restored, 2) == compute_shifted_hankel(
            ShiftedHankelRequest(prefix=restored, order=2)
        )

    def test_native_jacobi_composes_serialized_family_and_matches_mcp_adapter(
        self,
    ) -> None:
        """A native family producer composes into the native Jacobi kernel.

        The JSON round trip models a canonical producer payload received by a
        second native caller. The direct and MCP paths retain one identical
        result value while each owns its proper input boundary.
        """
        from jacobian.math.moments_orthogonal import (
            jacobi_matrix,
            orthogonal_polynomials,
        )
        from jacobian.math.moments_orthogonal.values import (
            OrthogonalPolynomialFamily,
        )

        family = orthogonal_polynomials(_prefix(_moments_uniform(7)), 3)
        restored = OrthogonalPolynomialFamily.model_validate_json(
            family.model_dump_json()
        )
        expected = compute_jacobi_matrix(JacobiMatrixRequest(family=restored))

        assert jacobi_matrix(restored) == expected

    def test_native_short_prefix_rejected_before_kernel(self) -> None:
        """The native surface applies the shared moment-count admission so
        it cannot fabricate omitted moments as zeros."""
        from jacobian.math.moments_orthogonal import native

        prefix = _prefix(
            (
                CanonicalRational(num="1", den="1"),
                CanonicalRational(num="2", den="1"),
            )
        )
        with pytest.raises(ValueError, match="need at least 3 moments"):
            native.orthogonal_polynomials(prefix, max_degree=1)

    def test_native_over_tall_prefix_rejected_by_height_gate(self) -> None:
        """The native surface enforces the same conservative height bound
        as the wire request."""
        from jacobian.math.moments_orthogonal import native

        prefix = _prefix(
            (
                CanonicalRational.from_fraction(Fraction(1, 10**20000)),
                CanonicalRational.from_fraction(Fraction(10) ** 20000),
                CanonicalRational.from_fraction(Fraction(10) ** 20000),
            )
        )
        with pytest.raises(ValueError, match="conservative"):
            native.orthogonal_polynomials(prefix, max_degree=1)

    def test_native_recurrence_rejects_zero_norm_family(self) -> None:
        """The reviewer's counterexample: the canonical non-quasi-definite
        family p_0=1,h_0=0; p_1=x,h_1=0 is authorable as a value, and the
        native recurrence wrapper must reject it with the shared admission
        error instead of leaking ZeroDivisionError from the norm division."""
        from jacobian.math.moments_orthogonal import native
        from jacobian.math.moments_orthogonal.values import (
            OrthogonalPolynomialFamily,
            OrthogonalPolynomialTerm,
        )

        def term(deg: int, coeffs: tuple[int, ...], h: int):
            return OrthogonalPolynomialTerm(
                degree=deg,
                coefficients=tuple(
                    CanonicalRational.from_fraction(Fraction(c)) for c in coeffs
                ),
                squared_norm=CanonicalRational.from_fraction(Fraction(h)),
            )

        family = OrthogonalPolynomialFamily(
            polynomials=(term(0, (1,), 0), term(1, (0, 1), 0)),
            variable="x",
            is_quasi_definite=False,
            is_positive_definite=False,
        )
        with pytest.raises(ValueError, match="non-terminal"):
            native.recurrence_coefficients(family)

    def test_native_recurrence_admits_terminal_zero_norm(self) -> None:
        """The reviewer's follow-up: with h_0 = 2, h_1 = 0 the recurrence
        stays exactly defined (alpha_0 from p_0,p_1; beta_1 = h_1/h_0 = 0;
        nothing divides by the terminal norm), so the native surface must
        return the typed value instead of rejecting a valid computation."""
        from jacobian.math.moments_orthogonal import native
        from jacobian.math.moments_orthogonal.values import (
            OrthogonalPolynomialFamily,
            OrthogonalPolynomialTerm,
        )

        family = OrthogonalPolynomialFamily(
            polynomials=(
                OrthogonalPolynomialTerm(
                    degree=0,
                    coefficients=(CanonicalRational(num="1", den="1"),),
                    squared_norm=CanonicalRational(num="2", den="1"),
                ),
                OrthogonalPolynomialTerm(
                    degree=1,
                    coefficients=(
                        CanonicalRational(num="0", den="1"),
                        CanonicalRational(num="1", den="1"),
                    ),
                    squared_norm=CanonicalRational(num="0", den="1"),
                ),
            ),
            variable="x",
            is_quasi_definite=False,
            is_positive_definite=False,
        )
        result = native.recurrence_coefficients(family)
        assert [a.as_fraction() for a in result.alpha] == [Fraction(0)]
        assert [b.as_fraction() for b in result.beta] == [Fraction(0), Fraction(0)]
        wire = compute_recurrence(RecurrenceRequest(family=family))
        assert wire == result

    def test_native_recurrence_matches_wire_result(self) -> None:
        """For an admitted quasi-definite family the direct native call and
        the wire request produce identical typed recurrence values."""
        from jacobian.math.moments_orthogonal import native

        family = native.orthogonal_polynomials(_prefix(_moments_uniform(7)), 3)
        rec_native = native.recurrence_coefficients(family)
        rec_wire = compute_recurrence(RecurrenceRequest(family=family))
        assert rec_native == rec_wire
        assert int(rec_wire.beta[1].num) == 1
        assert int(rec_wire.beta[1].den) == 3
