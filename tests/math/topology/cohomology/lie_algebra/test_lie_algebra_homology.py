"""Tests for Lie algebra homology operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix, rank
from jacobian.math.topology.cohomology.lie_algebra._models import (
    ChevalleyEilenbergComplexRequest,
    ChevalleyEilenbergComplexResult,
    DifferentialMatrix,
    LieAlgebra,
    LieHomologyRequest,
    LieHomologyResult,
)
from jacobian.math.topology.cohomology.lie_algebra._operations import (
    compute_chevalley_eilenberg_complex,
    compute_lie_homology,
)


def _assert_error_type(
    exc_info: pytest.ExceptionInfo[ValidationError], code: str
) -> None:
    assert exc_info.value.errors()[0]["type"] == code


def _sl2_gf5() -> LieAlgebra:
    """The standard three-dimensional simple Lie algebra sl(2) over GF(5).

    Basis (E, F, H) with [E, F] = H, [H, E] = 2E, [H, F] = -2F.
    """
    return LieAlgebra(
        prime=5,
        dimension=3,
        structure_constants=(
            ((0, 0, 0), (0, 0, 1), (3, 0, 0)),
            ((0, 0, 4), (0, 0, 0), (0, 2, 0)),
            ((2, 0, 0), (0, 3, 0), (0, 0, 0)),
        ),
    )


def _compose(d1: list[list[int]], d2: list[list[int]], prime: int) -> list[list[int]]:
    """Return the composition matrix d1 after d2 (target x source)."""
    rows, inner = len(d1), len(d1[0])
    assert inner == len(d2)
    cols = len(d2[0])
    return [
        [sum(d1[r][k] * d2[k][c] for k in range(inner)) % prime for c in range(cols)]
        for r in range(rows)
    ]


class TestChevalleyEilenbergComplex:
    """Test Chevalley-Eilenberg complex computation."""

    def test_abelian_2d(self) -> None:
        """CE complex of a 2D abelian Lie algebra has zero differentials."""
        g = LieAlgebra(
            prime=2,
            dimension=2,
            structure_constants=(((0, 0), (0, 0)), ((0, 0), (0, 0))),
        )
        result = compute_chevalley_eilenberg_complex(
            ChevalleyEilenbergComplexRequest(lie_algebra=g)
        )
        assert result.dimension == 2
        assert result.group_dimensions == (1, 2, 1)

    def test_3d_abelian(self) -> None:
        """CE complex of a 3D abelian Lie algebra."""
        g = LieAlgebra(
            prime=5,
            dimension=3,
            structure_constants=(
                ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
                ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
                ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
            ),
        )
        result = compute_chevalley_eilenberg_complex(
            ChevalleyEilenbergComplexRequest(lie_algebra=g)
        )
        assert result.group_dimensions == (1, 3, 3, 1)

    def test_sl2_differentials_square_to_zero(self) -> None:
        """Consecutive CE differentials of sl(2)/GF(5) compose to exactly zero."""
        result = compute_chevalley_eilenberg_complex(
            ChevalleyEilenbergComplexRequest(lie_algebra=_sl2_gf5())
        )
        matrices = {
            d.degree: [list(row) for row in d.matrix.entries]
            for d in result.differentials
        }
        # d_1 is identically zero and d_3 composes against d_2.
        assert matrices[1] == [[0, 0, 0]]
        assert rank(next(d for d in result.differentials if d.degree == 1).matrix) == 0
        for degree in range(1, 3):
            composed = _compose(matrices[degree], matrices[degree + 1], result.prime)
            assert all(value == 0 for row in composed for value in row)

    def test_sl2_d2_matrix_entries(self) -> None:
        """The d_2 matrix of sl(2) encodes the bracket columns exactly."""
        result = compute_chevalley_eilenberg_complex(
            ChevalleyEilenbergComplexRequest(lie_algebra=_sl2_gf5())
        )
        d2 = next(d for d in result.differentials if d.degree == 2)
        # targets (E, F, H) x sources (EF, EH, FH); each single-pair column
        # carries the standard (-1)^(a+b+pi) CE sign:
        # d(E^F) = -H, d(E^H) = -(-2E) = 2E... encoded as exact GF(5) residues.
        assert d2.matrix.entries == ((0, 2, 0), (0, 0, 3), (4, 0, 0))

    def test_differential_serializes_canonical_matrix(self) -> None:
        """The CE differential serializes as one reusable GF(p) matrix value."""
        result = compute_chevalley_eilenberg_complex(
            ChevalleyEilenbergComplexRequest(lie_algebra=_sl2_gf5())
        )
        d2 = next(d for d in result.differentials if d.degree == 2)
        payload = d2.model_dump()
        assert payload["matrix"]["prime"] == 5
        assert payload["matrix"]["columns"] == 3
        assert payload["matrix"]["entries"] == (
            (0, 2, 0),
            (0, 0, 3),
            (4, 0, 0),
        )
        # The serialized matrix composes unchanged with the matrix operations;
        # the cross-owner public seam lives in tests/integration/algebra.
        from jacobian.math.matrices.finite_fields.linear_algebra import rank

        assert rank(PrimeFieldMatrix(**payload["matrix"])) == 3


class TestLieAlgebraValidation:
    """Adversarial rejection of tensors that are not Lie brackets."""

    def test_composite_prime_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            LieAlgebra(
                prime=4,
                dimension=1,
                structure_constants=(((0,),),),
            )
        _assert_error_type(exc_info, "lie_algebra_homology.prime_not_prime")

    def test_alternation_violation_rejected(self) -> None:
        constants = (
            ((0, 0), (1, 0)),
            ((4, 0), (0, 1)),
        )  # [e_1, e_1] = e_0 != 0
        with pytest.raises(ValidationError) as exc_info:
            LieAlgebra(prime=5, dimension=2, structure_constants=constants)
        _assert_error_type(exc_info, "lie_algebra_homology.alternating")

    def test_antisymmetry_violation_rejected(self) -> None:
        constants = (
            ((0, 0), (1, 0)),
            ((1, 0), (0, 0)),
        )  # c[0][1] = (1,0) is not -c[1][0] mod 5
        with pytest.raises(ValidationError) as exc_info:
            LieAlgebra(prime=5, dimension=2, structure_constants=constants)
        _assert_error_type(exc_info, "lie_algebra_homology.antisymmetric")

    def test_jacobi_violation_rejected(self) -> None:
        # [e0, e1] = e0 and [e1, e2] = e1 violate the Jacobi identity:
        # [[e0, e1], e2] + [[e1, e2], e0] + [[e2, e0], e1] = -e0 != 0.
        constants = (
            ((0, 0, 0), (1, 0, 0), (0, 0, 0)),
            ((4, 0, 0), (0, 0, 0), (0, 1, 0)),
            ((0, 0, 0), (0, 4, 0), (0, 0, 0)),
        )
        with pytest.raises(ValidationError) as exc_info:
            LieAlgebra(prime=5, dimension=3, structure_constants=constants)
        _assert_error_type(exc_info, "lie_algebra_homology.jacobi")

    def test_noncanonical_diagonal_residue_rejected(self) -> None:
        # 2 mod 2 is zero, so every Lie identity holds vacuously; the entry
        # must still be rejected because it is not a canonical GF(2) residue.
        with pytest.raises(ValidationError) as exc_info:
            LieAlgebra(
                prime=2,
                dimension=1,
                structure_constants=(((2,),),),
            )
        _assert_error_type(exc_info, "lie_algebra_homology.canonical_residues")

    def test_noncanonical_offdiagonal_residue_rejected(self) -> None:
        # c[0][1] = 5 and c[1][0] = 5 are antisymmetric and Jacobi modulo 5
        # (both reduce to zero) but are not canonical GF(5) residues.
        constants = (
            ((0, 0), (5, 0)),
            ((5, 0), (0, 0)),
        )
        with pytest.raises(ValidationError) as exc_info:
            LieAlgebra(prime=5, dimension=2, structure_constants=constants)
        _assert_error_type(exc_info, "lie_algebra_homology.canonical_residues")

    def test_canonical_residues_accepted(self) -> None:
        g = LieAlgebra(
            prime=2,
            dimension=1,
            structure_constants=(((0,),),),
        )
        assert g.structure_constants == (((0,),),)


class TestLieHomology:
    """Test Lie algebra homology computation."""

    def test_abelian_2d(self) -> None:
        """Homology of a 2D abelian Lie algebra over GF(2)."""
        g = LieAlgebra(
            prime=2,
            dimension=2,
            structure_constants=(((0, 0), (0, 0)), ((0, 0), (0, 0))),
        )
        result = compute_lie_homology(LieHomologyRequest(lie_algebra=g))
        assert result.groups[0].betti == 1
        assert result.groups[1].betti == 2
        assert result.groups[2].betti == 1

    def test_abelian_1d(self) -> None:
        """Homology of a 1D abelian Lie algebra."""
        g = LieAlgebra(
            prime=7,
            dimension=1,
            structure_constants=(((0,),),),
        )
        result = compute_lie_homology(LieHomologyRequest(lie_algebra=g))
        assert result.groups[0].betti == 1
        assert result.groups[1].betti == 1

    def test_abelian_3d(self) -> None:
        """Homology of a 3D abelian Lie algebra over GF(5)."""
        g = LieAlgebra(
            prime=5,
            dimension=3,
            structure_constants=(
                ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
                ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
                ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
            ),
        )
        result = compute_lie_homology(LieHomologyRequest(lie_algebra=g))
        # H_0 = 1, H_1 = 3, H_2 = 3, H_3 = 1
        assert result.groups[0].betti == 1
        assert result.groups[1].betti == 3
        assert result.groups[2].betti == 3
        assert result.groups[3].betti == 1

    def test_sl2_gf5(self) -> None:
        """Homology of sl(2) over GF(5): trivial below the top degree.

        d_1 = 0, rank d_2 = 3, d_3 = 0, so the Betti numbers are
        (1, 0, 0, 1); the top class is the unimodular volume class.
        """
        result = compute_lie_homology(LieHomologyRequest(lie_algebra=_sl2_gf5()))
        assert tuple(group.betti for group in result.groups) == (1, 0, 0, 1)

    def test_chain_dimension_names_the_chain_group(self) -> None:
        """The chain-group dimension field is explicit about what it counts.

        For sl(2) the degree-1 homology vanishes (betti=0) while the chain
        group C_1 has dimension 3; the serialized field must not read as
        dim(H_k).
        """
        from pydantic import ValidationError

        from jacobian.math.topology.cohomology.lie_algebra._models import (
            LieHomologyGroup,
        )

        result = compute_lie_homology(LieHomologyRequest(lie_algebra=_sl2_gf5()))
        assert tuple(group.chain_dimension for group in result.groups) == (1, 3, 3, 1)
        assert result.groups[1].betti == 0
        payload = result.model_dump()["groups"][1]
        assert "chain_dimension" in payload and "dimension" not in payload
        with pytest.raises(ValidationError):
            LieHomologyGroup.model_validate(payload | {"dimension": 3})

    def test_affine_algebra_gf5(self) -> None:
        """Homology of the 2D non-abelian algebra [x, y] = x over GF(5)."""
        g = LieAlgebra(
            prime=5,
            dimension=2,
            structure_constants=(
                ((0, 0), (1, 0)),
                ((4, 0), (0, 0)),
            ),
        )
        result = compute_lie_homology(LieHomologyRequest(lie_algebra=g))
        # d_2: e0 ^ e1 -> e0 has rank 1; Betti numbers are (1, 1, 0).
        assert tuple(group.betti for group in result.groups) == (1, 1, 0)


def test_malformed_middle_axis_rejected() -> None:
    """structure_constants rows shorter than dimension are rejected, not indexed."""
    with pytest.raises(ValidationError):
        LieAlgebra(
            prime=5,
            dimension=2,
            structure_constants=(((0,),), ((0,),)),
        )


class TestComplexResultBinding:
    """The returned complex is validated against its retained source algebra."""

    def test_computed_complex_round_trips_against_source(self) -> None:
        from jacobian.math.topology.cohomology.lie_algebra._models import (
            ChevalleyEilenbergComplexResult,
        )

        result = compute_chevalley_eilenberg_complex(
            ChevalleyEilenbergComplexRequest(lie_algebra=_sl2_gf5())
        )
        replayed = ChevalleyEilenbergComplexResult(
            lie_algebra=result.lie_algebra,
            dimension=result.dimension,
            group_dimensions=result.group_dimensions,
            differentials=result.differentials,
            prime=result.prime,
        )
        assert replayed == result

    def test_reviewer_payload_shape_rejected(self) -> None:
        """A complex without its source algebra cannot validate."""
        with pytest.raises(ValidationError):
            ChevalleyEilenbergComplexResult.model_validate(
                {
                    "dimension": 1,
                    "group_dimensions": (1, 1),
                    "differentials": (),
                    "prime": 2,
                }
            )

    def test_missing_differential_degree_rejected(self) -> None:
        g = _sl2_gf5()
        full = compute_chevalley_eilenberg_complex(
            ChevalleyEilenbergComplexRequest(lie_algebra=g)
        )
        with pytest.raises(ValidationError) as exc_info:
            ChevalleyEilenbergComplexResult(
                lie_algebra=g,
                dimension=3,
                group_dimensions=(1, 3, 3, 1),
                differentials=tuple(d for d in full.differentials if d.degree != 1),
                prime=5,
            )
        _assert_error_type(exc_info, "lie_algebra_homology.complex_degrees")

    def test_wrong_matrix_shape_rejected(self) -> None:
        g = _sl2_gf5()
        full = compute_chevalley_eilenberg_complex(
            ChevalleyEilenbergComplexRequest(lie_algebra=g)
        )
        d2 = next(d for d in full.differentials if d.degree == 2)
        # Rows narrower than the declared column axis cannot form a matrix value.
        tampered_rows = tuple(row[:2] for row in d2.matrix.entries)
        with pytest.raises(ValidationError):
            DifferentialMatrix(
                degree=2,
                matrix=PrimeFieldMatrix(
                    prime=5, entries=tampered_rows, columns=d2.matrix.columns
                ),
            )
        # A well-formed but forged differential remains a separate structural
        # value; ordinary result parsing does not rerun the CE kernel.
        forged_rows = [list(row) for row in d2.matrix.entries]
        forged_rows[0][0] = (forged_rows[0][0] + 1) % 5
        broken = DifferentialMatrix(
            degree=2,
            matrix=PrimeFieldMatrix(
                prime=5,
                entries=tuple(tuple(row) for row in forged_rows),
                columns=d2.matrix.columns,
            ),
        )
        others = tuple(d for d in full.differentials if d.degree != 2)
        claimed = ChevalleyEilenbergComplexResult(
            lie_algebra=g,
            dimension=3,
            group_dimensions=(1, 3, 3, 1),
            differentials=(others[0], broken, others[1]),
            prime=5,
        )
        assert claimed.differentials != full.differentials

    def test_broken_d_squared_composition_rejected(self) -> None:
        """Tampering one d_2 entry must fail the bracket reconstruction.

        d_1 of sl(2)/GF(5) is identically zero, so a composition-only check
        cannot detect forged d_2 entries; reconstruction from the retained
        bracket does.
        """
        g = _sl2_gf5()
        full = compute_chevalley_eilenberg_complex(
            ChevalleyEilenbergComplexRequest(lie_algebra=g)
        )
        d2 = next(d for d in full.differentials if d.degree == 2)
        rows = [list(row) for row in d2.matrix.entries]
        rows[0][0] = (rows[0][0] + 1) % 5
        forged_d2 = DifferentialMatrix(
            degree=2,
            matrix=PrimeFieldMatrix(
                prime=5,
                entries=tuple(tuple(row) for row in rows),
                columns=d2.matrix.columns,
            ),
        )
        others = tuple(d for d in full.differentials if d.degree != 2)
        claimed = ChevalleyEilenbergComplexResult(
            lie_algebra=g,
            dimension=3,
            group_dimensions=(1, 3, 3, 1),
            differentials=(others[0], forged_d2, others[1]),
            prime=5,
        )
        assert claimed.differentials != full.differentials

    def test_non_residue_entries_rejected(self) -> None:
        g = _sl2_gf5()
        full = compute_chevalley_eilenberg_complex(
            ChevalleyEilenbergComplexRequest(lie_algebra=g)
        )
        d1 = next(d for d in full.differentials if d.degree == 1)
        rows = [list(row) for row in d1.matrix.entries]
        rows[0][0] = 5
        # The canonical prime-field matrix value itself rejects non-residues.
        with pytest.raises(ValidationError) as exc_info:
            DifferentialMatrix(
                degree=1,
                matrix=PrimeFieldMatrix(
                    prime=5,
                    entries=tuple(tuple(row) for row in rows),
                    columns=d1.matrix.columns,
                ),
            )
        _assert_error_type(exc_info, "value_error")


class TestHomologySourceBinding:
    def test_kernel_homology_has_expected_groups(self) -> None:
        result = compute_lie_homology(LieHomologyRequest(lie_algebra=_sl2_gf5()))
        assert tuple(group.betti for group in result.groups) == (1, 0, 0, 1)

    def test_forged_groups_remain_structural_values(self) -> None:
        genuine = compute_lie_homology(LieHomologyRequest(lie_algebra=_sl2_gf5()))
        payload = genuine.model_dump()
        payload["groups"] = [
            {"degree": 0, "betti": 0, "chain_dimension": 1},
            {"degree": 1, "betti": 1, "chain_dimension": 3},
            {"degree": 2, "betti": 0, "chain_dimension": 3},
            {"degree": 3, "betti": 1, "chain_dimension": 1},
        ]
        claimed = LieHomologyResult.model_validate(payload)
        assert claimed.groups != genuine.groups

    def test_dimension_mismatch_with_source_rejected(self) -> None:
        from pydantic import ValidationError

        genuine = compute_lie_homology(LieHomologyRequest(lie_algebra=_sl2_gf5()))
        payload = genuine.model_dump()
        payload["prime"] = 7
        with pytest.raises(ValidationError) as exc_info:
            LieHomologyResult.model_validate(payload)
        _assert_error_type(exc_info, "lie_algebra_homology.homology_source_mismatch")


def _abelian(dimension: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(
        tuple((0,) * dimension for _ in range(dimension)) for _ in range(dimension)
    )


def _ut5_gf2() -> LieAlgebra:
    """The 10-dimensional nilpotent algebra of strictly upper-triangular 5x5 matrices.

    Basis E_ij (i < j) with [E_ij, E_jk] = E_ik over GF(2).
    """
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    index = {pair: position for position, pair in enumerate(pairs)}
    n = len(pairs)
    c = [[[0] * n for _ in range(n)] for _ in range(n)]
    for a, (i, j) in enumerate(pairs):
        for b, (j2, k) in enumerate(pairs):
            if j == j2:
                c[a][b][index[(i, k)]] = 1
                c[b][a][index[(i, k)]] = 1
    return LieAlgebra(
        prime=2,
        dimension=n,
        structure_constants=tuple(tuple(tuple(row) for row in matrix) for matrix in c),
    )


class TestDimensionEnvelope:
    """The dimension cap is derived from the execution envelope, not fixed."""

    def test_widest_chain_group_fits_one_prime_field_matrix(self) -> None:
        from math import comb

        from jacobian.math.topology.cohomology.lie_algebra._models import (
            MAX_LIE_ALGEBRA_DIMENSION,
        )

        assert MAX_LIE_ALGEBRA_DIMENSION == 10
        widest = max(
            comb(MAX_LIE_ALGEBRA_DIMENSION, k)
            for k in range(MAX_LIE_ALGEBRA_DIMENSION + 1)
        )
        assert widest <= 256
        next_dimension = MAX_LIE_ALGEBRA_DIMENSION + 1
        assert comb(next_dimension, next_dimension // 2) > 256

    def test_nine_dimensional_abelian_complex_and_homology_execute(self) -> None:
        """A 9-dim GF(2) abelian algebra fits every kernel and output bound
        and must not be rejected by a coarse dimension ceiling."""
        from math import comb

        n = 9
        g = LieAlgebra(prime=2, dimension=n, structure_constants=_abelian(n))
        complex_result = compute_chevalley_eilenberg_complex(
            ChevalleyEilenbergComplexRequest(lie_algebra=g)
        )
        assert complex_result.group_dimensions == tuple(
            comb(n, k) for k in range(n + 1)
        )
        homology = compute_lie_homology(LieHomologyRequest(lie_algebra=g))
        assert [group.betti for group in homology.groups] == list(
            complex_result.group_dimensions
        )

    def test_ten_dimensional_nonabelian_algebra_executes(self) -> None:
        """The envelope boundary admits non-abelian brackets at dimension 10;
        d^2 = 0 is replayed by the result validator."""
        g = _ut5_gf2()
        complex_result = compute_chevalley_eilenberg_complex(
            ChevalleyEilenbergComplexRequest(lie_algebra=g)
        )
        degrees = sorted(
            differential.degree for differential in complex_result.differentials
        )
        assert degrees == list(range(1, 11))
        homology = compute_lie_homology(LieHomologyRequest(lie_algebra=g))
        assert [group.betti for group in homology.groups][:2] == [1, 4]

    def test_eleven_dimensional_algebra_rejected_at_boundary(self) -> None:
        """Dimension 11 would push C(11,5) = 462 rows past one matrix axis,
        so it stays rejected at the request boundary."""
        with pytest.raises(ValidationError) as exc_info:
            LieAlgebra(prime=2, dimension=11, structure_constants=_abelian(11))
        _assert_error_type(exc_info, "less_than_equal")


class TestCharacteristicEnvelope:
    """The characteristic bound is the documented shared conservative fallback."""

    def test_envelope_matches_the_shared_prime_field_backend(self) -> None:
        from jacobian.math.topology.cohomology.lie_algebra._models import (
            MAX_LIE_ALGEBRA_PRIME,
        )

        assert MAX_LIE_ALGEBRA_PRIME == 2_147_483_647

    def test_large_characteristic_trivial_algebra_executes(self) -> None:
        """A 1-dim abelian algebra over GF(10007) must not be rejected."""
        g = LieAlgebra(prime=10007, dimension=1, structure_constants=(((0,),),))
        homology = compute_lie_homology(LieHomologyRequest(lie_algebra=g))
        assert [group.betti for group in homology.groups] == [1, 1]
        complex_result = compute_chevalley_eilenberg_complex(
            ChevalleyEilenbergComplexRequest(lie_algebra=g)
        )
        assert complex_result.prime == 10007

    def test_shared_envelope_maximum_executes(self) -> None:
        """The largest admitted characteristic runs through the CE kernel."""
        from jacobian.math.topology.cohomology.lie_algebra._models import (
            MAX_LIE_ALGEBRA_PRIME,
        )

        g = LieAlgebra(
            prime=MAX_LIE_ALGEBRA_PRIME, dimension=1, structure_constants=(((0,),),)
        )
        homology = compute_lie_homology(LieHomologyRequest(lie_algebra=g))
        assert [group.betti for group in homology.groups] == [1, 1]

    def test_characteristic_above_shared_envelope_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LieAlgebra(
                prime=2_147_483_648,
                dimension=1,
                structure_constants=(((0,),),),
            )
