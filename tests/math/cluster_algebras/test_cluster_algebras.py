"""Tests for cluster algebra operations."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from jacobian.canonical import parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.cluster_algebras._models import (
    ExchangeMatrix,
    GVectorRequest,
    GVectorResult,
    SeedMutationRequest,
    SeedMutationResult,
)
from jacobian.math.cluster_algebras.operations import (
    g_vectors,
    mutate_seed,
)
from jacobian.math.cluster_algebras.operations import (
    mutate_seed as native_mutate_seed,
)


def compute_seed_mutation(request: SeedMutationRequest) -> SeedMutationResult:
    return mutate_seed(request.exchange_matrix, request.mutation_index)


def compute_g_vectors(request: GVectorRequest) -> GVectorResult:
    return g_vectors(request.exchange_matrix)


def em(
    n: int,
    entries: Sequence[Sequence[int | str]],
    symmetrizer: Sequence[int | str],
) -> ExchangeMatrix:
    """Build an ExchangeMatrix from raw integers using canonical strings."""
    return ExchangeMatrix(
        n=n,
        entries=tuple(tuple(str(v) for v in row) for row in entries),
        symmetrizer=tuple(str(d) for d in symmetrizer),
    )


def ientries(matrix: ExchangeMatrix) -> tuple[tuple[int, ...], ...]:
    """Parsed integer view of an ExchangeMatrix's entries."""
    return tuple(
        tuple(parse_canonical_integer(v) for v in row) for row in matrix.entries
    )


class TestSeedMutation:
    """Test cluster seed mutation."""

    def test_a2_mutation_at_0(self) -> None:
        """Mutate the A2 seed at index 0."""
        b = em(2, ((str(0), str(1)), (str(-1), str(0))), (str(1), str(1)))
        result = compute_seed_mutation(
            SeedMutationRequest(exchange_matrix=b, mutation_index=0)
        )
        assert result.exchange_matrix.entries[0][1] == "-1"
        assert result.exchange_matrix.entries[1][0] == "1"

    def test_a2_mutation_at_1(self) -> None:
        """Mutate the A2 seed at index 1."""
        b = em(2, ((str(0), str(1)), (str(-1), str(0))), (str(1), str(1)))
        result = compute_seed_mutation(
            SeedMutationRequest(exchange_matrix=b, mutation_index=1)
        )
        assert result.exchange_matrix.entries[0][1] == "-1"
        assert result.exchange_matrix.entries[1][0] == "1"

    def test_mutation_involutive(self) -> None:
        """Double mutation at the same index returns to the original."""
        b = em(2, ((str(0), str(1)), (str(-1), str(0))), (str(1), str(1)))
        result1 = compute_seed_mutation(
            SeedMutationRequest(exchange_matrix=b, mutation_index=0)
        )
        result2 = compute_seed_mutation(
            SeedMutationRequest(
                exchange_matrix=result1.exchange_matrix, mutation_index=0
            )
        )
        assert ientries(result2.exchange_matrix) == ientries(b)

    def test_3x3_mutation(self) -> None:
        """Mutate a 3x3 skew-symmetric matrix."""
        b = em(
            3,
            (
                (str(0), str(1), str(0)),
                (str(-1), str(0), str(0)),
                (str(0), str(0), str(0)),
            ),
            (str(1), str(1), str(1)),
        )
        result = compute_seed_mutation(
            SeedMutationRequest(exchange_matrix=b, mutation_index=1)
        )
        # The mutated matrix should still be skew-symmetric
        new = ientries(result.exchange_matrix)
        for i in range(3):
            assert new[i][i] == 0

    def test_invalid_mutation_index(self) -> None:
        """Mutation index out of range should fail."""
        b = em(2, ((str(0), str(1)), (str(-1), str(0))), (str(1), str(1)))
        request = SeedMutationRequest(exchange_matrix=b, mutation_index=2)
        with pytest.raises(OperationDomainValidationError) as caught:
            compute_seed_mutation(request)
        assert caught.value.errors()[0]["loc"] == ("mutation_index",)
        assert caught.value.errors()[0]["type"] == "cluster_algebra.mutation_index"

    def test_skew_symmetrizable(self) -> None:
        """A non-skew-symmetrizable matrix should fail."""
        with pytest.raises(ValueError, match="skew-symmetrizability"):
            em(2, ((str(0), str(1)), (str(1), str(0))), (str(1), str(1)))

    def test_zero_symmetrizer_rejected(self) -> None:
        """A symmetrizer with a zero entry is not an exchange matrix."""
        with pytest.raises(ValueError, match="strictly positive"):
            em(2, ((str(0), str(1)), (str(-1), str(0))), (str(0), str(2)))

    def test_negative_symmetrizer_rejected(self) -> None:
        """A symmetrizer with a negative entry is rejected."""
        with pytest.raises(ValueError, match="strictly positive"):
            em(2, ((str(0), str(1)), (str(-1), str(0))), (str(1), str(-1)))


def test_native_surface_accepts_exchange_matrix_value() -> None:
    matrix = em(2, ((0, 1), (-1, 0)), (1, 1))

    assert native_mutate_seed(matrix, 0).exchange_matrix.entries == (
        ("0", "-1"),
        ("1", "0"),
    )
    assert g_vectors(matrix).g_matrix == ((1, 0), (0, 1))


class TestCoefficientBounds:
    """Admitted seeds derive their mutation budget from their own entries."""

    def test_mutation_result_is_admissible_as_next_input(self) -> None:
        # A valid mutation output composes back into the same operation.
        edge = 10**63
        b = em(
            3,
            (
                (str(0), str(edge), str(0)),
                (str(-edge), str(0), str(edge)),
                (str(0), str(-edge), str(0)),
            ),
            (str(1), str(1), str(1)),
        )
        result = compute_seed_mutation(
            SeedMutationRequest(exchange_matrix=b, mutation_index=1)
        )
        assert (
            abs(parse_canonical_integer(result.exchange_matrix.entries[0][2]))
            == edge * edge
        )
        composed = SeedMutationRequest(
            exchange_matrix=result.exchange_matrix, mutation_index=1
        )
        assert ientries(compute_seed_mutation(composed).exchange_matrix) == ientries(b)

    def test_involutive_mutation_near_ceiling_admitted(self) -> None:
        # b01=10**64, b12=6*10**64, b02=10**128 mutates to b'02=7*10**128;
        # resubmitting that result must be admitted because the signed update
        # returns b''02=7*10**128 - 6*10**128 = 10**128, not 13*10**128.
        e = 10**64
        b = em(
            3,
            (
                (str(0), str(e), str(10**128)),
                (str(-e), str(0), str(6 * e)),
                (str(-(10**128)), str(-6 * e), str(0)),
            ),
            (str(1), str(1), str(1)),
        )
        once = compute_seed_mutation(
            SeedMutationRequest(exchange_matrix=b, mutation_index=1)
        )
        assert (
            parse_canonical_integer(once.exchange_matrix.entries[0][2]) == 7 * 10**128
        )
        involuted = compute_seed_mutation(
            SeedMutationRequest(exchange_matrix=once.exchange_matrix, mutation_index=1)
        )
        assert ientries(involuted.exchange_matrix) == ientries(b)

    def test_request_rejects_mutation_exceeding_representation_ceiling(self) -> None:
        edge = 10**100
        b = em(
            3,
            (
                (str(0), str(edge), str(0)),
                (str(-edge), str(0), str(edge)),
                (str(0), str(-edge), str(0)),
            ),
            (str(1), str(1), str(1)),
        )
        request = SeedMutationRequest(exchange_matrix=b, mutation_index=1)
        with pytest.raises(OperationDomainValidationError) as caught:
            compute_seed_mutation(request)
        assert caught.value.errors()[0]["loc"] == ("exchange_matrix",)
        assert caught.value.errors()[0]["type"] == "cluster_algebra.mutation_bounded"

    def test_request_accepts_large_entries_under_negation_only_mutation(self) -> None:
        edge = 10**129 - 1
        b = em(2, ((str(0), str(edge)), (str(-edge), str(0))), (str(1), str(1)))
        result = compute_seed_mutation(
            SeedMutationRequest(exchange_matrix=b, mutation_index=0)
        )
        assert ientries(result.exchange_matrix) == ((0, -edge), (edge, 0))

    def test_mutation_near_bound_stays_within_result_ceiling(self) -> None:
        # Mutating a path matrix at the middle index forms an entry of
        # magnitude ~N^2; it must remain an admissible exchange matrix.
        edge = 10**63
        b = em(
            3,
            (
                (str(0), str(edge), str(0)),
                (str(-edge), str(0), str(edge)),
                (str(0), str(-edge), str(0)),
            ),
            (str(1), str(1), str(1)),
        )
        result = compute_seed_mutation(
            SeedMutationRequest(exchange_matrix=b, mutation_index=1)
        )
        mutated = parse_canonical_integer(result.exchange_matrix.entries[0][2])
        assert abs(mutated) == edge * edge
        assert len(str(abs(mutated))) <= 129

    def test_rejects_symmetrizer_beyond_bound(self) -> None:
        # A zero row-pair keeps the matrix skew-symmetrizable so the
        # symmetrizer bound is what rejects the seed.
        with pytest.raises(ValidationError) as exc_info:
            em(2, ((str(0), str(0)), (str(0), str(0))), (str(1), str(10**64)))
        assert (
            exc_info.value.errors()[0]["type"] == "cluster_algebra.symmetrizer_bounded"
        )

    def test_result_ceiling_rejects_oversized_entries(self) -> None:
        # Even skew-symmetrizable matrices cannot carry unbounded integers.
        # Enforcement precedes integer conversion: the schema caps canonical
        # string length, and the digit-bound validator rejects values the
        # string cap alone admits (a 130-character positive value).
        beyond_schema = "9" * 500
        with pytest.raises(ValidationError):
            em(2, ((str(0), beyond_schema), (str(-1), str(0))), (str(1), str(1)))
        at_schema_cap = "1" + "0" * 129
        with pytest.raises(ValidationError) as exc_info:
            em(2, ((str(0), at_schema_cap), (str(-1), str(0))), (str(1), str(1)))
        assert (
            exc_info.value.errors()[0]["type"]
            == "cluster_algebra.exchange_entries_bounded"
        )

    def test_seventeen_by_seventeen_zero_matrix_is_admitted(self) -> None:
        # Work and output derive from cells times coefficient heights, not a
        # rank cap: 289 trivial updates returning a small exact matrix.
        n = 17
        b = em(n, tuple((0,) * n for _ in range(n)), (1,) * n)
        result = compute_seed_mutation(
            SeedMutationRequest(exchange_matrix=b, mutation_index=0)
        )
        assert ientries(result.exchange_matrix) == tuple((0,) * n for _ in range(n))

    def test_zero_matrix_beyond_cell_budget_rejected(self) -> None:
        n = 65  # 65**2 > MAX_EXCHANGE_CELLS even for an all-zero matrix
        with pytest.raises(ValidationError):
            em(n, tuple((0,) * n for _ in range(n)), (1,) * n)

    def test_gvector_request_accepts_representable_coefficients(self) -> None:
        # The g-vector result is the identity, so any representable seed works.
        b = em(
            2,
            ((str(0), str(10**64 + 1)), (str(-(10**64 + 1)), str(0))),
            (str(1), str(1)),
        )
        result = compute_g_vectors(GVectorRequest(exchange_matrix=b))
        assert result.g_matrix == ((1, 0), (0, 1))


class TestGVectorBinding:
    """Exact g-vector results must replay against their retained source."""

    def test_result_binds_to_source_and_convention(self) -> None:
        b = em(
            3,
            (
                (str(0), str(1), str(0)),
                (str(-1), str(0), str(1)),
                (str(0), str(-1), str(0)),
            ),
            (str(1), str(1), str(1)),
        )
        result = compute_g_vectors(GVectorRequest(exchange_matrix=b))
        assert result.exchange_matrix == b
        assert result.g_matrix == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        assert result.convention == "FOMIN_ZELEVINSKY"
        GVectorResult.model_validate(result.model_dump())

    def test_result_rejects_dimension_mismatch(self) -> None:
        b = em(2, ((str(0), str(1)), (str(-1), str(0))), (str(1), str(1)))
        with pytest.raises(ValidationError) as exc_info:
            GVectorResult(
                exchange_matrix=b,
                g_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                convention="FOMIN_ZELEVINSKY",
            )
        assert exc_info.value.errors()[0]["type"] == "cluster_algebra.g_matrix_shape"

    def test_result_rejects_arbitrary_convention(self) -> None:
        b = em(2, ((str(0), str(1)), (str(-1), str(0))), (str(1), str(1)))
        with pytest.raises(ValidationError):
            GVectorResult.model_validate(
                {
                    "exchange_matrix": b,
                    "g_matrix": ((1, 0), (0, 1)),
                    "convention": "anything",
                }
            )

    def test_forged_empty_payload_rejected(self) -> None:
        b = em(
            3,
            (
                (str(0), str(1), str(0)),
                (str(-1), str(0), str(1)),
                (str(0), str(-1), str(0)),
            ),
            (str(1), str(1), str(1)),
        )
        with pytest.raises(ValidationError):
            GVectorResult.model_validate(
                {"exchange_matrix": b, "g_matrix": (), "convention": "anything"}
            )


class TestGVector:
    """Test g-vector computation."""

    def test_initial_g_vectors(self) -> None:
        """Initial g-vectors should be the identity."""
        b = em(2, ((str(0), str(1)), (str(-1), str(0))), (str(1), str(1)))
        result = compute_g_vectors(GVectorRequest(exchange_matrix=b))
        assert result.g_matrix == ((1, 0), (0, 1))

    def test_3x3_g_vectors(self) -> None:
        """g-vectors for a 3x3 seed should be identity."""
        b = em(
            3,
            (
                (str(0), str(1), str(0)),
                (str(-1), str(0), str(0)),
                (str(0), str(0), str(0)),
            ),
            (str(1), str(1), str(1)),
        )
        result = compute_g_vectors(GVectorRequest(exchange_matrix=b))
        assert result.g_matrix == ((1, 0, 0), (0, 1, 0), (0, 0, 1))


class TestCanonicalTransport:
    def test_mutated_coefficients_survive_canonical_output(self) -> None:
        """N=10**8 path mutation squares an entry to 10**16 > 2**53-1.

        Coefficients are canonical integer strings, so the serialized result
        projects through canonical output validation instead of failing after
        request validation.
        """
        n = 10**8
        b = em(
            3,
            ((0, n, 0), (-n, 0, n), (0, -n, 0)),
            (1, 1, 1),
        )
        result = compute_seed_mutation(
            SeedMutationRequest(exchange_matrix=b, mutation_index=1)
        )
        payload = result.model_dump()
        assert payload["exchange_matrix"]["entries"][0][2] == str(n * n)
        assert SeedMutationResult.model_validate(payload) == result

    def test_entries_are_canonical_integer_strings(self) -> None:
        b = ExchangeMatrix(
            n=2,
            entries=(("0", "1"), ("-1", "0")),
            symmetrizer=("1", "1"),
        )
        assert ientries(b) == ((0, 1), (-1, 0))
        assert b.model_dump()["entries"] == (("0", "1"), ("-1", "0"))
