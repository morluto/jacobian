"""Exact contract tests for the sequence-derived Koszul complex owner."""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import product
from math import comb

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.koszul import KoszulComplexValue, koszul_complex
from jacobian.math.koszul._models import KoszulComplexRequest
from jacobian.math.koszul.values import (
    MAX_KOSZUL_DEGREE,
    MAX_KOSZUL_TERMS,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.math.topology.chain_complexes import homology_groups

OPERATION_ID = "koszul.complex.construct.compute"


def _poly(
    variables: tuple[str, ...], terms: dict[tuple[int, ...], Fraction]
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(coefficient),
                    exponents=exponents,
                )
                for exponents, coefficient in sorted(terms.items(), reverse=True)
            )
        ),
    )


def _monomial(
    variables: tuple[str, ...], exponents: tuple[int, ...], value: int = 1
) -> RationalPolynomial:
    return _poly(variables, {exponents: Fraction(value)})


def _terms(polynomial: RationalPolynomial) -> dict[tuple[int, ...], Fraction]:
    return {
        term.exponents: Fraction(term.coefficient.num, term.coefficient.den)
        for term in polynomial.polynomial.terms
    }


_XY = ("x", "y")
_X = _monomial(_XY, (1, 0))
_Y = _monomial(_XY, (0, 1))


def _dense_cells(matrix) -> dict[tuple[int, int], dict[tuple[int, ...], Fraction]]:
    return {
        (entry.row, entry.column): _terms(entry.polynomial) for entry in matrix.entries
    }


def _replay_differential_squares(value: KoszulComplexValue) -> None:
    """Independently compose consecutive differentials and require zero."""

    cells = tuple(_dense_cells(matrix) for matrix in value.differentials)
    for degree in range(2, len(value.sequence) + 1):
        outer = cells[degree - 2]
        inner = cells[degree - 1]
        middle_columns = {column for (_, column) in inner} | {row for (row, _) in inner}
        for target_row in range(value.basis_sizes[degree - 2]):
            for source_column in range(value.basis_sizes[degree]):
                composed: dict[tuple[int, ...], Fraction] = {}
                for middle in middle_columns:
                    left = outer.get((target_row, middle))
                    right = inner.get((middle, source_column))
                    if left is None or right is None:
                        continue
                    for right_exponents, right_coefficient in right.items():
                        for left_exponents, left_coefficient in left.items():
                            exponents = tuple(
                                a + b
                                for a, b in zip(
                                    left_exponents, right_exponents, strict=True
                                )
                            )
                            total = composed.get(exponents, Fraction(0)) + (
                                left_coefficient * right_coefficient
                            )
                            if total:
                                composed[exponents] = total
                            else:
                                composed.pop(exponents, None)
                assert composed == {}, (degree, target_row, source_column)


def _catalog_tool():
    return next(tool for tool in BUILTIN_TOOLS if tool.operation_id == OPERATION_ID)


def _wire_payload(
    variables: tuple[str, ...], sequence: tuple[RationalPolynomial, ...]
) -> str:
    return KoszulComplexRequest(
        variables=variables, sequence=sequence
    ).model_dump_json()


def test_known_answer_koszul_of_x_y() -> None:
    """K(x, y) over QQ[x, y]: ranks (1, 2, 1) and the exact wedge matrices."""

    value = koszul_complex(_XY, (_X, _Y))
    assert value.basis_sizes == (1, 2, 1)
    assert value.degrees == (((),), ((0,), (1,)), ((0, 1),))
    d1, d2 = value.differentials
    assert (d1.row_count, d1.column_count) == (1, 2)
    assert _dense_cells(d1) == {
        (0, 0): {(1, 0): Fraction(1)},
        (0, 1): {(0, 1): Fraction(1)},
    }
    assert (d2.row_count, d2.column_count) == (2, 1)
    # d_2(e_0 ^ e_1) = f_0 e_(1) - f_1 e_(0) = x e_(1) - y e_(0)
    assert _dense_cells(d2) == {
        (0, 0): {(0, 1): Fraction(-1)},
        (1, 0): {(1, 0): Fraction(1)},
    }
    assert value.differential_relation == ("CONSECUTIVE_DIFFERENTIALS_COMPOSE_TO_ZERO")
    # The ambient ring is not QQ, so no scalar chain-complex conversion exists.
    assert value.chain_complex is None


def test_one_element_sequence_is_the_two_term_complex() -> None:
    """c = 1 gives 0 -> R --f1--> R with a single differential."""

    value = koszul_complex(_XY, (_X,))
    assert value.basis_sizes == (1, 1)
    assert value.degrees == (((),), ((0,),))
    assert _dense_cells(value.differentials[0]) == {(0, 0): {(1, 0): Fraction(1)}}


def test_empty_sequence_is_the_identity_complex() -> None:
    """c = 0 gives the identity complex R concentrated in degree 0."""

    value = koszul_complex((), ())
    assert value.basis_sizes == (1,)
    assert value.degrees == (((),),)
    assert value.differentials == ()
    assert value.chain_complex is not None
    assert value.chain_complex.basis_sizes == (1,)
    assert value.chain_complex.differential_matrices == ()


def test_differential_square_replays_to_zero_independently() -> None:
    """An oracle recomposes d^2 from the retained sparse matrices."""

    xyz = ("x", "y", "z")
    sequence = (
        _monomial(xyz, (1, 0, 0)),
        _poly(xyz, {(0, 1, 0): Fraction(1), (0, 0, 1): Fraction(1)}),
        _monomial(xyz, (1, 1, 0), -3),
    )
    value = koszul_complex(xyz, sequence)
    assert value.basis_sizes == (1, 3, 3, 1)
    _replay_differential_squares(value)


def test_regular_sequence_rank_identities() -> None:
    """(x, y) in QQ[x, y]: complete ranks and zero Euler characteristic."""

    value = koszul_complex(_XY, (_X, _Y))
    assert value.basis_sizes == tuple(comb(2, degree) for degree in range(3))
    euler = sum((-1) ** degree * size for degree, size in enumerate(value.basis_sizes))
    assert euler == 0
    _replay_differential_squares(value)


def test_scalar_conversion_composes_with_shared_homology() -> None:
    """Over QQ the conversion feeds the existing homology operation."""

    zero = _poly((), {})
    two = _poly((), {(): Fraction(2)})

    value = koszul_complex((), (zero, zero))
    assert value.chain_complex is not None
    homology = homology_groups(value.chain_complex)
    assert tuple(group.betti_number for group in homology.homology_groups) == (1, 2, 1)

    acyclic = koszul_complex((), (two,))
    assert acyclic.chain_complex is not None
    homology = homology_groups(acyclic.chain_complex)
    assert tuple(group.betti_number for group in homology.homology_groups) == (0, 0)


def test_permuted_sequence_keeps_ranks_and_twists_signs() -> None:
    """K(y, x) has the same ranks but is not byte-identical to K(x, y)."""

    original = koszul_complex(_XY, (_X, _Y))
    permuted = koszul_complex(_XY, (_Y, _X))
    assert permuted.basis_sizes == original.basis_sizes
    assert permuted.degrees == original.degrees
    assert _dense_cells(permuted.differentials[0]) == {
        (0, 0): {(0, 1): Fraction(1)},
        (0, 1): {(1, 0): Fraction(1)},
    }
    # Sign-twisted reindexing: d_2(e_01) = y e_(1) - x e_(0).
    assert _dense_cells(permuted.differentials[1]) == {
        (0, 0): {(1, 0): Fraction(-1)},
        (1, 0): {(0, 1): Fraction(1)},
    }
    _replay_differential_squares(permuted)


def test_zero_entry_is_exact_and_sparse() -> None:
    """A zero sequence entry contributes no sparse cells anywhere."""

    zero = _poly(_XY, {})
    value = koszul_complex(_XY, (_X, zero))
    assert _dense_cells(value.differentials[0]) == {(0, 0): {(1, 0): Fraction(1)}}
    assert _dense_cells(value.differentials[1]) == {(1, 0): {(1, 0): Fraction(1)}}
    _replay_differential_squares(value)


def test_native_and_catalog_paths_agree() -> None:
    tool = _catalog_tool()
    payload = _wire_payload(_XY, (_X, _Y))
    catalog_result = tool.run(tool.request_type.model_validate_json(payload))
    native_result = koszul_complex(_XY, (_X, _Y))
    assert catalog_result == native_result


def test_catalog_example_executes() -> None:
    tool = _catalog_tool()
    for example in tool.examples:
        result = tool.run(
            tool.request_type.model_validate_json(json.dumps(example.input))
        )
        assert result.basis_sizes == (1, 2, 1)


def test_serialization_round_trip() -> None:
    value = koszul_complex(_XY, (_X, _Y))
    assert KoszulComplexValue.model_validate_json(value.model_dump_json()) == value


def test_ring_mismatch_is_typed_rejected_on_both_paths() -> None:
    """A sequence element from another ring never reaches the kernel."""

    foreign = _monomial(("x",), (1,))
    with pytest.raises(ValidationError) as wire:
        KoszulComplexRequest(variables=_XY, sequence=(foreign,))
    assert wire.value.errors(include_url=False)[0]["type"] == "koszul.ring_mismatch"
    with pytest.raises(OperationDomainValidationError) as native:
        koszul_complex(_XY, (foreign,))
    assert native.value.errors()[0]["type"] == "koszul.ring_mismatch"


def test_forged_result_is_rejected_structurally() -> None:
    """A mutated basis cardinality fails result validation."""

    value = koszul_complex(_XY, (_X, _Y))
    payload = json.loads(value.model_dump_json())
    payload["basis_sizes"] = [1, 2, 2]
    with pytest.raises(ValidationError) as forged:
        KoszulComplexValue.model_validate_json(json.dumps(payload))
    assert forged.value.errors(include_url=False)[0]["type"] in {
        "koszul.basis_cardinality",
        "koszul.basis_binomial_cardinality",
        "koszul.differential_shape",
    }


def test_envelope_accepts_length_eight() -> None:
    """c = 8 materializes all 256 wedge basis elements."""

    ring = ("x",)
    sequence = tuple(_monomial(ring, (exponent,)) for exponent in range(1, 9))
    value = koszul_complex(ring, sequence)
    assert value.basis_sizes == tuple(comb(8, degree) for degree in range(9))
    assert sum(value.basis_sizes) == 256
    # Wedge rank 70 exceeds the shared chain envelope; conversion is absent.
    assert value.chain_complex is None


def test_scalar_length_eight_conversion_stays_absent() -> None:
    """C(8, 4) = 70 wedge rank exceeds the shared chain basis bound."""

    sequence = tuple(_poly((), {(): Fraction(index + 1)}) for index in range(8))
    value = koszul_complex((), sequence)
    assert value.basis_sizes == tuple(comb(8, degree) for degree in range(9))
    assert value.chain_complex is None


def test_envelope_rejects_length_nine() -> None:
    ring = ("x",)
    sequence = tuple(_monomial(ring, (exponent,)) for exponent in range(1, 10))
    with pytest.raises(OperationResourceAdmissionError) as native:
        koszul_complex(ring, sequence)
    assert native.value.errors()[0]["type"] == "koszul.sequence_length_budget"
    with pytest.raises(ValidationError):
        KoszulComplexRequest.model_validate_json(_wire_payload(ring, sequence))


def test_term_budget_rejects_above_the_envelope() -> None:
    """Exactly 64 terms are admitted; 65 are rejected before expansion."""

    ring = ("x", "y")
    support = sorted(set(product(range(MAX_KOSZUL_DEGREE + 1), repeat=2)))
    accepted = _poly(
        ring, {exponents: Fraction(1) for exponents in support[:MAX_KOSZUL_TERMS]}
    )
    assert len(accepted.polynomial.terms) == MAX_KOSZUL_TERMS
    koszul_complex(ring, (accepted,))

    too_many = _poly(
        ring,
        {exponents: Fraction(1) for exponents in support[: MAX_KOSZUL_TERMS + 1]},
    )
    with pytest.raises(OperationResourceAdmissionError) as admission:
        koszul_complex(ring, (too_many,))
    assert admission.value.errors()[0]["type"] == "koszul.term_budget"


def test_degree_budget_rejects_above_the_envelope() -> None:
    ring = ("x",)
    oversized = _monomial(ring, (MAX_KOSZUL_DEGREE + 1,))
    with pytest.raises(OperationResourceAdmissionError) as admission:
        koszul_complex(ring, (oversized,))
    assert admission.value.errors()[0]["type"] == "koszul.degree_budget"


def test_replay_work_budget_rejects_dense_length_eight() -> None:
    """Full 64-term elements at c = 8 exceed the charged replay envelope."""

    ring = ("w", "x", "y", "z")
    support = sorted(set(product(range(4), repeat=4)))[:MAX_KOSZUL_TERMS]
    dense = _poly(ring, {exponents: Fraction(1) for exponents in support})
    assert len(dense.polynomial.terms) == MAX_KOSZUL_TERMS
    with pytest.raises(OperationResourceAdmissionError) as admission:
        koszul_complex(ring, (dense,) * 8)
    assert admission.value.errors()[0]["type"] in {
        "koszul.replay_work_budget",
        "koszul.product_term_budget",
    }


def test_operation_is_published_in_the_catalog() -> None:
    ids = {tool.operation_id for tool in BUILTIN_TOOLS}
    assert OPERATION_ID in ids
