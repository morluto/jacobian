import json

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology.chain_complexes.operations import (
    differential_squares_to_zero,
    homology_groups,
)
from jacobian.math.topology.chain_complexes.values import CoefficientRing
from jacobian.math.topology.simplicial_sets import chains as chains_module
from jacobian.math.topology.simplicial_sets import maps as maps_module
from jacobian.math.topology.simplicial_sets.chains import (
    UnnormalizedChainsRequest,
    unnormalized_chains,
)
from jacobian.math.topology.simplicial_sets.maps import (
    normalized_chains,
    normalized_homology,
)
from jacobian.math.topology.simplicial_sets.standard import standard_simplex


def _compose(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(second[index] for index in first)


def _dense_product(left, right):
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(len(right)))
            for column in range(len(right[0]) if right else 0)
        )
        for row in range(len(left))
    )


def test_unnormalized_delta_one_uses_every_simplex_and_is_chain_complex():
    source = standard_simplex(1, 3)
    for degree in range(2, source.max_degree + 1):
        for outer in range(degree + 1):
            for inner in range(outer):
                left = _compose(
                    source.face_maps[degree - 1][outer],
                    source.face_maps[degree - 2][inner],
                )
                right = _compose(
                    source.face_maps[degree - 1][inner],
                    source.face_maps[degree - 2][outer - 1],
                )
                assert left == right

    result = unnormalized_chains(UnnormalizedChainsRequest(simplicial_set=source))
    value = result.chain_complex
    assert result.simplex_bases == source.sets
    assert value.basis_sizes == (2, 3, 4, 5)
    assert value.differential_matrices == (
        ((0, -1, 0), (0, 1, 0)),
        ((1, 1, 0, 0), (0, 0, 0, 0), (0, 0, 1, 1)),
        (
            (0, -1, 0, 0, 0),
            (0, 1, 0, 0, 0),
            (0, 0, 0, -1, 0),
            (0, 0, 0, 1, 0),
        ),
    )
    integer_matrices = value.differential_matrices
    assert _dense_product(integer_matrices[0], integer_matrices[1]) == (
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )
    assert _dense_product(integer_matrices[1], integer_matrices[2]) == (
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
    )
    assert differential_squares_to_zero(value).is_valid

    normalized = normalized_chains(source)
    assert normalized.chain_complex.basis_sizes == (2, 1, 0, 0)
    assert value.basis_sizes != normalized.chain_complex.basis_sizes
    assert normalized.nondegenerate_bases == (
        ("(0)", "(1)"),
        ("(0,1)",),
        (),
        (),
    )
    assert differential_squares_to_zero(normalized.chain_complex).is_valid
    reusable_homology = homology_groups(normalized.chain_complex)
    assert [group.free_rank for group in reusable_homology.homology_groups[:2]] == [
        1,
        0,
    ]
    assert type(normalized).model_validate_json(normalized.model_dump_json()) == normalized


def test_unnormalized_chain_result_retains_reusable_canonical_value():
    source = standard_simplex(0, 2)
    result = unnormalized_chains(UnnormalizedChainsRequest(simplicial_set=source))
    rebuilt = result.chain_complex.model_validate_json(
        result.chain_complex.model_dump_json()
    )
    assert rebuilt == result.chain_complex
    assert result.simplex_bases == (("(0)",), ("(0,0)",), ("(0,0,0)",))


def test_unnormalized_chains_use_requested_exact_coefficient_context():
    source = standard_simplex(1, 2)
    rational = unnormalized_chains(
        UnnormalizedChainsRequest(
            simplicial_set=source, coefficient_ring=CoefficientRing.RATIONAL
        )
    )
    binary = unnormalized_chains(
        UnnormalizedChainsRequest(
            simplicial_set=source,
            coefficient_ring=CoefficientRing.PRIME_FIELD,
            prime=2,
        )
    )
    assert rational.chain_complex.coefficient_ring is CoefficientRing.RATIONAL
    assert binary.chain_complex.coefficient_ring is CoefficientRing.PRIME_FIELD
    assert binary.chain_complex.prime == 2
    assert binary.chain_complex.differential_matrices[0] == (
        (0, 1, 0),
        (0, 1, 0),
    )
    assert differential_squares_to_zero(binary.chain_complex).is_valid


def test_normalized_homology_reports_only_degrees_with_known_incoming_boundary():
    source = standard_simplex(1, 2)
    result = normalized_homology(source)
    assert [group.degree for group in result.homology_groups] == [0, 1]
    assert [group.free_rank for group in result.homology_groups] == [1, 0]
    assert result.nondegenerate_bases[0] == source.sets[0]
    # Degree 2 is intentionally absent: a degree-3 face map would be needed.
    assert result.chain_complex.degree_max == 2
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_normalized_homology_of_triangle_boundary_has_circle_group():
    # The three-edge simplicial circle has H_0 = Z and H_1 = Z. Its complete
    # degree-0..2 simplicial-set prefix contains the differential needed for
    # both groups, while the unused formal top group is not returned.
    from jacobian.math.topology.operations import canonicalize
    from jacobian.math.topology.simplicial_sets import simplicial_set_from_complex
    from jacobian.math.topology.simplicial_sets.complex_conversion_models import (
        SimplicialComplexPrefixRequest,
    )

    circle = canonicalize(("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c"))).complex
    source = simplicial_set_from_complex(
        SimplicialComplexPrefixRequest(complex=circle, max_degree=2)
    ).simplicial_set
    result = normalized_homology(source)
    assert [group.free_rank for group in result.homology_groups] == [1, 1]
    assert all(not group.torsion_generators for group in result.homology_groups)


def test_normalized_homology_manifest_example_is_composable():
    from jacobian.math.topology.simplicial_sets.maps_tools import TOOLS

    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "topology.simplicial_set.homology.compute"
    )
    request = tool.request_type.model_validate(tool.examples[0].input)
    result = tool.run(request)
    assert [group.degree for group in result.homology_groups] == [0, 1]
    assert result.chain_complex.basis_sizes == tuple(
        len(level) for level in result.nondegenerate_bases
    )


def test_output_admission_counts_serialized_tables_and_repeated_basis(monkeypatch):
    source = standard_simplex(1, 3)
    sizes = tuple(len(level) for level in source.sets)
    cells = sum(sizes[n - 1] * sizes[n] for n in range(1, len(sizes)))
    source_bytes = len(source.model_dump_json().encode("utf-8"))
    basis_bytes = len(
        json.dumps(source.sets, ensure_ascii=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    expected = (
        chains_module._CHAIN_RESULT_JSON_OVERHEAD_BOUND
        + source_bytes
        + basis_bytes
        + 12 * cells
    )
    assert chains_module._estimate_output_bytes(source, sizes) == expected
    assert '"face_maps"' in source.model_dump_json()
    assert '"degeneracy_maps"' in source.model_dump_json()

    monkeypatch.setattr(
        chains_module, "MAX_UNNORMALIZED_CHAIN_OUTPUT_BYTES", expected - 1
    )
    with pytest.raises(OperationResourceAdmissionError):
        chains_module._preflight(source)


def test_normalized_chain_output_is_admitted_before_identity_replay(monkeypatch):
    source = standard_simplex(1, 2)

    def reject(*_args, **_kwargs):
        raise AssertionError("identity replay started before output admission")

    monkeypatch.setattr(maps_module, "from_tables", reject)
    monkeypatch.setattr(
        maps_module,
        "MAX_NORMALIZED_CHAIN_OUTPUT_BYTES",
        maps_module._normalized_output_byte_bound(
            source,
            sum(
                len(source.sets[degree - 1]) * len(source.sets[degree])
                for degree in range(1, source.max_degree + 1)
            ),
        )
        - 1,
    )
    with pytest.raises(OperationResourceAdmissionError):
        normalized_chains(source)
