from __future__ import annotations

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.topology.chain_complexes._models import (
    MappingConeRequest,
    VerifyChainMapRequest,
)
from jacobian.math.topology.chain_complexes._tools import (
    _mapping_cone,
    _verify_chain_map,
)
from jacobian.math.topology.chain_complexes.values import ChainMapValue
from jacobian.math.topology.simplicial_sets._models import FiniteTruncatedSimplicialSet
from jacobian.math.topology.simplicial_sets.maps import (
    TruncatedSimplicialMap,
    induced_normalized_chain_map,
)
from jacobian.math.topology.simplicial_sets.operations import from_tables
from jacobian.math.topology.simplicial_sets.standard import standard_simplex


def _identity_map(
    simplicial_set: FiniteTruncatedSimplicialSet,
) -> TruncatedSimplicialMap:
    return TruncatedSimplicialMap(
        source=simplicial_set,
        target=simplicial_set,
        maps=tuple(tuple(range(len(level))) for level in simplicial_set.sets),
    )


ChainMatrix = tuple[tuple[int | Fraction, ...], ...]


def _matmul(
    left: ChainMatrix,
    right: ChainMatrix,
    *,
    columns: int,
) -> ChainMatrix:
    rows = len(left)
    inner = len(left[0]) if left else len(right)
    return tuple(
        tuple(
            sum(left[i][k] * right[k][j] for k in range(inner)) for j in range(columns)
        )
        for i in range(rows)
    )


def test_identity_induces_identity_on_exact_normalized_axes_and_round_trips() -> None:
    simplex = standard_simplex(1, 2)
    chain_map = induced_normalized_chain_map(_identity_map(simplex))

    assert chain_map.source.basis_sizes == (2, 1, 0)
    assert chain_map.source == chain_map.target
    assert chain_map.source_basis_labels == (("(0)", "(1)"), ("(0,1)",), ())
    assert chain_map.map_matrices == (
        ((1, 0), (0, 1)),
        ((1,),),
        (),
    )

    decoded = ChainMapValue.model_validate_json(chain_map.model_dump_json())
    assert _verify_chain_map(VerifyChainMapRequest(chain_map=decoded)).is_valid
    cone = _mapping_cone(MappingConeRequest(chain_map=decoded))
    assert cone.chain_map == decoded


def test_published_example_executes_and_decodes_through_catalog() -> None:
    catalog = Catalog.open()
    operation = catalog.operation(
        "topology.simplicial_set.map.induced_chain_map.compute"
    )
    assert operation is not None and operation.examples
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    decoded = ChainMapValue.model_validate_json(json.dumps(result.output))
    assert decoded.source_basis_labels == (("(0)", "(1)"), ("(0,1)",), ())


def test_constant_map_sends_degenerate_image_to_zero_and_obeys_chain_equation() -> None:
    source = standard_simplex(1, 2)
    target = standard_simplex(0, 2)
    constant = TruncatedSimplicialMap(
        source=source,
        target=target,
        maps=((0, 0), (0, 0, 0), (0, 0, 0, 0)),
    )
    chain_map = induced_normalized_chain_map(constant)

    assert chain_map.source_basis_labels == (("(0)", "(1)"), ("(0,1)",), ())
    assert chain_map.target_basis_labels == (("(0)",), (), ())
    assert chain_map.map_matrices == (((1, 1),), (), ())

    source_d = chain_map.source.differential_matrices
    target_d = chain_map.target.differential_matrices
    for degree, component in enumerate(chain_map.map_matrices[:-1]):
        columns = chain_map.source.basis_sizes[degree + 1]
        left = _matmul(
            target_d[degree], chain_map.map_matrices[degree + 1], columns=columns
        )
        right = _matmul(component, source_d[degree], columns=columns)
        assert left == right


def test_empty_prefix_induces_zero_rank_chain_map_without_losing_axes() -> None:
    checked = from_tables(1, ((), ()), (((), ()),), (((),),))
    assert checked.simplicial_set is not None
    empty = checked.simplicial_set
    map_value = TruncatedSimplicialMap(source=empty, target=empty, maps=((), ()))

    chain_map = induced_normalized_chain_map(map_value)
    assert chain_map.source.basis_sizes == (0, 0)
    assert chain_map.map_matrices == ((), ())
    assert chain_map.source_basis_labels == ((), ())


def test_nonnatural_map_and_invalid_carrier_are_rejected_at_operation_boundary() -> (
    None
):
    simplex = standard_simplex(1, 2)
    nonnatural = TruncatedSimplicialMap(
        source=simplex,
        target=simplex,
        maps=((0, 1), (1, 1, 2), (0, 1, 2, 3)),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        induced_normalized_chain_map(nonnatural)
    assert error.value.errors()[0]["type"] == "simplicial_map.face_naturality_failed"

    bad_faces = ((0, 0, 1), simplex.face_maps[0][1])
    bad_carrier = simplex.model_copy(
        update={"face_maps": (bad_faces, *simplex.face_maps[1:])}
    )
    malformed = TruncatedSimplicialMap(
        source=bad_carrier,
        target=simplex,
        maps=((0, 1), (0, 1, 2), (0, 1, 2, 3)),
    )
    with pytest.raises(OperationDomainValidationError):
        induced_normalized_chain_map(malformed)


def test_chain_map_value_binds_labels_shapes_and_exact_coefficients() -> None:
    simplex = standard_simplex(0, 1)
    map_value = induced_normalized_chain_map(_identity_map(simplex))
    payload = json.loads(map_value.model_dump_json())
    payload["source_basis_labels"][0] = []
    with pytest.raises(ValidationError, match="basis labels"):
        ChainMapValue.model_validate_json(json.dumps(payload))
