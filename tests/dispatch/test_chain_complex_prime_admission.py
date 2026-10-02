"""Prime admission remains typed through every shared chain-field caller."""

from __future__ import annotations

import json
from typing import Any

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.topology.chain_complexes.operations import (
    differential_squares_to_zero,
    homology_groups,
)
from jacobian.math.topology.chain_complexes.values import ChainComplexValue


def _complex(prime: int) -> dict[str, Any]:
    return {
        "coefficient_ring": "GF_p",
        "prime": prime,
        "degree_min": 0,
        "degree_max": 0,
        "basis_sizes": [1],
        "differential_matrices": [],
    }


@pytest.mark.parametrize(
    "operation_id",
    ["chain_complex.homology.compute", "chain_complex.verify_differential.compute"],
)
@pytest.mark.parametrize("prime", [4, 15, 1_000_000])
def test_native_and_dispatch_share_semantic_prime_admission(
    operation_id: str, prime: int
) -> None:
    payload = {"complex": _complex(prime)}
    catalog = Catalog.open()
    operation = catalog.operation(operation_id)
    assert operation is not None
    # Composite modulus remains structurally representable; only its consumer
    # establishes that the declared prime-field arithmetic is defined.
    operation.request_type.model_validate_json(json.dumps(payload), strict=True)
    source = ChainComplexValue.model_validate_json(
        json.dumps(payload["complex"]), strict=True
    )
    native = (
        homology_groups
        if operation_id == "chain_complex.homology.compute"
        else differential_squares_to_zero
    )
    with pytest.raises(OperationDomainValidationError) as native_error:
        native(source)
    with pytest.raises(OperationDomainValidationError) as dispatched:
        invoke_operation(operation_id, payload, catalog)
    assert (
        native_error.value.errors()
        == dispatched.value.errors()
        == (
            {
                "loc": ("complex", "prime"),
                "type": "chain_complex.prime_not_prime",
                "msg": f"prime {prime} is not prime",
            },
        )
    )


@pytest.mark.parametrize(
    ("operation_id", "payload", "location"),
    [
        (
            "chain_complex.construct.compute",
            {
                "coefficient_ring": "GF_p",
                "prime": 4,
                "basis_sizes": [1],
                "differential_matrices": [],
            },
            ("prime",),
        ),
        *[
            (
                operation_id,
                {
                    "chain_map": {
                        "source": _complex(4),
                        "target": _complex(4),
                        "map_matrices": [[["1"]]],
                    }
                },
                ("chain_map", "source", "prime"),
            )
            for operation_id in (
                "chain_complex.verify_chain_map.compute",
                "chain_complex.mapping_cone.compute",
            )
        ],
        (
            "chain_complex.tensor_product.compute",
            {"left": _complex(4), "right": _complex(4)},
            ("left", "prime"),
        ),
        (
            "chain_complex.tensor_product.compute",
            {"left": _complex(2), "right": _complex(4)},
            ("right", "prime"),
        ),
        (
            "homological.filtered_chain_complex.associated_graded.compute",
            {
                "complex": _complex(4),
                "filtration": [{"subspaces": [{"vectors": [["1"]]}]}],
            },
            ("complex", "prime"),
        ),
    ],
)
def test_shared_prime_admission_preserves_each_request_path(
    operation_id: str, payload: dict[str, Any], location: tuple[str, ...]
) -> None:
    with pytest.raises(OperationDomainValidationError) as caught:
        invoke_operation(operation_id, payload, Catalog.open())
    assert caught.value.errors() == (
        {
            "loc": location,
            "type": "chain_complex.prime_not_prime",
            "msg": "prime 4 is not prime",
        },
    )


def test_true_field_false_differential_stays_a_mathematical_verdict() -> None:
    source = {
        **_complex(3),
        "degree_max": 2,
        "basis_sizes": [1, 1, 1],
        "differential_matrices": [[["1"]], [["1"]]],
    }
    catalog = Catalog.open()
    result = invoke_operation(
        "chain_complex.verify_differential.compute", {"complex": source}, catalog
    )
    assert result.output["is_valid"] is False
    assert result.output["complex"] == source
    with pytest.raises(ValueError, match=r"violates d\^2=0"):
        invoke_operation("chain_complex.homology.compute", {"complex": source}, catalog)


@pytest.mark.parametrize(
    ("operation_id", "code", "location"),
    [
        (
            "topology.simplicial_set.unnormalized_chain_complex.compute",
            "chain_complex.prime_not_prime",
            ("prime",),
        ),
        (
            "topology.simplicial_set.degenerate_submodule.compute",
            "simplicial_set.degenerate_submodule_scalar_context",
            ("coefficient_ring",),
        ),
        (
            "topology.cubical_complex.chain_complex.compute",
            "cubical_complex.chain_coefficient_invalid",
            ("coefficient_ring", "prime"),
        ),
        (
            "topology.filtered_cubical_complex.lower_star_from_vertices.compute",
            "cubical_complex.lower_star_prime_invalid",
            ("prime",),
        ),
        (
            "topology.filtered_cubical_complex.from_top_cells.compute",
            "cubical_complex.top_cell_prime_invalid",
            ("prime",),
        ),
    ],
)
def test_external_shared_helper_consumers_keep_admission_and_recovery(
    operation_id: str, code: str, location: tuple[str, ...]
) -> None:
    from jacobian.math.topology.cubical_complexes.operations import (
        chain_complex,
        from_top_cell_values,
        lower_star_from_vertices,
    )
    from jacobian.math.topology.simplicial_sets.chains import unnormalized_chains
    from jacobian.math.topology.simplicial_sets.degenerate_submodule import (
        degenerate_submodule,
    )

    catalog = Catalog.open()
    operation = catalog.operation(operation_id)
    assert operation is not None
    payload = json.loads(json.dumps(operation.examples[0].input))
    if operation_id in {
        "topology.simplicial_set.unnormalized_chain_complex.compute",
        "topology.simplicial_set.degenerate_submodule.compute",
        "topology.cubical_complex.chain_complex.compute",
    }:
        payload["coefficient_ring"] = "GF_p"

    def native(request: Any) -> Any:
        if "unnormalized_chain" in operation_id:
            return unnormalized_chains(request)
        if "degenerate_submodule" in operation_id:
            return degenerate_submodule(request)
        if "lower_star" in operation_id:
            return lower_star_from_vertices(
                request.cells, request.vertex_values, request.prime
            )
        if "from_top_cells" in operation_id:
            return from_top_cell_values(
                request.cells, request.top_cell_values, request.prime
            )
        return chain_complex(request.cells, request.coefficient_ring, request.prime)

    payload["prime"] = 4
    request = operation.request_type.model_validate_json(
        json.dumps(payload), strict=True
    )
    with pytest.raises(OperationDomainValidationError) as native_error:
        native(request)
    with pytest.raises(OperationDomainValidationError) as dispatched:
        invoke_operation(operation_id, payload, catalog)
    assert native_error.value.errors() == dispatched.value.errors()
    assert dispatched.value.errors()[0]["type"] == code
    assert dispatched.value.errors()[0]["loc"] == location
    payload["prime"] = 2
    request = operation.request_type.model_validate_json(
        json.dumps(payload), strict=True
    )
    expected = native(request).model_dump(mode="json")
    output = invoke_operation(operation_id, payload, catalog).output
    assert output == expected
    assert (
        operation.result_type.model_validate_json(
            json.dumps(output), strict=True
        ).model_dump(mode="json")
        == output
    )
