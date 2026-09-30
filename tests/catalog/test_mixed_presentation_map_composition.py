"""Public mixed composition survives the result schema and downstream decoding."""

import json

from jsonschema import Draft202012Validator

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cohomology.operations._models import SimplicialMap
from jacobian.math.topology.edge_paths._models import (
    FundamentalGroupMapRequest,
    FundamentalGroupMapResult,
    PresentationTransportedSimplicialMap,
)
from jacobian.math.topology.edge_paths.presentation_maps import (
    induced_fundamental_group_map,
)


def test_public_mixed_composition_can_be_composed_again_after_wire_round_trip() -> None:
    source = canonical_complex(("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c")))
    target = canonical_complex(("x", "y", "z"), (("x", "y"), ("x", "z"), ("y", "z")))
    catalog = Catalog.open()
    path = invoke_operation(
        "topology.simplicial.fundamental_group.basepoint_change.compute",
        {
            "path": {
                "complex": source.model_dump(mode="json"),
                "source_base_vertex": "a",
                "target_base_vertex": "b",
                "path_vertices": ["a", "b"],
            }
        },
        catalog,
    ).output
    mapped = induced_fundamental_group_map(
        FundamentalGroupMapRequest(
            map=SimplicialMap(source=source, target=target, vertex_map=target.vertices),
            source_base_vertex="b",
            target_base_vertex="y",
        )
    )
    composition_id = "topology.simplicial.fundamental_group.map.compose.compute"
    output = invoke_operation(
        composition_id,
        {"first": path, "second": mapped.model_dump(mode="json")},
        catalog,
    ).output
    Draft202012Validator(
        FundamentalGroupMapResult.model_json_schema(mode="serialization")
    ).validate(output)
    decoded = FundamentalGroupMapResult.model_validate_json(json.dumps(output))
    assert decoded.source_presentation.complex == source
    assert decoded.target_presentation.complex == target
    assert isinstance(decoded.map, PresentationTransportedSimplicialMap)
    assert decoded.map.basepoint_path.path_vertices == ("x", "y")
    identity = induced_fundamental_group_map(
        FundamentalGroupMapRequest(
            map=SimplicialMap(source=target, target=target, vertex_map=target.vertices),
            source_base_vertex="y",
            target_base_vertex="y",
        )
    )
    again = invoke_operation(
        composition_id,
        {"first": output, "second": identity.model_dump(mode="json")},
        catalog,
    ).output
    assert again == output
