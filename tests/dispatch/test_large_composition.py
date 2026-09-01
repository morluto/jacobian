"""Producer-consumer composition across the shared dispatch boundary."""

from itertools import islice

from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_dispatch_accepts_valid_payload_beyond_default_transport_bytes() -> None:
    """Shared dispatch must not impose an unconfigured transport ceiling."""
    vertex_prefix = chr(0x1D567) * 60
    edge_prefix = chr(0x1D556) * 60
    vertices = [f"{vertex_prefix}{index:04}" for index in range(256)]
    memberships = islice(
        (
            (vertices[first], vertices[second], vertices[third], vertices[fourth])
            for first in range(253)
            for second in range(first + 1, 254)
            for third in range(second + 1, 255)
            for fourth in range(third + 1, 256)
        ),
        9_000,
    )
    edges = [
        [f"{edge_prefix}{index:04}", list(members)]
        for index, members in enumerate(memberships)
    ]
    payload = {"hypergraph": {"vertices": vertices, "edges": edges}}

    assert len(encode_strict_json(payload)) > CanonicalLimits().max_input_bytes
    result = invoke_operation(
        "hypergraph.parameters.compute", payload, Catalog.open()
    )

    assert result.output["edge_count"] == 9_000
    assert result.output["total_incidences"] == 36_000
