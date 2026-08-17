"""Canonical small-order graph6 decoding."""

from __future__ import annotations

from hashlib import sha256
from typing import Literal, Self

from pydantic import Field, StrictInt, StrictStr, model_validator

from jacobian._digest import Sha256Digest
from jacobian._models import StrictModel
from jacobian.canonical import canonicalize_json


class Graph6Edge(StrictModel):
    first: StrictInt = Field(ge=0, le=62)
    second: StrictInt = Field(ge=0, le=62)

    @model_validator(mode="after")
    def require_canonical_endpoints(self) -> Self:
        if self.first >= self.second:
            raise ValueError("graph6 edge endpoints must be strictly increasing")
        return self


class Graph6DecodeValue(StrictModel):
    graph6: StrictStr = Field(min_length=1, max_length=352)
    order: StrictInt = Field(ge=0, le=62)
    edges: tuple[Graph6Edge, ...] = Field(max_length=1891)
    degrees: tuple[StrictInt, ...] = Field(max_length=62)
    graph_digest: Sha256Digest
    format: Literal["GRAPH6_SMALL_ORDER"] = "GRAPH6_SMALL_ORDER"
    bit_order: Literal["COLUMN_MAJOR_UPPER_TRIANGLE"] = "COLUMN_MAJOR_UPPER_TRIANGLE"

    @model_validator(mode="after")
    def bind_dimensions(self) -> Self:
        if len(self.degrees) != self.order:
            raise ValueError("graph6 degree sequence must match graph order")
        pairs = tuple((edge.first, edge.second) for edge in self.edges)
        if pairs != tuple(sorted(pairs)) or len(pairs) != len(set(pairs)):
            raise ValueError("graph6 edges must be unique and sorted")
        if any(edge.second >= self.order for edge in self.edges):
            raise ValueError("graph6 edge endpoint exceeds graph order")
        expected = [0] * self.order
        for edge in self.edges:
            expected[edge.first] += 1
            expected[edge.second] += 1
        if tuple(expected) != self.degrees:
            raise ValueError("graph6 degree sequence does not match edges")
        return self


def decode_graph6(encoded: str) -> Graph6DecodeValue:
    value = encoded[10:] if encoded.startswith(">>graph6<<") else encoded
    if not value or value[0] in {":", "&"}:
        raise ValueError("only standard graph6 is supported")
    codes = [ord(character) - 63 for character in value]
    if any(code < 0 or code > 63 for code in codes) or codes[0] == 63:
        raise ValueError("graph6 payload is malformed or uses an extended header")
    order = codes[0]
    bit_count = order * (order - 1) // 2
    if len(codes) != 1 + (bit_count + 5) // 6:
        raise ValueError("graph6 payload length does not match its order header")

    from jacobian.math.graphs import _networkx

    graph = _networkx.graph_from_graph6(value)
    if graph.number_of_nodes() != order:
        raise ValueError("graph6 payload is malformed or uses an extended header")
    if _networkx.graph6_canonical_bytes(graph) != value.encode("ascii"):
        raise ValueError("unused graph6 padding bits must be zero")
    edges = tuple(
        Graph6Edge(first=first, second=second)
        for first, second in sorted(
            (min(left, right), max(left, right)) for left, right in graph.edges
        )
    )
    degrees = tuple(int(graph.degree(vertex)) for vertex in range(order))
    digest_payload = {
        "order": order,
        "edges": [[edge.first, edge.second] for edge in edges],
    }
    return Graph6DecodeValue(
        graph6=value,
        order=order,
        edges=edges,
        degrees=degrees,
        graph_digest="sha256:" + sha256(canonicalize_json(digest_payload)).hexdigest(),
    )


__all__ = ["Graph6DecodeValue", "Graph6Edge", "decode_graph6"]
