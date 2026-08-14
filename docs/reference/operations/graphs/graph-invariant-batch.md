# Graph invariants

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

Graph invariants are individual direct operations, not selections from a
server-side registry. The catalog includes clique number, independence number,
chromatic number, diameter, radius, girth, edge and vertex connectivity,
Eulerian status, spanning-tree count, triangle count, and maximum matching.

Each operation accepts one typed finite graph and returns only its own bounded
result. Search the catalog for the exact request and result schema; a caller
that needs several invariants composes several ordinary calls.
