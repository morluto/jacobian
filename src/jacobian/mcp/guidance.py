"""Agent-facing guidance for Jacobian's MCP surface."""

from __future__ import annotations

SERVER_DESCRIPTION = (
    "Direct typed tools for atomic, composable computations in higher mathematics."
)

SERVER_INSTRUCTIONS = (
    "Jacobian provides local typed operations for mathematical computation and "
    "structural analysis. Each admitted operation is directly callable under its "
    "operation ID with its owner-local input schema and returns its canonical result. "
    "Use math.find only when semantic catalog search or exact contract inspection adds "
    "value beyond client tool discovery. The caller owns composition and may retain a "
    "returned value when a later operation accepts it. math.run remains available "
    "during the direct-tool migration but is not required for ordinary operation calls."
)

MATH_FIND_DESCRIPTION = """\
Search, browse, or inspect public Jacobian MCP operations. This is authoritative for
their local discovery and exact operation inspection; it does not enumerate the broader
native Python API. Use math.find when a task may benefit from one exact computation,
finite search, or structural analysis.

Forms:
- `request.op="search"`: a short atomic mathematical outcome (compact cards), not a
  complete proof goal. Search globally unless the primary namespace is already known.
- `request.op="browse"`: compact operation cards in operation-ID order for one known
  primary namespace.
- `search` accepts optional exact `namespace` and `limit` 1-20 (default 5); `browse`
  accepts the same namespace filter (default limit 20). A namespace matches the first
  operation-ID segment only; declaration tags never filter results.
- Follow `next_cursor` with the same query or namespace filter to continue.
- Ranking is deterministic lexical retrieval; ordered matches are candidates, not
  recommendations or applicability claims.
- `request.op="inspect"`: exact ID with authoritative schemas and examples.
- `operation://catalog` is the exact bulk-export fallback when the full catalog is
  needed, not the ordinary discovery path.

Examples:
- `{"request":{"op":"search","query":"matrix determinant","namespace":"matrix","limit":3}}`
- `{"request":{"op":"browse","namespace":"matrix","limit":20}}`
- `{"request":{"op":"search","query":"counterexample to associativity"}}`
- `{"request":{"op":"inspect","operation_id":"polynomial.compute.gcd"}}`
"""

MATH_RUN_DESCRIPTION = """\
Run one installed math tool by ID with its typed `payload`. A successful call returns
the operation-owned canonical mathematical value in `output`; a later inspected
operation may accept that complete value unchanged. Read its fields to determine what
the calculation established. MCP reports malformed payloads, unknown IDs, and host
failures as tool errors, not as mathematical results. If the payload shape is unknown,
inspect the exact operation with math.find. When it publishes an `examples` item, copy
and adapt that item's `input` object as the `payload`; otherwise, form the payload from
the input schema and its field descriptions. Do not call math.run with an empty
`payload` merely to discover required fields; inspection is authoritative.

Timeout, incomplete search, and missing witnesses appear only in the concrete domain
result that owns them; none is a mathematical conclusion by itself.

Examples:
- `{"operation_id":"integer.compute.extended_gcd","payload":{"left":"84","right":"30"}}`
"""
