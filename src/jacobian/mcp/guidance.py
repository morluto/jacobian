"""Agent-facing guidance for Jacobian's MCP surface."""

from __future__ import annotations

SERVER_DESCRIPTION = "Direct typed, atomic, composable tools for higher mathematics."

SERVER_INSTRUCTIONS = (
    "Jacobian provides local typed operations for mathematical computation and "
    "structural analysis. Discover and call the matching operation directly when a "
    "problem contains an exact computation, finite search, or structural analysis. "
    "Each direct call returns an operation-owned canonical mathematical value that a "
    "caller may retain and reuse when a later contract accepts it."
)

SERVER_INSTRUCTIONS_WITH_MATH_FIND = (
    f"{SERVER_INSTRUCTIONS} This evaluation surface also exposes math.find as a "
    "mathematical-vocabulary control for alternate terminology, neighboring "
    "postconditions, and exact contract inspection; it is never required before a "
    "direct operation call."
)

MATH_FIND_DESCRIPTION = """\
Evaluation-only vocabulary control for searching, browsing, or inspecting Jacobian's
public mathematical operations. Generic client tool search should discover an ordinary
directly callable operation; this control isolates whether alternate terminology,
neighboring postconditions, or exact catalog inspection adds measurable value. It does
not enumerate the broader native Python API.

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
- Inspection is optional: every admitted operation already publishes the same request
  and result contract as its direct MCP tool schema.
- `operation://catalog` is the exact bulk-export fallback when the full catalog is
  needed, not the ordinary discovery path.

Examples:
- `{"request":{"op":"search","query":"matrix determinant","namespace":"matrix","limit":3}}`
- `{"request":{"op":"browse","namespace":"matrix","limit":20}}`
- `{"request":{"op":"search","query":"counterexample to associativity"}}`
- `{"request":{"op":"inspect","operation_id":"polynomial.compute.gcd"}}`
"""
