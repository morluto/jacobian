"""Agent-facing guidance for Jacobian's MCP surface."""

from __future__ import annotations

SERVER_DESCRIPTION = (
    "Search and run atomic, composable Jacobian tools for higher mathematics."
)

SERVER_INSTRUCTIONS = (
    "RECOMMENDED for bounded mathematical computation:\n"
    "- math.run instead of an ad-hoc CAS or solver script, or lengthy exact manual "
    "calculation: returns a typed result from an installed operation.\n"
    "- math.find before math.run when no matching operation is known: searches the "
    "local operation catalog and provides exact input schemas and examples.\n\n"
    "Use the typed math.run output for the requested claim. The model owns "
    "representation, decomposition, composition, and stopping. Jacobian retains no "
    "workspace, artifact, replay record, or proof session. A timeout, incomplete "
    "search, or missing witness is not a mathematical conclusion."
)

MATH_FIND_DESCRIPTION = """\
Search, browse, or inspect locally installed Jacobian math tools. This is authoritative
for local discovery and exact operation inspection; internet search is not. Use math.find
when a task may benefit from exact computation, search, or structural analysis.

Forms:
- `request.op="search"`: plain-language mathematical outcome (compact cards).
- `request.op="browse"`: compact operation cards in operation-ID order, optionally
  filtered to one domain; use this to map an unfamiliar domain.
- `search` accepts optional `domain` and `limit` 1-20 (default 5); `browse` accepts
  the same filters (default limit 20).
- Follow `next_cursor` with the same search query or browse filters to continue.
- Ranking is deterministic lexical retrieval; matches are not recommendations.
- `request.op="inspect"`: exact ID with authoritative schemas and examples.
- `operation://catalog` remains an exact bulk export, not the ordinary discovery path.

Examples:
- `{"request":{"op":"search","query":"exact matrix determinant","domain":"matrix","limit":3}}`
- `{"request":{"op":"browse","domain":"matrix","limit":20}}`
- `{"request":{"op":"search","query":"counterexample to associativity"}}`
- `{"request":{"op":"search","query":"check a bounded Lean source snippet","domain":"lean"}}`
- `{"request":{"op":"inspect","operation_id":"polynomial.compute.gcd"}}`
"""

MATH_RUN_DESCRIPTION = """\
Run one installed math tool by ID with its typed `payload`. A successful call returns
the operation-owned mathematical value in `output`; read its fields to determine what
the calculation established. MCP reports malformed payloads, unknown IDs, and host
failures as tool errors, not as mathematical results. If the payload shape is unknown,
inspect the exact operation with math.find, select one item from `examples`, and copy
and adapt that item's `input` object as the `payload`. Do not call math.run with an
empty `payload` merely to discover required fields; inspection is authoritative.

Timeout, incomplete search, and missing witnesses appear only in the concrete domain
result that owns them; none is a mathematical conclusion by itself.

Examples:
- `{"operation_id":"integer.compute.gcd","payload":{"left":"84","right":"30"}}`
"""
