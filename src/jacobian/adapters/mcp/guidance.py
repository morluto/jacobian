"""Agent-facing guidance for Jacobian's MCP surface."""

from __future__ import annotations

SERVER_DESCRIPTION = (
    "Search and run atomic, composable Jacobian tools for higher mathematics."
)

SERVER_INSTRUCTIONS = (
    "Use math.find to discover a typed mathematical function when its ID or input "
    "shape is unknown, then use math.run to calculate one bounded result. The model "
    "owns representation, decomposition, composition, and stopping. Jacobian retains "
    "no workspace, artifact, replay record, or proof session. A timeout, incomplete "
    "search, or missing witness is not a mathematical conclusion."
)

MATH_FIND_DESCRIPTION = """\
Search or inspect locally installed Jacobian math tools by desired outcome or exact ID.
This is authoritative for local search and exact operation inspection; internet search
is not. Read `operation://catalog` when the complete installed inventory is needed. Use
math.find when a task may benefit from exact computation, search, or structural analysis.

Forms:
- `request.op="search"`: plain-language mathematical outcome (compact cards).
- Optional `domain` filter; `limit` 1-20 (default 5).
- Follow `next_cursor` with the same query and filters to continue.
- Ranking is deterministic lexical retrieval; matches are not recommendations.
- `request.op="inspect"`: exact ID with authoritative schemas and examples.

Examples:
- `{"request":{"op":"search","query":"exact matrix determinant","domain":"matrix","limit":3}}`
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
