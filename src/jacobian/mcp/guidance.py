"""Agent-facing guidance for Jacobian's MCP surface."""

from __future__ import annotations

SERVER_DESCRIPTION = (
    "Search and run atomic, composable Jacobian tools for higher mathematics."
)

SERVER_INSTRUCTIONS = (
    "Jacobian provides local typed operations for mathematical computation and "
    "structural analysis. Reach for math.find and math.run proactively when a problem "
    "contains an exact computation, finite search, or structural analysis that may "
    "match a public MCP operation. Use math.find to match candidates or inspect one "
    "exact contract, and math.run to execute it. Each call returns an "
    "operation-owned canonical mathematical value that a caller may retain and reuse "
    "when a later contract accepts it."
)

MATH_FIND_DESCRIPTION = """\
Find or inspect public Jacobian MCP operations.

Forms:
- `request.op="match"`: describe one local mathematical need in ordinary language.
  Preserve established mathematical names from the task, the supplied objects and
  constraints, the requested computation or decision, the full scalar, batch, or
  exhaustive scope, and whether the requested result is a value, witness,
  certificate, obstruction, profile, or complete enumeration. Do not replace a
  supplied named property only with its expanded definition, translate the need into
  catalog tags, or submit the surrounding proof goal.
- `request.op="inspect"`: pass one exact `operation_id` to receive its authoritative
  input and output schemas plus operator-authored examples.

For matching, use `namespace` only when the primary operation-ID namespace is already
known with high confidence. Matching returns 10 candidates by default; request up to
20 when a wider first page is useful. Follow `next_cursor` with the same need and
namespace to continue. Ordered matches are deterministic retrieval candidates, not
applicability claims; inspect a promising operation before math.run. Read
`operation://catalog` only when the complete bulk catalog is genuinely needed.

Examples:
- `{"request":{"op":"match","need":"exact determinant of a rational matrix","namespace":"matrix","limit":3}}`
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
