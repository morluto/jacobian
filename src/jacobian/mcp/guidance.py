"""Agent-facing guidance for Jacobian's MCP surface."""

from __future__ import annotations

SERVER_DESCRIPTION = (
    "Search and run atomic, composable Jacobian tools for higher mathematics."
)

SERVER_INSTRUCTIONS = (
    "Jacobian provides local typed operations for mathematical computation and "
    "structural analysis. Reach for math.find and math.run proactively when a problem "
    "contains an exact computation, finite search, or structural analysis that may "
    "match a public MCP operation. Use math.find to discover candidates, math.inspect "
    "to read one exact contract, and math.run to execute it. Each call returns an "
    "operation-owned canonical mathematical value that a caller may retain and reuse "
    "when a later contract accepts it."
)

MATH_FIND_DESCRIPTION = """\
Find public Jacobian MCP operations for one local mathematical need. Describe what the
operation should establish in ordinary mathematical language: preserve the supplied
objects and constraints, the computation or decision, and whether you need a value,
witness, certificate, profile, or exhaustive result. A short phrase is enough; do not
translate the need into catalog tags or submit the surrounding proof goal.

Use `namespace` only when the primary operation-ID namespace is already known with high
confidence. Follow `next_cursor` with the same need and namespace to continue. Ordered
matches are deterministic retrieval candidates, not applicability claims; call
math.inspect on a promising operation before math.run. Read `operation://catalog` only
when the complete bulk catalog is genuinely needed.

Examples:
- `{"need":"exact determinant of a rational matrix","namespace":"matrix","limit":3}`
- `{"need":"decide associativity of a finite operation table and return a counterexample triple"}`
"""

MATH_INSPECT_DESCRIPTION = """\
Inspect one installed Jacobian operation by exact `operation_id`. The result contains
the authoritative input and output schemas plus operator-authored examples. Use this
after math.find and before the first math.run call. Unknown IDs return a structured
resolution error.

Example:
- `{"operation_id":"polynomial.compute.gcd"}`
"""

MATH_RUN_DESCRIPTION = """\
Run one installed math tool by ID with its typed `payload`. A successful call returns
the operation-owned canonical mathematical value in `output`; a later inspected
operation may accept that complete value unchanged. Read its fields to determine what
the calculation established. MCP reports malformed payloads, unknown IDs, and host
failures as tool errors, not as mathematical results. If the payload shape is unknown,
inspect the exact operation with math.inspect. When it publishes an `examples` item, copy
and adapt that item's `input` object as the `payload`; otherwise, form the payload from
the input schema and its field descriptions. Do not call math.run with an empty
`payload` merely to discover required fields; inspection is authoritative.

Timeout, incomplete search, and missing witnesses appear only in the concrete domain
result that owns them; none is a mathematical conclusion by itself.

Examples:
- `{"operation_id":"integer.compute.extended_gcd","payload":{"left":"84","right":"30"}}`
"""
