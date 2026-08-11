"""Agent-facing guidance for Jacobian's MCP surface."""

from __future__ import annotations

SERVER_DESCRIPTION = (
    "Use specialized exact and symbolic mathematics with explicit scope, "
    "completeness, assurance, and optional independent verification."
)

SERVER_INSTRUCTIONS = (
    "Use Jacobian whenever a task may benefit from a specialized exact mathematical "
    "operation, including matrix or polynomial computation, symbolic transformation, "
    "structural analysis, examples or counterexamples, bounded search, formal-environment "
    "inspection, or requested independent verification. This applies even when the "
    "user does not name Jacobian and shell code could also calculate the result. Unless "
    "an exact installed capability ID and its typed contract are already available, call "
    "math.find with a plain-language desired local mathematical outcome; no capability "
    "ID is required. math.run may execute a known contract directly. "
    "Do not report that no specialized mathematical operation is available without "
    "checking math.find. When independent checking is requested, multiple calculations "
    "or programs authored by the same model are not independent checker evidence. "
    "Search or browse again whenever the objective or available evidence changes. "
    "The model owns representation, decomposition, composition, iteration, verification "
    "timing, and stopping. Results keep execution status, scope, completeness, "
    "mathematical conclusion, and assurance separate. No descriptor match, timeout, "
    "bounded or exhausted search, or failure to find a witness is a mathematical "
    "conclusion. Only assurance level VERIFIED with a local verification record is "
    "verified. A verification record for an input, premise, factorization, or related "
    "artifact does not verify a model-derived conclusion; the record must be bound to "
    "the exact final claim."
)

MATH_FIND_DESCRIPTION = """\
Search or inspect installed math tools by desired outcome or exact ID. Use when a
task may benefit from exact computation, search, structural analysis, or a separate
checker tool—even if shell code could also calculate the answer.

Forms:
- `query`: plain-language mathematical outcome (compact tool cards).
- Optional `domain` filter; `limit` 1-20 (default 5).
- Omit arguments to browse; follow `next_cursor` with the same filters to continue.
- Ranking is deterministic lexical retrieval; matches are not recommendations.
- `capability_id`: exact inspect (SUMMARY / CONTRACT / FULL views).

Checker tools are separate IDs (often `*.verify`), not a switch on producers.

Examples:
- `{"query":"compute an exact matrix determinant","domain":"matrix","limit":3}`
- `{"query":"find a counterexample to associativity","domain":"universal_algebra"}`
- `{"capability_id":"polynomial.compute.gcd","view":"CONTRACT"}`
"""

MATH_RUN_DESCRIPTION = """\
Run one installed math tool by ID with its typed `payload`. Read the mathematical
value in `output` first, then execution status. If the payload shape is unknown,
use math.find with view CONTRACT.

Ordinary tools return calculations. Independent checking uses a separate checker
tool ID (for example `polynomial.identity.verify` or `case.partition.finite.verify`),
not a switch on the producer. Failed or incomplete runs are not mathematical conclusions.

Examples:
- `{"capability_id":"integer.compute.gcd","payload":{"left":"84","right":"30"}}`
- `{"capability_id":"polynomial.identity.verify","payload":{"variables":["x"],"left":{"terms":[]},"right":{"terms":[]}}}`
"""
