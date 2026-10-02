---
name: audit-public-operation-contracts
description: Design or audit a Jacobian operation’s mathematical contract, boundedness, exact results, and composition.
---

# Audit Public Operation Contracts

Determine whether an operation is a bounded, truthful, composable mathematical
instrument. Record the revision, operation IDs, inspected scope, and whether the
request includes implementation. Audit-only work does not authorize repository
or external mutations.

## Design before implementation

Before adding a public operation or library abstraction, apply the
[vocabulary audit](../audit-mathematical-vocabulary/SKILL.md) to the proposed
boundary. Keep concise evidence in the existing issue or PR, reusing what is
already established rather than creating a mandatory review form:

- State the exact mathematical postcondition and essential hypotheses.
- Name the nearest existing operation or composition and the specific result
  it cannot supply. Give a smallest separating example and a nearby case the
  proposal deliberately does not solve.
- Walk one input through its canonical result to an intended consumer. Decide
  only relevant questions about source identity, multiplicity, ordering,
  empty or degenerate carriers, finite versus infinite solution sets, and one
  witness versus exhaustive output.
- Choose the appropriate disposition: public operation, native-only function,
  shared representation, private helper, regression fixture, or caller reasoning.
  A composition or no-gap finding is a successful outcome, not a reason to
  manufacture another operation.

A justified boundary has one stable mathematical result, not a whole-proof API,
assurance wrapper, workflow state, or compatibility metadata. Keep the existing
composition and scale-first obligations below; a separating example motivates
investigation but does not establish an admitted implementation.

## Trace the contract

Treat request representation, canonical values, semantic admission, kernel,
trusted result construction, and downstream consumers as one execution path.
Inspect schemas, declarations, examples, and MCP errors when public projection
is in scope. The implementation alone does not define the advertised contract.

Use the applicable sections of the
[operation library](../../../docs/reference/domain-operation-library.md).
For public admission, consult the
[admission contract](../../../docs/reference/public-operation-admission.md);
for a backend boundary, consult the
[backend contract](../../../docs/reference/mathematical-backends.md).
Record applicable evidence in the requested audit or existing issue/PR
description; do not create a universal review form.

## Probe the plausible failures

Use small deterministic reproductions to test the suspected mechanism:

- accepted requests beyond the backend domain or admitted work/growth bounds;
- useful structured requests rejected by an ambient-size or expansion-based limit;
- repeated admission or computation during validation and result construction;
- lost units, multiplicities, axes, parents, witnesses, or reconstruction data;
- independently supplied claims accepted without establishing the needed property;
- incompatible producer/consumer values, including empty or singular cases;
- implicit changes of ring, field, parent, or axes; and
- advertised semantics that disagree with the typed result or kernel.

Check serialized producer-consumer composition when relevant. Defining-invariant
proof belongs in tests or an admitted caller-claim operation, not replay during
ordinary result construction. Use current official backend documentation and the
pinned implementation for consequential backend claims.

For a representation, performance, admission-limit, or backend-selection
investigation, read [scale and backends](references/scale-and-backends.md).
Audit what the contract unnecessarily excludes as well as what it unsafely
accepts. Preserve the motivating request and exact invariant; a fast but weaker
result is not a scale improvement.

## Establish the finding and finish

After proving a defect, inspect the owner, shared helper, and its callers for the
same mechanism, keeping adjacent candidates confirmed, disproved, or untested.
Choose a repair at the invariant's owning boundary. Preserve a regression that
fails on the base for the intended reason when feasible, using an independent
oracle, defining identity, or adversarial composition rather than source-text
assertions. Select validation through the contributor guide's owning lanes.

Report the public claim, reproduction and observed result, violated invariant,
affected scope, smallest repair, and meaningful proof gaps. If implementation
was authorized, complete the focused repair and affected checks before handing
back; existing authorization persists, but local investigation does not grant
permission for external writes.
