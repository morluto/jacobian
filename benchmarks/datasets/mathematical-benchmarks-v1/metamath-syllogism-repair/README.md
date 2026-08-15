# Metamath syllogism proof repair

Repair two misplaced assertion applications in an alpha-renamed Metamath
syllogism proof and independently replay the resulting substitution stack.

## Provenance

- Source: `metamath/set.mm`, `set.mm`, theorem `syl`
- Revision: `044c9e9f138dd05c518880800245ebe059c35c7e`
- License: CC0-1.0 / public domain
- Derivation: the variables are alpha-renamed and two assertion labels are
  transposed to create a deterministic repair task.

## Quality and shortcut audit

Quality score: **87/100**. Benchmark family: **Regression**. The single
objective is formal proof-trace repair. Difficulty is **Hard (provisional)**:
the agent must recover a typed reverse-Polish proof and produce a consistent
per-step substitution transcript; empirical baseline calibration is pending.

The source proof is public, but copying its compressed string does not satisfy
the alpha-renamed replay contract. The checker independently applies every
assertion, rejects direct-target and partial-trace shortcuts, and requires
exactly two edits. This adds kernel-style substitution
replay rather than another abstract dependency DAG or propositional rewrite.

Assurance is capped at `COMPUTED`: the verifier checks this frozen formal
fragment and its assertion registry, not the full upstream database or an
external Metamath executable.
