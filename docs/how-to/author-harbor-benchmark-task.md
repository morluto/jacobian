# Author a Harbor benchmark task

[Documentation home](../index.md) · [Benchmark contracts](../reference/evaluations/benchmark-contracts.md)

Harbor tasks are external evaluation assets. Choose a bounded, difficult
mathematical claim because it reveals a capability needed for serious
mathematical investigation, not because an existing Jacobian operation makes
it easy. A task can expose an operation gap; tool availability belongs in the
experimental treatment, not in the definition of a good task.

Copy [the task template](../../benchmarks/templates/task/README.md) into a
registered dataset. Keep `instruction.md` and `environment/` agent-visible;
keep `solution/` and `tests/` Oracle/verifier-only. Do not copy an existing
task's ceremonial `answer.txt`, hidden `expected.json` predicate, keyword
gate, or universal certificate union into a new task.

## Write the public contract first

The agent-visible instruction and `environment/submission_schema.json` are the
complete rewarded protocol. They must agree with the verifier:

1. Default to a typed `result`. Add a `witness` only when replay needs a
   distinct finite object that is not already in `result`.
2. Put the smallest task-owned mathematical type in the schema: structured
   rationals, sparse maps, finite enums, or a family-specific certificate.
   Do not score `"2/8"` strings, formula prose, or keyword-bearing sentences.
3. State that equivalent encodings are accepted unless canonicalization is
   itself the mathematical outcome. Do not require lowest terms in public
   prose while the verifier constructs `Fraction`.
4. For a generated family, emit one schema licensed by that task's claim
   family. Do not publish a universal `oneOf` of every certificate kind.

Declare the protocol in `tests/public_contract.json` and generate the marked
submission block plus schema with `benchmarks.tooling.public_contract`. Do not
hand-maintain a second protocol.

## Implement replay, not a gold comparison

The verifier recomputes or checks the claim from the frozen `input.json` copy
and the submitted value. Keep `tests/expected.json` as an Oracle regression
fixture only. A function that loads both input and expected, then scores
equality with hidden expected fields while ignoring the input, is not a
mathematical verifier.

Normalize represented values before comparison: unreduced rationals, scaled
rational functions, and unordered collections (sets, maps, sparse polynomials,
distributions) unless order is part of the public object. Reject booleans as
integers, zero or negative denominators under the stated sign convention, and
resource-bound violations.

Do not award credit for:

- a preferred rendering of an already-checked value;
- substrings, length, or negation regexes over mathematical prose;
- a digest, field copy, or nonempty file that only restates `result`;
- an unread `answer.txt` whose path and hash are the only checks.

Independent propositions must be derived independently. Do not couple a
collision, invertibility, or completeness claim to an unrelated corrupted
field.

## Validate the exact task

```sh
make harbor-prepare-task DATASET=<dataset-id> TASKS="<task-id>"
make harbor-validate-task DATASET=<dataset-id> TASKS="<task-id>"
```

`harbor-prepare-task` formats selected task Python and refreshes public-contract
and verifier checksums. `harbor-validate-task` is the source-read-only leaf
gate: static contracts, host tests, then the exact Oracle. Neither command
starts a model. Use `make harbor-check` only when changing shared Harbor
tooling, schemas, registry, or suite policy. Create a snapshot lock only when
freezing an intentional evaluation or publication set.

The task's verifier owns correctness; Jacobian owns neither that verifier nor
its data lifecycle. Keep the task's score distinct from Jacobian's two-tool
operation contract.
