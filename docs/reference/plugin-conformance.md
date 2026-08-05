# Plugin conformance kit

[Documentation home](../index.md)

- Status: Capability-package conformance requirements

The conformance kit tests whether an independently installed package crosses
Jacobian's registry, execution, artifact, and assurance boundaries without core
or MCP changes. It is an executable extension gate, not a second unit-test
framework. A conforming package exposes namespaced capability IDs through the
installed catalog; agents inspect them with `math.find` and execute
them with `math.run`.

## Scope

The kit proves that:

- discovery and capability resolution use the sealed installed package;
- every declared capability has a valid descriptor and schema-valid successful
  invocation case;
- failure, malformed output, and timeout remain operational states;
- traversal, symlink, and changed-byte attacks fail closed;
- an untrusted proposing, transforming, searching, or evaluating capability
  cannot promote evidence.

It does not prove that the plugin's mathematics is correct, that its search is
complete, or that installed code is safe for the host. Those properties require
domain tests, authorized checkers, and operator isolation policy.

## Disposable target package

Fault injection belongs in a separate conformance-only package installed into
isolated test state. Do not expose the conformance selector from a production
plugin.

The synthetic package declares ordinary internal `Proposer`, `Refiner`,
`Evaluator`, and `HypothesisTransformer` implementations. A fixture-owned
adapter exposes coherent test operations through namespaced capability IDs; it
does not expose each backend role as a separate agent-facing tool. Fault
selection remains private to this disposable package and is passed through the
invocation payload:

| Value | Required behavior |
| --- | --- |
| `execution-success` | Return a schema-valid finite proposal and complete normally |
| `declared-failure` | Raise a controlled error containing `declared plugin failure` |
| `malformed-output` | Return a non-object response |
| `timeout` | Remain active beyond the supplied wall budget |

The refiner and evaluator return valid ordinary contract responses. The
hypothesis transformer attempts a verified parameter-region label so the
invocation can demonstrate fail-closed rejection.

## Runner inputs

The conformance runner requires:

- an isolated `JacobianRuntime`;
- the installed synthetic plugin artifact URI;
- the expected capability IDs and their successful invocation payloads;
- fault-injection payloads for declared failure, malformed output, and timeout;
- the package implementation file to modify and restore;
- a disposable in-package symlink path and an outside target;
- optionally, an import marker written by the package's `__init__.py`.

For each expected ID, the runner reads the installed descriptor before
invocation. It rejects a missing descriptor, an invalid schema, an invocation
that bypasses the capability registry, or a result that violates the common
execution and assurance contract.

## Standard matrix

| Check | Boundary exercised | Passing result |
| --- | --- | --- |
| Discovery | Installed catalog and `math.find` | Exact descriptor for each declared namespaced capability ID |
| Execution success | Registry resolution and `math.run` | Schema-valid result with no verification promotion |
| Declared failure | Capability worker lifecycle | `ERROR` with the declared detail |
| Malformed output | Capability worker JSON boundary | `ERROR`; response is not accepted as mathematical evidence |
| Timeout | Worker and durable budget | `TIMEOUT` with wall-limit stop reason |
| Path attack | Implementation registration | Traversal is rejected |
| Symlink attack | Whole-package measurement | Package symlink is rejected |
| Changed bytes | Capability resolution | Installed snapshot refuses changed source |
| Evidence promotion | Assurance boundary | Rejected result, no verification record, and `UNVERIFIED` assurance |

Each suite execution uses isolated state so repeated runs execute the installed
capabilities again instead of returning earlier durable results. The runner
removes only the supplied disposable import marker and symlink, and restores
the implementation file after the changed-byte check.

## Discovery without import

An import marker makes the no-import rule observable. Package installation and
capability resolution must leave the marker absent. The first successful worker
execution imports the package in its child process and creates the marker.

The registry measures all regular source files in the package, not just the
selected entrypoint file. A symlink anywhere in that measured package invalidates
resolution.

## Related references

- [Testing strategy](testing-strategy.md)
