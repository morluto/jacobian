# Compiled operation architecture

[Documentation home](../index.md)

- Status: Current architecture
- Scope: operation declarations, compiled catalogs, and selected-operation execution

This document records the catalog and selected-execution architecture behind
Jacobian's two-tool MCP surface. The [product model](product-blueprint.md),
[architecture](architecture.md), and reference contracts define its supported
behavior.

## Vocabulary and wire cutover

The public vocabulary is `operation` / `math operation`. The existing
`math.find` and `math.run` tool names remain. In this breaking release,
`capability_id` becomes `operation_id`, `capability_version` becomes
`operation_version`, and the complete inventory moves from
`capability://catalog` to `operation://catalog`. There is no compatibility URI,
field alias, or parallel contract. Operation ID values themselves do not
change.

The public contracts are `OperationRequest`, `OperationResult`,
`OperationDescriptor`, `OperationDiagnostic`, and `OperationCatalogSnapshot`.
Checker operations remain ordinary catalog entries with distinct operation
IDs; a producer is never switched into a checker role.

## Pure declarations and the compiled catalog

An `OperationDeclaration` is immutable data describing one operation: its ID,
version, title, description, tags, typed request and result models, execution,
publication policy, ports, examples, preflight, postcondition, and effect.
Ordinary operations call a typed mathematical kernel directly. Exceptional
process and checker operations may name a bounded worker or an authorized
checker identity.

Declarations contain no stores, registered schema URIs, service objects, live
providers, authorization records, or installer callbacks. Each mathematical
domain exposes an immutable tuple of declarations from a pure declaration
module. A fixed source-controlled module index lists the built-ins;
third-party plugin discovery and a domain-bundle framework are not part of the
architecture.

`jacobian init` and `jacobian update` compile the declarations into one
revisioned catalog. The persisted catalog has an active revision, compact
search cards, exact descriptors, declaration locators and digests, and
checker-binding identity. Search reads cards; exact
inspection reads one descriptor. The full `operation://catalog` resource is
materialized only when explicitly requested. Visibility filtering is applied
to search, inspection, execution, and the resource without rebuilding the
compiled snapshot.

The catalog is inert searchable data. Each entry carries either a declaration
module locator or an explicit `family:<name>` binding origin. The
runtime-local `OperationRegistry` reads that locator only after an operation is
selected, validates the loaded declaration or family adapter against the
persisted identity, schemas, and digest, and caches the result. The fixed
family table is assembled by the runtime; graph, polynomial, Lean, and SAT/SMT
modules own their selected IDs and binding logic. The runtime owns any
closeable resources acquired by those binders and releases them at shutdown.

## Mathematical backends, execution, and publication

Ordinary operations do not acquire operation-specific measured runtimes. Their
maintained math-library dependencies are part of the Jacobian installation and
are imported only by the selected declaration or its private backend module.
Exact external executables and authorized checkers retain the identity and
readiness observations required for reproducibility and fail-closed trust.
There is no provider framework, entry-point discovery, or dependency-injection
container.

The execution path is:

```text
OperationRequest
  → prepare_operation_request
  → resolve selected declaration
  → execute_operation
  → publish_operation_result
  → OperationResult
```

Preparation binds typed value references, enforces canonical byte limits, and
strictly validates the concrete Pydantic request model. Execution returns a
typed completed value, non-conclusion, or failure. It performs preflight,
binding execution, result validation, and postconditions; it does not construct
wire or persistence envelopes. Publication alone owns inline values, value
references, durable artifacts, previews, and artifact URIs. The final wire
projection constructs `OperationResult` once.

Ordinary mathematical operations delegate to `jacobian.math` when the same
function belongs in the native API. Maintained-library imports stay in private
domain backends or bounded-worker entrypoints. MCP-only process and
publication operations need not be added to the native API.

## Startup, update, and checker authorization

`init` creates a current state or reports that state is already current.
`update` migrates existing state, refreshes exceptional external-runtime
observations, authorizes bundled checkers, compiles the catalog, and atomically selects the new
revision. Checker authorization is selected with
`--checker-authorization bundled|none` on those operator commands. Serving
does not migrate state, compile a catalog, or reconstruct checker manifests.

Checker authorization measures shared source, dependency, and executable
identity once during `init/update`, then persists operation-ID-to-checker-ID
bindings. A selected checker still enters a bounded worker and remeasures its
exact executable before and after execution. Missing, revoked, changed,
malformed, timed-out, or cancelled checkers fail closed.

A serving process requires the current state format, an active catalog
snapshot, and a matching package/catalog version. Missing or stale state is a
stable `STATE_INITIALIZATION_REQUIRED` or `STATE_UPDATE_REQUIRED` diagnostic
with the exact `jacobian init` or `jacobian update` command. Serving never
repairs state automatically; a successful update requires a restart.

One installation owns one catalog and checker-authorization index. Remote
tenants share those mathematical definitions while retaining isolated artifact
stores. Host-private request ownership keeps an active tenant runtime from
being evicted while its work is still running; this is not an agent-visible
execution-lease contract. Tenant-specific checker authorizations are not
copied.

## Migration and non-goals

Revision 12 is the current operation-catalog cutover. `jacobian update`
migrates supported revision-11 stores, retires the superseded generic runtime
tables, and selects the new catalog. Checker identity is remeasured rather
than trusted from copied records. There are no compatibility aliases for the
public wire contract.

The target is not a workflow engine, automatic planner, plugin-discovery
framework, generic dependency-injection system, universal backend wrapper, or
third semantic type system. Agents still own decomposition, sequencing,
checker choice, and stopping. Catalog search remains factual and does not
recommend a proof strategy or hidden next step.

The lifecycle principle is intentionally similar to Code Mode and Executor:
catalog entries are cheap, inert descriptions, while implementation and any
exceptional external runtime loading happen only after selection. This keeps startup catalog-only,
discovery independent of execution services, and invocation proportional to
the operation the agent actually chose.
