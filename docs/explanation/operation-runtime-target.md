# Operation runtime target

[Documentation home](../index.md)

- Status: Accepted target for the next breaking release
- Scope: planned operation declarations, compiled catalogs, and selected-operation execution

This document records the accepted target architecture. It is not a claim that
the current pre-stable release already implements every lifecycle described
below. The current supported behavior remains defined by the [product model](product-blueprint.md),
[architecture](architecture.md), and the reference contracts.

## Vocabulary and wire cutover

The public vocabulary is `operation` / `math operation`. The existing
`math.find` and `math.run` tool names remain. In the next breaking release,
`capability_id` becomes `operation_id`, `capability_version` becomes
`operation_version`, and the complete inventory moves from
`capability://catalog` to `operation://catalog`. There is no compatibility URI,
field alias, or parallel contract. Operation ID values themselves do not
change.

The target contracts are `OperationRequest`, `OperationResult`,
`OperationDescriptor`, `OperationDiagnostic`, and `OperationCatalogSnapshot`.
Checker operations remain ordinary catalog entries with distinct operation
IDs; a producer is never switched into a checker role.

## Pure declarations and the compiled catalog

An `OperationDeclaration` is immutable data describing one operation: its ID,
version, title, description, tags, typed request and result models, binding,
provider requirement, publication policy, ports, examples, preflight,
postcondition, and effect. Bindings are one of `DirectKernel`,
`BoundedWorker`, or `AuthorizedChecker`.

Declarations contain no stores, registered schema URIs, service objects, live
providers, authorization records, or installer callbacks. A `DomainBundle` is
likewise a passive declaration containing domain identity, semantics, and its
operations. Built-in bundles are explicitly listed; third-party plugin
discovery is not part of the target.

`jacobian init` and `jacobian update` compile the declarations into one
revisioned catalog. The persisted catalog has an active revision, compact
search cards, exact descriptors, declaration locators and digests, provider
inventory identity, and checker-binding identity. Search reads cards; exact
inspection reads one descriptor. The full `operation://catalog` resource is
materialized only when explicitly requested. Visibility filtering is applied
to search, inspection, execution, and the resource without rebuilding the
compiled snapshot.

The catalog is inert searchable data. A locator such as a bundle module plus
operation ID is resolved by `OperationRegistry` only after an operation is
selected. The registry then verifies the loaded declaration against the
persisted identity, schemas, and digest.

## Providers, execution, and publication

Declarations name a `ProviderRequirement`; they do not load a backend.
`init/update` record `ProviderObservation` values containing exact version,
digest, platform, availability, and diagnostics. A small explicit mapping of
built-in provider definitions resolves only the selected requirement. There is
no entry-point discovery or dependency-injection container.

The intended execution path is:

```text
OperationRequest
  → prepare_operation_request
  → resolve selected declaration/provider/binding
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
`update` migrates existing state, refreshes provider observations, authorizes
bundled checkers, compiles the catalog, and atomically selects the new
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
stores and execution leases. Tenant-specific checker authorizations are not
copied.

## Migration and non-goals

The cutover is the next pre-1.0 breaking minor release (planned as `0.13.0`)
and uses the next state revision. Revision-11 artifacts remain readable during
the update, while checker identity is remeasured rather than trusted from
copied records. There are no compatibility aliases for the public wire
contract.

The target is not a workflow engine, automatic planner, plugin-discovery
framework, generic dependency-injection system, universal backend wrapper, or
third semantic type system. Agents still own decomposition, sequencing,
checker choice, and stopping. Catalog search remains factual and does not
recommend a proof strategy or hidden next step.

The lifecycle principle is intentionally similar to Code Mode and Executor:
catalog entries are cheap, inert descriptions, while implementation and
provider loading happen only after selection. This keeps startup catalog-only,
discovery independent of execution services, and invocation proportional to
the operation the agent actually chose.
