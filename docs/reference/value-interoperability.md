# Schemas, mathematical values, and conversions

## Default: return the value

An exact operation returns its mathematical answer and the context needed to
interpret it. A determinant returns an exact scalar. A polynomial product
returns a polynomial with its coefficient ring and ordered variables.

**Do not add certificates, source digests, `verified` flags, or generic assurance
wrappers by default.** MCP does not require them. Correctness comes from the
admitted kernel and its mathematical tests, not from extra result metadata.

Use these rules when designing or repairing an operation:

1. **Reuse the value type.** The same mathematical value and interpretation use
   the same domain-owned type. Preserve ring, dimensions, axes, and relevant
   normalization, including empty cases. Intentional storage or basis variants
   need explicit semantics; they do not justify operation-specific duplicates.
2. **Keep parsing structural.** Models check encoding, shape, and cheap intrinsic
   consistency. The admitted computation establishes mathematical properties.
3. **Make transformations explicit.** A change of ring, field presentation,
   basis, or axes needs its specified map. A duplicate schema needs a shared
   type, not an adapter to another duplicate.
4. **Include witnesses for a mathematical purpose.** Add the actual witness
   when the operation requests it, needs it for downstream construction, or
   provides a distinct independent-checking result. A certificate label is not
   a reason to add a witness or another operation.
5. **Check only claims the consumer relies on.** Receiving a polynomial does
   not require proving where it came from. If the consumer relies on “this is
   the characteristic polynomial of A,” it must establish that additional
   relation under its own admission.

## Values, witnesses, and source binding

| Need | Appropriate result |
| --- | --- |
| Determinant or polynomial product | The exact value; no proof wrapper. |
| Smith diagonal | The canonical Smith value. |
| Recover integer multipliers using Smith reduction | The same Smith value plus the actual unimodular transformations and their source relation. |
| Identify an exact algebraic root | Its polynomial and root-selection information. An isolating interval identifies the value; it is not generic assurance. |
| Independently check infeasibility | The multiplier witness and source system, with admitted verification when a consumer uses an authored witness. |

Distinguish three things:

- **Context** tells the caller what a value means: its parent, basis, axes, or
  selected root. It must remain available after serialization.
- **Source binding** states which inputs a conclusion or witness concerns.
  A digest can detect a stale binding, but anyone can recompute it. Add one only
  when that identity mechanism is needed; retaining the source may suffice.
- **A mathematical witness** supplies data satisfying a useful relation, such
  as `D = U A V`. Its producer establishes that relation; the witness's name or
  digest does not prove it to a later consumer.

Serialization preserves data, not trusted provenance. This does **not** mean
that every downstream operation must verify every upstream computation. A
consumer treats an ordinary value as its input. It checks an additional claim
only if its own mathematical conclusion depends on that claim. Do not build a
universal certificate layer or a recursive proof-history checker.

## Put each check at its owning boundary

| Boundary | Responsibility |
| --- | --- |
| Schema and model parsing | Scalar encoding, fields, shape, ordering, and cheap intrinsic consistency. |
| Native admission and computation | Applicability, work and growth bounds, and mathematical checks required by the operation. |
| Trusted result construction | Package the established result without repeating its computation. |
| SDK output validation and serialization | Check and encode the result's structure. No mathematical backend calls. |
| Owning tests | Independently check defining identities, reconstruction, and producer-consumer composition. |

For example, checking that a polynomial has one variable and leading
coefficient one is structural. Proving irreducibility or independence of a
proposed basis is mathematical work. A class name such as “field,” “subspace,”
or “certified matrix” does not move that work into parsing.

**Moving a check out of a validator is a migration, not deletion.** Identify
all public native and MCP consumers that rely on the property, put the admitted
check there, and test false authored claims. Reuse facts within one admitted
execution instead of checking the same field or basis for every entry. Do not
leave a consumer trusting the property that its former constructor checked.

## Schema and conversion rules

Generate public schemas from the actual domain-owned request and result types.
Required fields, defaults, literals, and result branches must agree with public
parsing. Explain constraints that JSON Schema cannot express in descriptions
and executable examples. Do not force a field to be required in the schema
while silently defaulting it during wire parsing.

Requests contain canonical values plus operation parameters. Results return
those values without redefining them as parallel fields or backend strings.
Operation-specific work and degree limits belong in admission, not in a
second, narrower value class.

A consumer that also accepts an input presentation must keep the canonical
value branch in its schema. For example, simplicial structural operations
accept both vertices with facets and an unchanged canonical complex returned by
canonicalization, deletion, or subdivision. They may normalize the presentation
internally; MCP callers need no Python-only projection step. Native consumers
that rely on an authored canonical face ledger admit its closure before use.

| Representation question | Decision |
| --- | --- |
| Characteristic and minimal polynomials use different coefficient layouts | Share the rational-polynomial encoding. Keep coefficient-list views as native helpers. |
| A `0 × n` integer matrix loses its width | Repair the common matrix type to retain dimensions. Do not introduce a certificate-only matrix type. |
| Sparse storage avoids a large dense expansion | Keep the intentional representation and admit expansion before allocating it. Prefer consumers that exploit the sparse form. |
| Bernstein coefficients use a different basis | Retain the box, axis order, degree, and basis interpretation; test reconstruction. |
| A map changes a coefficient field or ordered axes | Require the explicit map and its applicability conditions. Never infer an embedding or silently reorder coordinates. |
| A conversion merely copies or extracts existing fields | Use a native helper or property; do not automatically publish another tool. |

Native conversions accept canonical values and genuine map parameters, not
private wire request objects. Any MCP projection calls the same admitted native
path. Backend conversion remains private. A public conversion must separately
pass the [operation admission gates](public-operation-admission.md#admission-gates).
In this pre-stable repository, update affected producers and consumers together
instead of adding aliases or a network of adapters around defective schemas.

## Repair the contract, not each call site

Use the observed failure to identify the owning change:

| Failure | Repair |
| --- | --- |
| Boolean multilinear extension returns a printed backend expression | Return the domain-owned polynomial with its coefficient ring and ordered variables. A display string is optional presentation, not the composable value. |
| An algebra center or empty basis loses its ambient algebra | Retain the parent and the inclusion or basis coordinates needed to interpret the result. Do not infer the parent from nonempty entries. |
| Plain and witness-producing Smith operations encode the same diagonal differently | Return the same canonical value in both; transformations are additional mathematical data. |
| Factorization output cannot feed a root operation | Share the polynomial carrier; preserve factor multiplicities in the factorization result. The consumer still admits its own degree and work. |
| Two declarations for the same Walsh computation admit different variable counts | Establish one semantic contract and admission path. A different algorithm or discovery phrase does not justify a second contract. |
| A sparse input is rejected using the cost of dense expansion | Admit the algorithm actually used, including intermediates and exact output; preserve a useful sparse boundary regression. |

Shared representation guarantees that values cross the typed boundary unchanged;
it does not guarantee that every consumer can afford every representable input.
Explain operation limits separately from carrier limits. Do not promise a larger
accepted domain until its motivating request executes successfully.

## State precisely what a relation establishes

A witness is useful only with a named relation and sufficient source context.
For example, `H = U A` alone does not establish Hermite normal form: the contract
also needs the relevant unimodularity and normal-form conditions. Likewise, a
valid graph matching need not be maximum, and an integer matrix of proposed
lattice coordinates proves inclusion only when it reconstructs the claimed
basis. Tests must cover the full advertised postcondition, not merely the
cheapest identity.

This does not require every producer to run a second solver or attach a proof.
Its algorithm establishes the postcondition; independent tests check it. A
consumer accepting an authored witness checks the particular relation and
properties on which its own answer depends, under its own admission. Context,
a source digest, and successful structural parsing cannot replace those checks.

## MCP Python SDK v2

The repository pins `mcp==2.1.0`. The official
[v2 documentation](https://py.sdk.modelcontextprotocol.io/v2/) and
[structured-output guide](https://py.sdk.modelcontextprotocol.io/v2/servers/structured-output)
describe typed data exchange and output validation. **Neither requires
mathematical certificates nor establishes mathematical truth.** SDK release
numbers and date-based MCP protocol revisions are distinct.

Use the typed result as structured content with a matching generated schema.
Installed SDK 2.1.0's `FuncMetadata.convert_result` validates non-error
`CallToolResult.structured_content` when an `output_model` is supplied.
Jacobian's direct-tool adapter supplies one. Returning a `CallToolResult`
therefore does not bypass model validation on this path. Keep those validators
structural; do not disable output validation to conceal repeated kernel work.
Test the actual adapter path rather than assuming every SDK API behaves alike.

The [tools specification](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
distinguishes successful results from execution errors. A mathematical answer
of `false`, infeasible, or not-applicable is a successful computation when it
belongs to the operation's codomain. A timeout or backend failure establishes
no such answer.

## Required evidence for a repair

Test the relevant boundary rather than introducing a generic verification
framework:

- Pass a producer's serialized subvalue directly to its consumer without field
  renaming or reconstructed context. Include a relevant empty or degenerate case.
- Prove a conversion's defining identity independently; test round trips only
  for invertible maps and rejection for incompatible parents.
- Check schema/parser agreement and native/MCP parity where they are affected.
- Show that false authored claims are checked by the consumer that relies on
  them, while parsing and output construction do not repeat mathematical work.

Catalog examples establish first-call usability. They do not replace these
mathematical and composition tests.
