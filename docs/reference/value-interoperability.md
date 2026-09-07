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

### Native integers and JSON encoding

Serialization makes a value transferable across a process or language boundary,
or persistable for later use; it is not needed merely to perform arithmetic in
Python. Decimal strings address a specific interoperability problem: ordinary
JavaScript JSON consumers can round large JSON numbers. A decimal string keeps
every digit until the consumer decodes it into an exact integer.

This is separate from a mathematical library's internal representation.
[SymPy uses exact `Integer` objects](https://docs.sympy.org/latest/modules/core.html#sympy.core.numbers.Integer),
not decimal strings for arithmetic.
[NumPy's usual integer types are fixed-width and can overflow](https://numpy.org/doc/stable/user/basics.types.html#overflow-errors);
its standard array persistence uses the
[binary `.npy` format](https://numpy.org/doc/stable/reference/generated/numpy.save.html).
Neither library requires mathematical values to be stored as JSON strings.

Each domain owns one mathematical value type with:

- Exact integers in Python.
- Canonical decimal strings when serialized to JSON.
- Explicit validation and decoding back to exact integers when reading JSON.

This keeps native arithmetic, comparisons, and tests numeric while preserving
lossless producer-consumer composition across JSON. It does not require
parallel native and wire mathematical value classes.

JSON safety does not require Python-facing fields to store strings. Use native
constructors and numeric accessors for computation; encode and decode only at
the JSON boundary.

Producers, consumers, validators, serializers, worker codecs, and schemas share
this boundary. When changing an exact-integer contract, preserve its canonical
JSON encoding and test both native and serialized composition.

An encoding migration changes the leaves of a canonical value, not the value's
mathematical shape. Do not replace an owned polynomial or map with a bare
coefficient tuple merely to expose its integers. For example, rational dynamics
accepts `RationalPolynomial`, and finite-field functional-graph operations accept
`FinitePolynomialMap`; those carriers retain their ring or field, variable, and
domain/codomain context. Migrate the integer fields inside those values, then
prove that a serialized producer result decodes directly as the consumer input.

Test arithmetic with native exact values wherever the native API accepts them.
Test serialization separately for canonical spelling, malformed inputs, and
lossless round trips, and retain producer-consumer composition tests across the
wire boundary. String encoding adds boundary tests; it should not require every
mathematical test to construct JSON or compare textual numbers.

#### Requirements for a native-integer codec

The shared codec uses Pydantic's
[separate Python and JSON validation](https://docs.pydantic.dev/latest/api/pydantic_core_schema/#pydantic_core.core_schema.json_or_python_schema)
and JSON-only serialization on the same annotated native type.

| Boundary | Exact-integer behavior |
| --- | --- |
| Native construction and Python validation | Accept exact integers; reject booleans, floats, and decimal strings rather than silently coercing them. |
| Python `model_dump()` | Retain integers for native consumers. |
| JSON validation | Accept canonical ASCII decimal strings, validate spelling and digit bounds before conversion, and decode to integers. Reject JSON numbers for these fields even at small magnitudes. |
| `model_dump(mode="json")` and `model_dump_json()` | Encode integers as canonical decimal strings at every magnitude. |

The encoding is selected for the field's complete admitted domain, not for the
magnitude of each value. If a field can contain integers outside JSON's safe
integer range, every instance uses the string form. Switching between a JSON
number for `2` and a string for a larger value would give one field two wire
types and make schemas and consumers branch on magnitude. For example:

```python
from jacobian.math.finite_fields import FiniteFieldPresentation

field = FiniteFieldPresentation(
    characteristic=2,
    modulus_coefficients=(1, 1, 0, 1),
    generator="a",
)
assert field.characteristic == 2
assert field.model_dump()["characteristic"] == 2
assert field.model_dump(mode="json")["characteristic"] == "2"
```

Accordingly, generated JSON Schema examples show strings even when their sample
values are small. They describe the wire contract, not Python construction or
the in-memory mathematical representation.

For this split, `model_validate()` on an already-decoded wire dictionary is not
equivalent to `model_validate_json()`: the former selects Python validation.
Use the owning wire entry point for decoded transport payloads. Jacobian's
`parse_operation_input()` selects JSON validation after strict JSON encoding;
native callers should not serialize merely to construct native values. Check
worker request and response codecs explicitly rather than assuming that the
MCP path covers them.

Publish request schemas with `mode="validation"` and result schemas with
`mode="serialization"`; Pydantic documents these as
[distinct schema modes](https://docs.pydantic.dev/latest/concepts/json_schema/#configuring-the-jsonschemamode).
For exact-integer fields, both describe strings even though Python
fields hold integers. The generated schema and runtime validator must agree on
spelling and digit limits. For example, a three-digit envelope accepts `"999"`
and `"-999"`, but rejects `"9999"`. A four-character maximum alone does not
express that rule: the extra character is allowed only for the minus sign.

Keep three limits distinct: the interoperable JSON-number range, the structural
cost of parsing and formatting a decimal value, and the operation's admitted
arithmetic work. Python's
[decimal-conversion guard](https://docs.python.org/3/library/stdtypes.html#integer-string-conversion-length-limitation)
protects conversion cost; it is not an integer arithmetic ceiling. Using a
conversion backend does not remove the need for bounded input and output.

Before declaring a migration complete, test native and JSON round trips,
nested and empty values, producer-consumer and worker composition, and schema
agreement on accepted and rejected examples. Include values beyond the JSON
safe-integer range and Python's configured decimal-conversion guard, positive
and negative digit-boundary cases, and rejection of booleans, floats, leading
zeros, plus signs, negative zero, whitespace, exponent notation, and non-ASCII
digits. A standalone scalar round trip is not evidence of complete migration.

### Exact integers: representation is not a work limit

For native integer fields, use `Annotated[int, DecimalIntegerEncoding(max_digits=...)]`
from `jacobian._exact`, with the owning structural digit envelope. This is codec
metadata, not a second mathematical value class. It validates native integers
and canonical JSON strings separately; both JSON schema modes publish the
string encoding. Python `model_dump()` remains numeric, while
`model_dump(mode="json")` and `model_dump_json()` serialize strings. Read wire
values with `model_validate_json()` (the dispatcher uses this path), not
`model_validate()` on an already JSON-decoded dictionary.

Use the shared `ExactInteger` annotation when the domain uses the common
32,768-digit envelope. Add `DecimalIntegerEncoding` directly only for a
different domain-owned envelope; do not introduce synonymous integer wrappers.

The same boundary applies to shared integer matrices, polynomial coefficients,
and the integer components of `CanonicalRational`. A rational remains one
compound value: `CanonicalRational(num=3, den=7)` in Python serializes as
`{"num":"3","den":"7"}` in JSON. Its denominator must be positive, its
components reduced, and zero represented as `0/1`; decoding validates these
invariants rather than silently normalizing an authored representation.

Mathematical integer fields whose domain or results can exceed the interoperable
JSON-number range, `[-(2**53 - 1), 2**53 - 1]`, must encode consistently as decimal
strings, including small values: `"0"`, `"42"`, `"1000000016000000063"`.
Canonical spelling has no leading plus, leading zeros, exponent notation,
whitespace, or negative zero. Do not add a second big-integer wrapper or switch
between JSON numbers and strings according to magnitude.

The string is the **wire encoding of an exact integer**, not a symbolic
expression and not an instruction to compute with text. Use
`parse_canonical_integer` and `format_canonical_integer` from
`jacobian.canonical` at arithmetic/transport boundaries; these helpers also
handle values beyond Python's ordinary decimal-conversion digit limit.
Compute with exact Python or backend integers. Never pass through a float or
JavaScript `Number` to parse or serialize them. Keep intrinsically bounded
indices, dimensions, and counters as JSON integers when their complete domain
fits that range.

There are three separate questions:

| Limit | What it protects | Where it belongs |
| --- | --- | --- |
| JSON safe-integer range | Exact exchange with ordinary JSON consumers | Scalar field encoding; if the field can exceed the range, use canonical strings for every value. |
| Input bytes, decimal digits, collection size | Bounded parsing and materialization | Documented transport or structural guards, before expensive conversion/allocation. |
| Arithmetic work, intermediate growth, memory, output size | A bounded exact computation | Admission for the actual operation and algorithm. |

Arbitrary-precision encoding does not promise unlimited computation. Conversely,
a transport number limit must not become a mathematical limit merely because a
field was originally encoded as a JSON number. Keep the strict JSON encoder's
safety check; fix the mathematical value that crosses it.

An integer-range repair is a shared-value migration:

1. Freeze an exact value that fails today, including a result that grows beyond
   its individually small input scalars.
2. Change the domain-owned scalar fields, then migrate every producer, consumer,
   native entry point, worker codec, catalog example, and generated schema that
   uses them. Preserve parents and ordered axes, including empty values.
3. Update arithmetic conversions and exact output bounds. Do not leave a
   narrowing conversion to a machine integer or float in an intermediate path.
4. Test strict-JSON round trips and serialized producer-consumer composition
   beyond `2**53 - 1`, malformed encodings, and retained resource rejections.
   Encoding a large value is not evidence that an operation admits it.

For the separate computation repair, follow the
[execution-envelope review](public-operation-admission.md#execution-envelope-review).

### Shared schemas and explicit conversions

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
