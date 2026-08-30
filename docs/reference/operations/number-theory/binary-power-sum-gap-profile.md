# Real-embedded binary power-sum gap profiles

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

`number_field.real_embedding.binary_power_sum_gap_profile.compute` returns the
complete finite family

```text
sum(epsilon_i * q^i, i=0,...,m-1),  epsilon_i in {0,1},
```

for one exact simple-number-field element `q` and one selected real embedding.
It is the finite carrier used in [Erdős Problem #1096](https://www.erdosproblems.com/1096),
beta-expansion collision experiments, Bernoulli convolutions, and algebraic
separation questions. The operation makes no claim about an infinite sequence
of least gaps or its limit.

## Input and recognized binding

The request contains a `SimpleNumberFieldRealEmbeddingBinding` and a
nonnegative `exponent_count` `m`. The binding retains:

- the presented quotient `QQ(alpha) = QQ[x]/(f)`;
- the reduced power-basis coordinates of `q`; and
- one complete real-embedding record produced by
  `number_field.embeddings.compute`.

That carrier is intentionally structural. Parsing checks that the element and
record name the same presentation; it does not recognize irreducibility, root
identity, or isolation evidence. Before arithmetic or order, the operation
reruns the complete bounded embedding producer and requires the supplied record
to equal one returned real record exactly. It then requires the selected image
to satisfy the first-release slice `1 < sigma(q) < 2`.

## Exact equality, order, and gaps

A representation is the bit vector
`(epsilon_0,...,epsilon_(m-1))`, in increasing exponent order. The recurrence

```text
P_0 = {0: [()]}
P_(i+1) = P_i union {s + q^i: s in P_i}
```

retains every vector. SymPy 1.14 supplies exact `QQ(alpha)` arithmetic through
its `AlgebraicField`/`ANP` implementation, constructed only from canonical
integer and rational objects. No caller string reaches a parser or evaluator.

Collision buckets use equality in `QQ[x]/(f)`, independently of the selected
embedding. Only distinct reduced values are sorted. One request-scoped order
evaluator starts from the recognized root's exact rational interval and applies
a rational midpoint/derivative enclosure to each power-basis difference. If an
interval cannot decide the sign, it refines the same selected root; a remaining
ambiguous comparison falls back to the admitted exact minimal polynomial and
real-root isolation of that difference. Floating approximations are never used.

Each adjacent gap stores the exact reduced difference and a closed rational
enclosure lying strictly above zero. `least_gap` and `largest_gap` retain exact
field elements and identify their first matching gap indices. Equal gap values
therefore have a deterministic summary representative.

For `m=0`, the only bucket is zero with representation `()`. There are no gaps,
and the least/largest gap values and indices are absent. For every other result:

- the buckets partition all `2^m` bit vectors exactly once;
- each bucket's representations are lexicographically sorted;
- distinct values appear in strictly increasing selected-real order;
- each gap reconstructs as `value[j+1] - value[j]` and has positive exact
  enclosure evidence; and
- multiplicities sum to `2^m`.

## Execution envelope

Admission occurs before embedding recognition or enumeration. It separately
bounds:

- at most 4,096 retained source representations and 49,152 bit-vector slots;
- at most 16,384 exact field additions/multiplications;
- at most 100,000 exact selected-embedding comparisons;
- every power, sum, and difference coordinate within the canonical 256-digit
  numerator/denominator carrier;
- selected-image elimination, root refinement, and enclosure components through
  the shared exact real-embedding comparison envelope; and
- the complete retained result, including worst-case distinct buckets, every
  source bit vector, every gap, and summary values, within 10,485,760 bytes.

The coordinate proof writes `q=H(alpha)/D`, bounds `||H^i||_1`, and accounts
for every recursive reduction by the nonmonic defining relation after clearing
the leading coefficient. All power denominators divide a common denominator at
the largest exponent, so the subset-sum bound adds scaled numerators rather than
multiplying unrelated denominators. The comparison proof clears every bounded
coordinate denominator, bounds the Sylvester resultant
`Res_x(f(x), D*y-H(x))`, and applies a Landau--Mignotte factor bound before
minimal-polynomial or root-isolation work.

The recurrence, comparisons, result construction, and final serialization check
share a ten-minute owner envelope. The embedding producer retains its stricter
two-minute disposable-worker subdeadline, bounded by any earlier caller
deadline; after recognition, the outer operation resumes its original envelope.
Cancellation and expiry remain execution failures, never mathematical
conclusions or partial profiles.

## Scope

This operation implements #2892 and depends on the exact embedding profile from
#1689 plus selected-real-embedding element order. It does not add a Pisot
recognizer, beta-expansion normal form, arbitrary algebraic subset-sum API,
numerical sampler, or decision procedure for the global statement in Erdős
#1096.
