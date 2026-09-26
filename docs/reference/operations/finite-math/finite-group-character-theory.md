# Finite-group character operations

[Finite mathematics operations](index.md) · [Tool surface](../../tools.md)

## Decomposing an S3 tensor product

`finite_group.character.tensor_product.decompose.compute` takes two row
indices and a concrete complete class partition. It recomputes the canonical
character table from that partition, so callers cannot supply an unauthenticated
claim that arbitrary rows are a complete irreducible basis. This release is
limited to S3. The result contains the pointwise tensor-product class function
and its exact multiplicities, in table row order, computed by Hermitian inner
products against every canonical irreducible row.

For S3, the standard character `(2, 0, -1)` has tensor square `(4, 0, 1)`;
its multiplicities in the trivial, sign, and standard rows are `(1, 1, 1)`.
The operation preflights predicted product coefficient heights, every basis
inner product, aggregate work, and output size before product or pairing
arithmetic. It then checks the computed multiplicities are nonnegative
integers.

## Character ring coordinates

`finite_group.class_function.character_ring_decompose.compute` expresses a
class function in the canonical irreducible basis and returns a
`CharacterRingElement`. That value retains the complete character table and
one exact integer coordinate per irreducible row, so subsequent operations
can compose in the representation ring without carrying a loose coefficient
list. Signed coordinates represent virtual characters; nonnegative
coordinates represent ordinary characters. The operation rejects a class
function whose exact Hermitian pairings with the irreducibles are not integers.

The character basis is reconstructed from the concrete source group before
use. This release supports only the trivial group, finite cyclic groups up to
order 60, and `S3`. It does not trust a caller-supplied character table or claim
to decompose a class function on an unsupported group. The exact coordinate
formula is the standard Hermitian pairing
`<f, chi> = (1/|G|) sum_C |C| f(C) conjugate(chi(C))`; irreducible characters
form an orthonormal basis of the complex class functions. GAP's reference
manual documents the scalar product and distinguishes ordinary from virtual
characters in its [class-function chapter](https://gap-system.github.io/gap/doc/ref/chap72_mj.html).

Rational-valued input functions are included exactly into the table's
cyclotomic field by sending `q` to the constant power-basis coefficient `q`.
Inputs in any other field must already use the table's declared field; no
nontrivial field embedding is inferred.

Before conjugacy expansion, admission caps the concrete group order and bounds
the worst-case complete table, all basis-pairing work, input coefficient
growth, and serialized result size. The returned model checks coordinate axis
and shape during deserialization; consumers that rely on table mathematics
must reconstruct the canonical table from its group.

## Tensoring virtual characters

`character.tensor_product.compute` multiplies two `CharacterRingElement`
values in the representation ring and returns their exact irreducible
multiplicities. It requires both values to retain the same canonical table for
the same group. The operation expands their coordinates into class values,
uses the pointwise product (the character of the tensor product), and computes
the exact Hermitian inner product with every irreducible row. It checks that
the resulting coordinates reconstruct the product exactly. Signed input
coordinates remain valid and can produce signed output coordinates.

The supported canonical tables are for the trivial group, cyclic groups of
order at most 60, and `S3`. This is a bounded supported-family contract, not a
claim about arbitrary finite groups. Symmetric and exterior powers and Adams
operations are separate operations. The second symmetric and exterior powers
use the exact identities `Sym^2(x) = (x tensor x + psi^2(x))/2` and
`Lambda^2(x) = (x tensor x - psi^2(x))/2`; `psi^2` evaluates the character at
the square of each class representative. They accept table-bound virtual
characters, rebuild the canonical table, derive the squaring map from its
complete element partition, and return exact integral irreducible coordinates
only after exact reconstruction. Admission includes tensor arithmetic,
class-map construction, pairings, and the output bound before class expansion.
Higher symmetric/exterior powers and general Adams operations remain open.
GAP describes
class-function multiplication and scalar products in its
[class-function reference](https://gap-system.github.io/gap/doc/ref/chap72_mj.html).
The source permutation presentation has its own work bound before group-order
computation. The operation derives the order from the source permutations and
stops group-closure enumeration after finding a 61st element; a forged table
partition cannot reduce this bound.

## Character kernels

`character.kernel.compute` accepts a `CharacterRingElement` with nonnegative
coordinates in its canonical table. It returns a `CharacterKernel` retaining
both the ambient permutation group and the subgroup generated by the exact
kernel elements. The operation rebuilds the trivial/cyclic/S3 table and
selects the complete conjugacy classes where `chi(g) = chi(1)`. For a finite
group representation over characteristic zero, this equality holds exactly
when every eigenvalue of `rho(g)` is 1, so the selected classes form the
representation's kernel. This is the class criterion documented by GAP's
[`KernelOfCharacter`](https://gap-system.github.io/gap/doc/ref/chap72_mj.html#X7EA0161782BF529C).

Virtual characters with negative coordinates are rejected because they do not
define an ordinary representation whose kernel this operation can return.
Admission caps the concrete group order at 60 and jointly bounds class-value
arithmetic, table validation, subgroup generation, and serialized parent plus
kernel before conjugacy expansion.

## Scaling a class function

`class_function.scale.compute` multiplies every value by one exact scalar in
the cyclotomic field declared by the function's class axis. It preserves the
axis and returns a `FiniteClassFunction`; it does not assert that the result is
a character. Cyclotomic product work, coefficient height, and output size are
admitted using the same exact multiplication bounds as pointwise products.
This is the usual scalar action on the vector space of class functions.

For the standard S3 character `(2, 0, -1)`, scaling by `-2` gives
`(-4, 0, 2)` on the identical class axis. A scalar from a different
cyclotomic field is rejected rather than implicitly transported.

## Restricting a class function

`finite_group.class_function.restrict_to_subgroup.compute` restricts one exact
class function on a concrete finite permutation group to a specified subgroup.
The class-function axis retains the source group, its canonical class sizes,
and canonical class representatives. The subgroup is a permutation group on
the same domain, and each of its generators must belong to the source group.
These checks establish the inclusion without inferring a subgroup from its
order.

The operation returns the subgroup's complete canonical conjugacy partition,
the map from each subgroup class to its containing source class, and the exact
restricted class function on the subgroup axis. The class map is well-defined:
conjugation by a subgroup element is also conjugation in the source group.
The source axis is checked against the source group's canonical conjugacy
partition before its values are used.

Admission currently allows source order at most 256 and subgroup order at most
128, along with bounded permutation degree, class count, exact coefficient
size, partition work, and serialized output. The smaller target order cap
ensures its complete class axis fits the shared class-function carrier. The
operation acts on class functions; when its input is an irreducible character,
the same output is its character restriction.

The S3 example restricts the standard degree-two character to a transposition
subgroup. The explicit class map is `(identity, transposition) ->
(identity, transposition)`, and the restricted values `(2, 0)` decompose as the
sum of the two irreducible characters of C2.

## Inducing a class function

`finite_group.class_function.induce.compute` computes the exact induced
class function for a class function on a concrete subgroup `H` of a parent
permutation group `G`. The groups use the same permutation domain, and every
subgroup generator must belong to `G`. For each parent-class representative
`g`, the value is

`(1 / |H|) * sum(phi(x^-1 * g * x) for x in G if x^-1 * g * x is in H)`.

The result retains the subgroup and parent conjugacy partitions, the map from
subgroup classes to parent classes, and the induced values on the parent
class axis. It uses the same cyclotomic field as the source class function.
Admission bounds subgroup order by 128, parent order by 256, class count,
exact coefficient growth, arithmetic work, and serialized output before
conjugacy expansion. The serialized result is bounded by the canonical 10 MB
output limit.

Inducing the trivial class function from a transposition subgroup `C2` to `S3`
gives values `(3, 1, 0)` on `(identity, transposition, 3-cycle)`. Their inner
products with the S3 irreducible characters `(1, 1, 1)`, `(1, -1, 1)`, and
`(2, 0, -1)` are `(1, 0, 1)`, independently confirming the expected
decomposition into the trivial and standard representations.

Deserialization checks the restriction and induction result shapes, axis
parents, and class-index ranges. It does not recompute either class map or
authenticate the relation between a map and the carried function values. It
also does not reauthenticate a tensor result's character table or recompute its
pointwise product and multiplicities. A consumer relying on any serialized
relation must check it: for restriction, compare each target value with its
mapped source value; for induction, check each mapped subgroup class lies in the
stated parent class and the induced values satisfy the class-sum formula; for
tensor decomposition, reauthenticate the canonical table and check the
pointwise product and inner-product multiplicities. The operation constructs
these relations from admitted canonical class partitions.
