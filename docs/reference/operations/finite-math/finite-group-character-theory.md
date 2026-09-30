# Finite-group character operations

[Documentation home](../../../index.md) · [Tool surface](../../tools.md) · [Operation references](../index.md) · [This domain](index.md)

[Finite mathematics operations](index.md) · [Tool surface](../../tools.md)

## Bounded complete character tables

`finite_group.character_table.compute` returns a complete irreducible table for
the trivial group, supported cyclic groups, S3, and a concrete nonabelian
permutation group of order eight. For the last case, the complete canonical
conjugacy partition must have class sizes `1, 1, 2, 2, 2`; this is the shared
profile of D8 and Q8. The result retains the input group and exact class axis,
uses rational character values, and includes four linear rows plus one
degree-two row.

The operation derives the linear rows from group multiplication. A pair of
noncommuting elements generates a nonabelian order-eight group, so the four
choices of signs on those generators exhaust its homomorphisms to `{±1}`. The
degree-two row takes values `2` on the identity, `-2` on the other central
element, and `0` on the three noncentral classes. Exact row orthogonality and
the degree-square sum are checked before publication. GAP's character table
reference gives the same table for D8 and Q8, with their class representative
orders distinguishing the groups: [GAP Character Table Library](https://docs.gap-system.org/pkg/ctbllib/doc2/manual.pdf).

This order-eight extension does not claim tables for general nonabelian groups
or for noncyclic abelian groups such as `C2 × C2 × C2`.

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
size, partition work, and the output's class-cell count and coefficient digit
width. The smaller target order cap
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
exact coefficient growth, arithmetic work, and the exact output's retained
cells and digit widths before conjugacy expansion. The canonical transport
boundary separately bounds the serialized result at 10 MB.

Inducing the trivial class function from a transposition subgroup `C2` to `S3`
gives values `(3, 1, 0)` on `(identity, transposition, 3-cycle)`. Their inner
products with the S3 irreducible characters `(1, 1, 1)`, `(1, -1, 1)`, and
`(2, 0, -1)` are `(1, 0, 1)`, independently confirming the expected
decomposition into the trivial and standard representations.

Deserialization checks the restriction and induction result shapes, axis
parents, and class-index ranges. It does not recompute either class map or
authenticate the relation between a map and the carried function values. A
consumer relying on a serialized relation must check it: for restriction,
compare each target value with its mapped source value; for induction,
check each mapped subgroup class lies in the stated parent class and the
induced values satisfy the class-sum formula. The operations construct these
relations directly from admitted canonical class partitions.

## Class multiplication constants

`group.class_multiplication_constants.compute` returns the multiplication
constants of the class algebra. Each constant is the number of ordered pairs of
class elements whose product equals one fixed element of the target class, not
the total number of pairs landing anywhere in that class. The implementation
divides the raw pair count by the target class size and retains an exact result
for every admitted constant. This is group data, not a convolution of a
particular character, so a consumer combining it with a class function must
check the class axis it uses.

## Character kernel

`character.kernel.compute` returns a `CharacterKernel` containing the ambient
group and the subgroup whose elements satisfy `chi(g) == chi(1)`. It does not
return a virtual character, sum over a kernel class, or average values. The
result's subgroup is the exact kernel of the supplied ordinary character.

## Character center

`character.center.compute` returns a `CharacterCenter` containing the source
character, the subgroup on which the representation acts by scalars, and the
selected class indices with their normalized scalar values. It does not perform
centralizer averaging or return a virtual-character projection.

## Symmetric and exterior squares

`character.symmetric_square.compute` and `character.exterior_square.compute`
return the virtual characters for the symmetric and exterior square
representations, in the same irreducible basis as their input. Their sum is the
tensor square, so a consumer can cross-check a symmetric/exterior pair against
`character.tensor_product.compute`.

## Adams operations

`character.adams_operation.compute` returns the virtual character whose value at
`g` is `chi(g**k)` for a positive exponent `k`, computed from the class-power map
of the canonical complete conjugacy partition. See
[Finite-group Adams operations](../finite-group-adams-operation.md).

## Character degree

The native `jacobian.math.groups.characters.degree.character_degree` helper
returns the exact degree `chi(1)` for an ordinary
character represented by nonnegative irreducible multiplicities in a
`CharacterRingElement`. It sums each multiplicity times the corresponding
canonical irreducible degree and retains the source character/table with the
integer value. Signed virtual characters are rejected because their value at
the identity is a virtual dimension, not the degree of an ordinary
representation. It is a deterministic projection of the retained irreducible
coordinates and row degrees, so it is not published in `math.find` / `math.run`.
The helper reconstructs the canonical table from the
retained concrete group before using row degrees, with source-group order at
most 60 and bounded table, arithmetic work, and result size.
