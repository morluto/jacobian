# Finite-group character operations

[Finite mathematics operations](index.md) · [Tool surface](../../tools.md)

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
authenticate the relation between a map and the carried function values. A
consumer relying on a serialized relation must check it: for restriction,
compare each target value with its mapped source value; for induction,
check each mapped subgroup class lies in the stated parent class and the
induced values satisfy the class-sum formula. The operations construct these
relations directly from admitted canonical class partitions.
