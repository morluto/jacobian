# Exact homogeneous free-algebra ideal components

`free_algebra.two_sided_ideal.degree_component.compute` returns the exact
degree-`n` vector subspace of a finitely generated homogeneous two-sided ideal
in `QQ<X>`, represented by its canonical reduced row-echelon basis in the
degree-`n` word coordinates.

For each homogeneous generator `g` of degree `d <= n`, the operation forms all
context multiples

```text
u g v, with |u| + |v| = n - d
```

and takes their rational row span. Each context length split and each word on
the declared ordered alphabet is included. Thus generators of different
degrees contribute correctly to different context families. Zero generators
contribute no rows. Nonhomogeneous generators and left or right ideals are
rejected because their component semantics are different.

The result is an independent finite linear-algebra calculation; it does not
depend on a Gröbner–Shirshov basis or its completion status. The returned
basis gives a consumer the complete degree-`n` ideal subspace. Membership of a
homogeneous degree-`n` polynomial can be decided by solving an exact linear
system against its word coefficient vector.

Admission is performed before context generation or matrix construction. One
request is limited to 128 ambient words, 256 context rows, 32,768 matrix cells,
64 decimal digits in the Hadamard bound for rational row reduction, and a
conservative 2 MB serialized result estimate including the source presentation.
These are operation limits; they
do not assert that larger homogeneous components are mathematically undefined.

## Bounded ideal membership

`free_algebra.two_sided_ideal.membership.decide` decides whether a sparse
polynomial belongs to a finitely generated two-sided ideal in `QQ<X>`. It
requires every ideal generator to be homogeneous, completes the existing
degree-lexicographic Gröbner–Shirshov basis through the candidate's largest
word degree, and then computes the candidate's exact normal form. Homogeneity
makes the finite cutoff sound: compositions and reductions preserve total
degree, so completion through the candidate degree decides that candidate.

`MEMBER` and `NOT_MEMBER` are returned only after completion and reduction
finish. `UNKNOWN` has no normal form or completion claim and is returned when
the bounded work or coefficient envelope stops the calculation. A nonzero
normal form witnesses nonmembership relative to the completed basis; a zero
normal form establishes membership. Nonhomogeneous generators are rejected
because degree-bounded completion alone would not certify their membership
behavior.

This is a bounded decision operation, not a global completion claim. The
existing limits on ordered-pair checks, generated compositions, reductions,
coefficient growth, words, and terms remain in force.
