# Weyl Bruhat intervals

`weyl_group.bruhat_interval.compute` returns the complete closed interval
`[u,v] = {x : u <= x <= v}` in the strong Bruhat order of a finite
crystallographic Weyl group. Both endpoints are canonical `WeylElement`
values on the same ordered Cartan parent. The result retains both endpoints,
the exact Weyl elements in the interval, and a finite poset. Entry `elements[i]`
is the Weyl element named by `poset.elements[i]`; the poset also carries all
strict comparisons, cover relations, incomparable pairs, and length ranks
relative to `u`.

The operation admits only ambient groups of order at most 64. It computes that
order from the Weyl exponents and estimates enumeration, pairwise Bruhat tests,
and the complete output size before enumerating elements. The returned set is
always complete. If `u` and `v` are incomparable, the interval is represented
by the empty poset while the endpoints remain available. If `u=v`, the result
is the singleton interval.

The implementation uses the Coxeter subword property: for a fixed reduced
expression of `v`, an element `x` satisfies `x <= v` exactly when a reduced
expression of `x` occurs as a (not necessarily consecutive) subword. It
enumerates subword actions together with selected lengths and keeps an action
only when the selected length equals that element's shortest length. The
Bruhat order is graded by Coxeter length, which supplies the interval ranks.
The proof basis is the Subword Property (Theorem 2) and Chain Property
(Theorem 4) in Tom Denton, *Lifting Property and Poset Structure of Finite
Coxeter Groups*, Lecture 9
([UC Davis lecture notes](https://www.math.ucdavis.edu/~anne/WQ2009/MAT280-Lecture9.pdf)).

Independent type-A correctness evidence compares all of `S_4` against the
permutation rank-matrix criterion for Bruhat order. This checks the computed
strict relation independently of the reduced-word subword implementation.
