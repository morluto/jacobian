# Nonnegative Petri-net invariant generators

`petri_net.nonnegative_invariant_generators.compute` returns the complete
Hilbert bases of

```text
{ y in N^P : y^T C = 0 }     and     { x in N^T : C x = 0 },
```

where `C = Post - Pre` is the exact incidence matrix of the supplied net. The
first basis generates the nonnegative P-invariant monoid; the second generates
the nonnegative T-invariant monoid. Both are bound to the place and transition
axes of that net.

These monoid generators are different from the signed integer-lattice bases
returned by `petri_net.invariants.compute`. For example, for `C = [2, 3, -5]`,
the nonnegative T-invariant Hilbert basis is
`(0,5,3), (1,1,1), (5,0,2)`: the interior generator cannot be recovered from
the two extreme rays alone.

The operation uses exact bounded enumeration. A Steinitz bound on the length
of conformally indecomposable kernel vectors gives a finite complete search
bound; the candidate and pairwise decomposition work and serialized result
size are admitted before enumeration. Requests above these limits are rejected
without a partial basis. The bound follows from the Graver-basis estimate
derived from the Steinitz lemma; see [Eisenbrand, Hunkenschröder, and Klein,
“About the complexity of two-stage stochastic IPs”](https://link.springer.com/article/10.1007/s10107-021-01698-z).

A T-invariant is only an algebraic transition-count vector satisfying the
incidence state equation. It does not specify a firing order or imply that the
transitions are enabled from any marking. In particular, nonzero generators
may exist when the initial marking enables no transition.
