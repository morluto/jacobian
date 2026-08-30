# Monomial-ideal graded Betti profiles

[Documentation home](../../../index.md) · [Polynomial operations](index.md) ·
[Tool surface](../../tools.md)

`polynomial.monomial_ideal.graded_betti_table.compute` returns the complete
minimal multigraded and standard-graded Betti data of one canonical monomial
ideal in a standard-graded polynomial ring over `QQ`. The exact result also
returns its Castelnuovo--Mumford regularity and whether its minimal resolution
is linear.

The public source is the ordered variable axis and the unique minimal family
of monomial exponent vectors. Generators are descending-lexicographic and
pairwise nondividing; coefficients are unnecessary because every nonzero
rational monomial coefficient is a unit. The unit ideal is outside this
operation's nontrivial-resolution domain.

## Exact kernel and returned profile

The kernel uses the lcm-lattice formula of Gasharov, Peeva, and Welker. For
each nonzero lcm-lattice element `m`, its crosscut complex contains exactly
the generator subsets whose lcm strictly divides `m`. The reduced homology in
degree `i-1` over `QQ` is the multigraded Betti number `beta_(i,m)(I)`.
FLINT computes the ranks of the integral simplicial boundary matrices over
`QQ`.

The result retains every lcm-lattice multidegree together with the crosscut
face counts, boundary ranks, and reduced-homology dimensions used in that
formula. It then gives every nonzero multigraded Betti number and its complete
standard-total-degree coarsening. This is computed once by the kernel; public
result parsing checks only canonical bounded shape and does not replay
lcm-lattice enumeration or matrix ranks.

Reference: V. Gasharov, I. Peeva, and V. Welker, “The lcm-lattice in monomial
resolutions,” *Mathematical Research Letters* 6 (1999), 521–532.

## Bounded domain

One request admits at most eight variables and eight minimal generators, with
each exponent at most 64. Thus it has at most 256 generator subsets and 255
nonzero lcm-lattice elements. Every crosscut boundary matrix has at most 70
rows or columns, and the returned profile has at most 255 lattice rows and
2,040 nonzero Betti slots. These fixed limits bound subset enumeration,
integer-matrix rank work, intermediate memory, and exact serialized output
before the kernel runs.

The contract is specific to `QQ`: reduced homology is taken over the rational
coefficient field, so it does not silently claim characteristic-independent
Betti numbers. General homogeneous ideals, quotient rings, module
resolutions, differentials, and Tor computations remain outside this
operation.
