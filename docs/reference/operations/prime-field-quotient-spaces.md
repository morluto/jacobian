# Prime-field quotient spaces

[Operation index](index.md) · [Finite mathematics](finite-math/index.md)

`prime_field.vector_space.quotient.compute` constructs the finite-dimensional
quotient `V/W`, where `V = GF(p)^n` and `W` is the span of caller-supplied
vectors. Its `PrimeFieldQuotientSpace` result retains the prime, ambient axis,
denominator generators, quotient representatives, and the projection matrix.
For an ambient column vector `x`, multiplying by the returned matrix gives
coordinates of its coset in the returned quotient basis.

The construction follows the usual finite-dimensional quotient-space model:
extend a basis of `W` to a basis of `V`; the added vectors represent a basis of
`V/W`, and the quotient map has kernel `W`. The implementation chooses a
deterministic complement from the ambient coordinate vectors. The particular
representatives are therefore canonical for the declared coordinate axis,
while the abstract quotient is independent of that choice. See MIT's
[multilinear algebra notes, §1.2](https://math.mit.edu/classes/18.952/spring2011/chapter1.pdf),
especially the quotient projection and basis-extension exercises on pp. 6–7.

The native helper `jacobian.math.matrices.finite_fields.project_quotient_vector`
consumes the serialized quotient value unchanged. Because it is a deterministic
projection of the public `quotient.compute` result, it remains a native package
export rather than a catalog discovery entry. Before using caller-supplied
projection data, it checks that the projection annihilates the denominator,
sends the retained quotient basis to standard coordinates, and that denominator
and quotient dimensions span the ambient space. This establishes that the
projection kernel is exactly `W`; structural decoding alone does not certify a
caller-authored relation. The result is a `PrimeFieldQuotientVector` parented by
the same quotient value.

The published construction and the native projection consumer use exact
arithmetic over prime fields. The request admits the
prime before elimination, caps ambient dimension plus generator count at 1024,
and bounds elimination work and combined basis/projection cells. The
projection consumer applies the corresponding bound to the exact checks it
performs. Empty and full denominator subspaces are supported, including their
zero-dimensional quotient and explicit empty matrix axes.

These values provide a reusable target for homology and spectral-sequence
pages, where cycles modulo boundaries must retain their ambient parent and
quotient map. The prime-field contract currently does not include `QQ` or
extension fields.
