# Dirichlet-character conductors

`dirichlet_character.conductor.compute` returns the exact conductor of a
character whose finite unit-group parent has been supplied explicitly. The
result is the least positive divisor `d` of the source modulus `N` through
which the character factors.

For every divisor `d | N`, reduction from `(Z/NZ)^*` to `(Z/dZ)^*` is
surjective. A character factors through that reduction exactly when it is
trivial on the reduction kernel, the units congruent to `1 mod d`. The
operation computes exact cyclotomic exponents for the admitted unit table,
then checks these kernels in increasing divisor order. It does not search for
or approximate a conductor.

Admission is bounded by the source character-group envelope (`N <= 2,048`,
complete unit table `phi(N) <= 2,048`) and a maximum of 4,194,304
divisor/unit checks. The operation either returns the exact least divisor or
rejects before the kernel scan.

Example: the nonprincipal character modulo `8` with value `-1` at `5` and
`1` at `7` has conductor `8`; the character with value `-1` at `3` and `7`
has conductor `4`.

The result also contains the inducing character modulo `d`, represented as a
serialized `PrimitiveDirichletCharacter`: the canonical dual coordinates and
the exact conductor `d`. For every unit `a mod N`, its value at `a mod d`
agrees with the source character's value at `a`. The operation finds it by
lifting target generators to source units and transporting exact values; the
kernel criterion ensures the lift choice does not matter. The carrier records
the primitive claim for composition. Operations relying on primitivity check
that claim against the exact conductor at their boundary.

[Number-theory operations](index.md) · [Tool surface](../../tools.md)
