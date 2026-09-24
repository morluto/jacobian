# Restriction of a Dirichlet character to a divisor modulus

`dirichlet_character.restrict_modulus.compute` asks whether a source character
modulo `m` factors through the reduction of unit groups
`(Z/mZ)^* -> (Z/dZ)^*`, for one positive divisor `d` of `m`.

The reduction map is surjective. Therefore the character descends exactly when
it is constant on every fiber. On success, the result contains the exact
character modulo `d`, all target unit residues in canonical order, and one
canonical source-unit lift for each target unit. The operation chooses the
least source residue in each fiber. Pulling the descended character back by
`dirichlet_character.inflate.compute` reconstructs the source character.

If the character does not factor, the result has status `does_not_factor` and
contains two source units in the same reduction fiber with distinct exact
root-of-unity values. Their reduction residue and target group are included,
so the values form a direct obstruction to descent. The operation does not
restrict to arbitrary subgroups or infer coordinate isomorphisms.

Both moduli use the canonical character-group bound of `2,048`. Source and
target group construction, fiber comparison work, and the largest success or
obstruction serialization are admitted before expanding the fiber map.

This agrees with Sage's restriction contract: restriction to a divisor modulus
is defined when the divisor is a multiple of the character conductor
([Sage Dirichlet-character reference](https://doc.sagemath.org/html/en/reference/modfrm/sage/modular/dirichlet.html)).
Here factorization is established directly by exact value constancy on the
reduction fibers; failure returns a concrete witness rather than a bare claim.

The serialized result's structural validation checks its shape, moduli, unit
membership, alignment, and internal witness consistency. Those checks do not
recompute the source character's values. A consumer that relies on a
deserialized relation must authenticate it against the source character: on
success, compare the target values with the source values at the aligned lifts;
on failure, compare both witness values with the source character. The
operation constructs both relations from exact values before returning them.
