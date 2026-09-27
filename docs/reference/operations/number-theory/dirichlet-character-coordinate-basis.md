# Dirichlet character coordinate-basis transport

`dirichlet_character.change_coordinate_basis.compute` changes the coordinates
of a character when the same finite unit group is presented using a different
generator basis. It keeps the value of the character at each residue fixed.

The request contains a source character and an explicit coordinate
isomorphism. The isomorphism identifies each target generator with its
coordinates in the source basis. The operation checks the complete source and
target unit decompositions and verifies that the supplied map sends every
canonical unit residue to itself. Both bases must use the same modulus and
cyclotomic root order.

If source generator orders are `d_i`, source character coordinates are `c_i`,
and a target generator has source unit coordinates `r_i`, its character value
has exponent

```text
sum_i c_i * (E / d_i) * r_i modulo E,
```

where `E` is the common group exponent. Dividing this exponent by `E / e_j`,
where `e_j` is the target generator order, gives the target character
coordinate. The result includes the transported character and its complete
table on residues `0,...,N-1`; nonunits have value `null`, representing the
Dirichlet extension by zero.

The finite unit-group modulus and output table are bounded by the existing
2,048-residue character-group envelope. Coordinate-rank work is preflighted
before the map is expanded.

## Example

Modulo 8, the ordered generators `(3,5)` and `(5,3)` describe the same unit
group. The supplied map sends target generator 5 to source coordinates `(0,1)`
and target generator 3 to `(1,0)`. A source character with coordinates `(1,0)`
is returned in target coordinates `(0,1)`, with its four unit values
unchanged.
