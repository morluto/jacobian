# Monic left division of differential Ore operators

`ore.differential.operator.left_division.compute` returns exact operators
`Q` and `R` satisfying `A = B*Q + R`, with `order(R) < order(B)`, in
`QQ(x)<D>` under the Weyl rule `D*a = a*D + a'`. The factor order is part of
the contract; differential-operator multiplication is noncommutative.

The current bounded domain accepts input orders at most four, integer
polynomial coefficients in `ZZ[x]` of degree at most two with at most three
terms and two-digit coefficients, and a monic nonzero divisor. Monicity
ensures each leading cancellation remains in `ZZ[x]`. The implementation
admits coefficient degree and height growth, cancellation work, and serialized
output size before expansion. The result retains both inputs and reconstructs
the exact left-division identity. Ore's treatment of noncommutative polynomial
division provides the algebraic context ([Ore, 1933](https://webhomes.maths.ed.ac.uk/~v1ranick/papers/ore.pdf)).

The finite fixture divides
`x*D^2 + (x^2+2)*D + x + 1` by `D+x`; its quotient is `x*D+1` and its
remainder is `1`. This distinguishes `A=B*Q+R` from right division.

[Polynomial operations](index.md)
