# Stabilizer erasure correctability

`quantum.stabilizer.erasure_correctability.compute` decides whether an explicit
set `E` of qubit axes is correctable for a supplied phase-free isotropic check
space `S`. It returns the dimensions of the supported normalizer
`S^perp ∩ V_E`, the supported stabilizer `S ∩ V_E`, and their quotient, plus a
boolean and (when the quotient is nonzero) a supported logical Pauli witness.
The output check space is canonicalized over GF(2), and the erasure axes are
returned in register order.

The criterion is exact: erasure of `E` is correctable precisely when there is
no nontrivial logical Pauli supported entirely in `E`. Algebraically, this
means `S^perp ∩ V_E ⊆ S`. The operation computes the kernel of the restricted
symplectic commutation equations and the kernel of the stabilizer projection
to coordinates outside `E`; their dimensions determine the logical quotient
supported in `E`. This is the stabilizer erasure criterion in Preskill's
[*Cleaning lemma for stabilizer codes*](https://www.preskill.caltech.edu/ph219/topological-stabilizer-codes-2022.pdf).

The uncorrectable branch includes an exact phase-free Pauli in `S^perp ∩ V_E`
that is outside `S`. It is not a decoder and makes no statement about stochastic
noise. The typed request admits at most 32 labelled qubits and 64 check rows;
work for isotropy checks, binary elimination, kernel construction, and witness
membership is bounded before elimination. The duplicated typed source and
witness result are also admitted against a 1 MB output envelope.
