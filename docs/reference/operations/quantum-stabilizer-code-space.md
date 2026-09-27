# Stabilizer code spaces

`quantum.stabilizer.code.compute` accepts an exact qubit stabilizer group and
one strict `+1` or `-1` eigenvalue for each independent generator. It absorbs
the selected character into the exact Pauli phases and returns the canonical
RREF group whose generators all have eigenvalue `+1` on the selected code
space. It encodes `n-r` logical qubits and has Hilbert-space dimension
`2^(n-r)`, where `r` is the group rank. The
result remains a compact register-bound value; no state vector or dense
projector is constructed.

This is the standard stabilizer-code definition as a simultaneous eigenspace
of an Abelian Pauli subgroup. See [Preskill, Chapter 7](https://www.preskill.caltech.edu/ph229/notes/chap7.pdf).

```json
{
  "group": {
    "register": {"qubit_ids": ["q0"]},
    "generators": [
      {
        "phase_free": {
          "register": {"qubit_ids": ["q0"]},
          "x_bits": [0],
          "z_bits": [1]
        },
        "phase": 0
      }
    ]
  },
  "generator_eigenvalues": [-1]
}
```

The example selects the `-1` eigenspace of `Z`; the returned exact generator
is `-Z` and has `+1` eigenvalue throughout the represented code space.

The operation admits at most 32 qubits and 64 group generators. It validates
the authored exact group at the operation boundary, bounds row reduction and
output materialization, and rejects dependent group presentations. It does
not enumerate codewords or create Hilbert-space vectors.
