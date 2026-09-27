# Pauli measurement on stabilizer states

`quantum.stabilizer_state.measure_pauli.compute` measures one Hermitian exact
Pauli on a pure stabilizer state: an exact stabilizer code value with `n`
independent generators on `n` ordered qubits. The Pauli and state must use the
same register. With the phase convention `P = i^r X^x Z^z`, Hermiticity means
`r + x·z` is even.

If the observable commutes with every state generator, it is a signed product
of those generators. The result gives its deterministic eigenvalue, the
generator subset as a bit vector, the residual scalar phase (`0` gives `+1`,
`2` gives `-1`), and the unchanged state. Otherwise, the observable
anticommutes with at least one generator, so the result contains both outcomes
`+1` and `-1`, each with exact probability `1/2`, and the corresponding exact
post-measurement stabilizer state. For each branch, one anticommuting generator
is replaced by the signed observable and the other anticommuting generators
are multiplied by the pivot. No random sample, state vector, or dense
projector is returned.

The operation admits up to 32 qubits and performs its work and result-size
admission before canonicalizing the input tableau or constructing branches.
This rule is the standard pure-state stabilizer measurement update described
by [Aaronson and Gottesman](https://arxiv.org/abs/quant-ph/0406196); the
implementation uses Jacobian's exact phase-lifted Pauli and stabilizer values,
with no external quantum simulator backend.
