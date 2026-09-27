# Pauli family commutation matrix

`quantum.pauli.family.commutation_matrix.compute` accepts an ordered family
of named phase-free Pauli values and returns its complete binary symplectic
pairing matrix. The result retains the original family, including its row and
column identifiers and its ordered `QubitRegister`; same-shaped vectors from a
different register cannot be mixed into the family.

For rows (P_i=(x_i\mid z_i)), the entry is

\[
  C_{ij}=x_i\cdot z_j+z_i\cdot x_j\pmod 2.
\]

Thus `C[i][j] = 1` exactly when the corresponding Paulis anticommute.
The matrix is symmetric with zero diagonal over \(\mathbf F_2\). This uses
the binary symplectic convention of the stabilizer formalism; see
[Gottesman's thesis](https://thesis.caltech.edu/2900/). The operation is useful
for inspecting a supplied family and composing its output with check-space
construction. It does not infer stabilizer phase consistency or assert that
the family is an independent generating set.

Families contain 1–64 Paulis on a register of at most 32 qubits. The complete
pairing work is bounded by \(m^2 n\), and both the family and resulting square
matrix are bounded before pair evaluation. The output is an exact matrix, not
a partial prefix.

For `X`, `Z`, and `Y` on one qubit in that order, the matrix is

\[
\begin{pmatrix}0&1&1\\1&0&1\\1&1&0\end{pmatrix}.
\]
