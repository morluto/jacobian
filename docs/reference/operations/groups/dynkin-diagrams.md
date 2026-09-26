# Finite Dynkin diagrams

`root_system.dynkin_diagram.compute` returns a complete labeled graph together
with its finite Cartan datum. The node axis contains every simple-root index,
including isolated nodes in a reducible datum. An edge joining `i < j` stores
the ordered Cartan pair `(A[i,j], A[j,i])` and its multiplicity
`A[i,j] * A[j,i]`. This retains the exact direction of a multiple bond under
the supplied root ordering; the graph does not collapse B/C or G2 orientation
to an undirected edge label.

For example, the B2 Cartan matrices `[[2,-2],[-1,2]]` and
`[[2,-1],[-2,2]]` have the same undirected double edge but distinct ordered
labels `(-2,-1)` and `(-1,-2)`. G2 carries `(-3,-1)` or its reverse and
multiplicity 3. An A1×A1 input yields two nodes and no edges.

The operation accepts finite Cartan matrices of rank at most 8. Admission
reserves the complete node/edge result and datum matrices before scanning the
bounded Cartan axis; the result contains at most 28 edges. Its typed result is
reusable as Cartan context by the existing root and Weyl operations.

The ordered labels preserve the entries of the supplied Cartan matrix, so the
graph and node ordering retain enough information to reconstruct that matrix.

The named constructor uses `A[i,j] = <alpha_i^vee, alpha_j>`. Along the
linear labeling `0 - ... - (n-1)`, type `B_n` has the short root at the final
node (`A[n-2,n-1] = -1`, `A[n-1,n-2] = -2`); type `C_n` has the long root
there, with those entries reversed. The symmetrizer and root-length profiles
follow this same orientation.
