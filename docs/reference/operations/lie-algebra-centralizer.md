# Lie algebra common centralizer

`lie_algebra.subalgebra.centralizer.compute` takes a finite-dimensional Lie
algebra over `QQ` and at most `dim(g)` exact vectors on its ordered basis axis.
It returns

\[
  C_{\mathfrak g}(S)=\{x\in\mathfrak g:[x,s]=0\text{ for every }s\in S\}
\]

as canonical RREF generator rows in a `LieSubalgebra` value that retains the
exact source algebra. The input family may be empty; in that case the result
is the whole algebra. Supplying a basis of a subalgebra computes its
centralizer, since commuting with the basis is equivalent to commuting with
its linear span.

The kernel is formed from the stacked linear equations `[x,s]=0` in the
coordinates of `x`. The exact kernel is complete for accepted inputs. The
canonical Lie-algebra value establishes Jacobi at construction; this operation
separately bounds family length, rational coordinate digits, and linear-system
work before the solve. The returned centralizer is closed under the bracket by
Jacobi.

This operation returns the subalgebra itself, not an ideal claim. A consumer
that relies on a caller-supplied `LieSubalgebra` checks its bracket closure at
the point of use.

Sage's finite-dimensional Lie algebra API independently uses the same
centralizer boundary: it accepts either a subalgebra or a list of elements
and returns a centralizer basis ([Sage documentation](https://doc.sagemath.org/html/en/reference/categories/sage/categories/finite_dimensional_lie_algebras_with_basis.html)).
