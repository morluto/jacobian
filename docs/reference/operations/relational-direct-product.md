# Finite relational direct products

`relational.structure.direct_product.compute` forms the categorical direct
product of two finite relational structures when their ranked signatures agree
symbol for symbol and in order. The carrier label `i * |B| + j` represents the
pair `(i,j)` in lexicographic order. A tuple belongs to a product relation
exactly when its left and right coordinate tuples belong to the corresponding
factor relations.

The result retains both factors and the coordinate projections. These maps
are homomorphisms and can be passed directly to the relational homomorphism
checker. Projection arrays and the product structure survive JSON round trips.
Nullary relations use the ordinary product of truth values: the empty tuple is
present exactly when it is present in both factors. Empty carriers are also
supported.

Admission checks the Cartesian carrier size, each product relation's row
count, and the aggregate coordinate work before expanding any pair of relation
rows; the admitted visits bound the complete product and both projection
arrays. A refused request therefore does not return a partial structure. This
binary operation is also the
reusable carrier construction needed to represent polymorphisms as
homomorphisms from relational powers; callers can compose it repeatedly while
respecting the same bounds at each step.
