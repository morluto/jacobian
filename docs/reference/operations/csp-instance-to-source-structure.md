# Finite CSP instances and source structures

`csp.instance.to_source_structure.compute` converts a finite CSP instance to
the canonical finite relational structure on its variables. Each named
constraint occurrence retains its relation symbol and ordered variable scope.
The output table for a relation contains the distinct scopes named for that
symbol. Thus repeated variables such as `E(x, x)` remain repeated coordinates,
while duplicate occurrences remain distinguishable in the input instance and
collapse to one tuple in the mathematical relation.

For a template `B`, a total assignment from the instance variables to `B` is a
solution exactly when it is a homomorphism from the returned source structure
to `B`. This follows directly from the two definitions: every source tuple is
one constraint scope, and preservation requires the assigned tuple to belong
to the named template relation. No solver or sampling is involved.

The carrier and variable labels are canonical integers. The template uses the
existing bounds for carrier size, signature, relation arity, and complete
relation tables. An instance admits at most 4,096 named constraint occurrences
and 16,384 aggregate scope coordinates. Each scope must name a relation in the
template, have its declared arity, and use only declared variables.

This operation constructs the source structure only. Bounded homomorphism
search and exact homomorphism counting are separate operations over the
resulting structure and the template.
