# Primitive positive formula evaluation

`pp_formula.evaluate_relation.compute` evaluates one typed primitive-positive
formula on one finite relational structure and returns its complete defined
relation. Formulas are conjunctions of relation atoms and logical equality
atoms. Variables have canonical integer coordinates; the listed
`free_variables` are ordered result axes, and all other declared variables are
existentially quantified. A relation atom must name a symbol in the exact
structure signature and use exactly its declared arity. An empty conjunction
is true, including the empty tuple for a true sentence.

Formula construction canonicalizes the conjunction: it orders equality
endpoints, removes repeated atoms, and sorts the remaining atoms
deterministically. The public input allows at most 64 atoms before this
normalization. Relation-atom term order is preserved, and the explicit
free-variable axis order remains part of the formula value. Thus equivalent
atom orderings and repetitions have the same serialized formula value without
quotienting variable renamings or result-axis permutations.

The `PPDefinedRelation` value retains the source structure, formula, and sorted
unique tuple table. Its tuple coordinates follow the formula's free-variable
order. This relation is an exact finite value after serialization; it does not
claim that an arbitrary first-order formula is supported. In particular,
negation, disjunction, universal quantification, function symbols, and
many-sorted signatures are outside this contract.

Evaluation exhausts assignments over the finite carrier. The kernel admits the
candidate-assignment count, worst-case atom checks, and worst-case defined
relation size, and coordinate-extraction work before constructing relation
indexes or expanding assignments.
Exceeding any envelope is a resource refusal and never yields a partial
relation. This first slice supplies the formula-to-relation semantics used by
conjunctive queries and CSP solution relations; canonical-database conversion,
formula composition, and solver-based evaluation remain separate operations.
