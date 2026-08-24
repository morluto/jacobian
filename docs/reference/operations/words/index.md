# Combinatorics on words

[Documentation home](../../../index.md) · [Tool surface](../../tools.md)

The words domain owns finite words, finite word morphisms, and substitutions.
The live catalog remains authoritative for request and result schemas. Three
substitution operations complement the existing finite-word factor, period,
and incidence-matrix operations:

- `substitution.dependency_graph.compute` returns the exact edge `a -> b` for
  every letter `b` occurring in `sigma(a)`. Each edge retains its multiplicity
  and all zero-based occurrence positions, and the canonical graph retains its
  source substitution. The aggregate returned occurrence ledger is limited to
  10,000 positions.
- `substitution.primitivity_profile.compute` accepts that canonical graph
  unchanged. It returns deterministic strongly connected components,
  aperiodicity, and either the least exponent whose Boolean dependency power is
  positive or a reducible/periodic obstruction. Checking powers through the
  Wielandt bound makes a negative result complete; the operation does not
  publish a second matrix carrier or materialize potentially enormous integer
  powers.
- `substitution.fixed_point_prefix.compute` accepts a substitution that is
  prolongable on a named seed: `sigma(a) = a u` with nonempty, nonmortal `u`.
  Images outside the growing seed orbit may erase. It returns at most 500
  letters, the least sufficient iterate depth, and the retained prefix length
  at every generation. The source conditions imply unbounded nested seed
  iterates. Admission bounds aggregate source occurrences, retained-generation
  work, and the predicted serialized result; each generation collects at most
  the requested number of letters. The initial envelope admits 20,000 source
  image occurrences, 1,000,000 generation work units including result replay,
  and a 512,000-byte
  predicted result.

All three results replay against their exact sources during validation.
Alphabet order fixes morphism rows, matrix axes, edge order, and component
order; coherent relabelling changes labels but not scalar primitivity data.
NetworkX 3.6 supplies the maintained SCC and aperiodicity kernels. A bounded
bitset Boolean-product kernel is used only for the finite matrix-power
criterion and least exponent.

These operations make no infinite factor-language claim. In particular,
observing that factors stabilize across a few prefixes or substitution
iterates does not prove completeness for the generated language. Complete
substitution factor languages, factor-complexity prefixes, and Rauzy graphs
remain deferred until a reviewed finite-completeness algorithm can retain the
required occurrence and saturation evidence.
