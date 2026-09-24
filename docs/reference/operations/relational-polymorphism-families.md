# Finite relational polymorphism families

`relational.polymorphisms.arity.enumerate` returns the complete set of
relation-preserving `m`-ary operation tables on one exact finite relational
structure. Each operation table has one value for every lexicographically
ordered input in `A^m`; the returned tables use lexicographic table order and
retain the source structure and arity once. A polymorphism preserves every
basic relation: applying it coordinatewise to any ordered `m`-tuple of rows
from a relation returns a row in that same relation.

The result covers one fixed arity only. It does not enumerate a full clone,
compute term operations, classify CSP complexity, or return a sampled or
partial family. Use `relational.polymorphism.check` to check one supplied
operation when complete family enumeration is outside the admitted envelope.

The operation preflights the complete function-space size `|A|^(|A|^m)`,
per-table size `|A|^m`, worst-case relation-product checks, coordinatewise
table work, and a conservative output-size bound before constructing any
candidate table or relation index. The current envelope admits at most 65,536
candidate tables and 8,388,608 aggregate work steps; output is also limited to
8 MiB. Exceeding an envelope refuses the request and never returns a partial
family. These limits make the operation useful for small finite templates;
they are execution bounds, not a change to the mathematical meaning of a
polymorphism family.

For the empty carrier and positive arity, there is exactly one total operation
from the empty power to the empty carrier: the empty table. It preserves all
relations vacuously where there are no relation-row combinations, including
the exact true/false behavior of nullary relations.
