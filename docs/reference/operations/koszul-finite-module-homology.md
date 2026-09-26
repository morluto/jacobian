# Finite module Koszul complexes and homology

The `homological.koszul.complex.compute` operation constructs the full sequence
derived Koszul complex of a finite based module `M` over a finite dimensional
commutative `QQ` algebra `A`. In degree `k`, the basis is ordered as module
basis coordinates followed by canonical increasing `k`-subsets of the ordered
sequence. The differential uses

```text
d(m tensor e_(i1 ^ ... ^ ik))
  = sum_j (-1)^(j-1) (f_ij m) tensor e_(i1 ^ ... omit ij ^ ... ^ ik).
```

The operation reconstructs each sparse matrix from the supplied module action
and checks every consecutive product is zero. A selected degree can be
constructed separately with `homological.koszul.differential.compute`, which
retains the parent, sequence, and both wedge axes.

`homological.koszul.homology.compute` consumes a serialized complete complex,
revalidates its algebra, module action, axes, and sequence, checks `d^2=0`, and
computes exact rational bases. For each degree `k`, it returns

```text
dim H_k = dim K_k - rank(d_k) - rank(d_(k+1))
```

along with cycle and boundary dimensions and exact bases for cycles, boundaries,
and homology, expressed in the retained chain basis. Homology representatives
are selected deterministically by extending the boundary basis with cycle
vectors in canonical source-coordinate order. The empty sequence therefore returns
`dim H_0 = dim M`; a unit entry on the regular module gives zero homology; and
for `A = QQ[x]/(x^2)`, `M=A`, and sequence `(x)`, both `H_0` and `H_1` have
dimension one.

The current finite module envelope limits the sequence length to 6, algebra
basis dimension to 6, module basis dimension to 8, and total chain basis size
`dim(M) * 2^length` to 256. Before replaying source actions, square checks, or
RREF, homology admission rejects numerator or denominator components over 128
decimal digits, estimates the serialized retained complex plus homology bases
at no more than 8 MiB, and rejects a coefficient-aware exact linear algebra work estimate
above `2^40` units. For each matrix, the estimate bounds rowwise
common denominators by their product, clears the row scales, applies Hadamard's
bound to the pivot minors, and accounts for transient Fraction products and
subtractions. The dense scan and pivot update count is multiplied by a
quadratic bit-width cost bound. This intentionally conservative estimate may
reject some matrices that happen to reduce cheaply. The output estimate counts
every retained rational, sparse matrix cell, representative coordinate, and
basis label; dimensions are at most 256 and the profile has at most seven
degrees. Degree zero has the same quotient space as
`homological.koszul.module_quotient.compute`; that operation additionally
returns its quotient algebra-module structure and projection. Homology does not
infer that a sequence is regular. A positive
degree homology dimension is an exact obstruction to acyclicity above degree
zero; vanishing dimensions describe this supplied complex only. The current
result is over `QQ`; finite fields and extension coefficients are not accepted.

The estimate bounds elimination work, not wall-clock time. A resource failure
does not imply any homology dimension or exactness conclusion.

`homological.koszul.module_quotient.compute` constructs the degree-zero value
`M/(f_1,...,f_r)M` as a based module over the same algebra. It returns the
canonical RREF basis of the relation submodule, quotient basis representatives,
and the exact projection from the source module. The empty sequence returns the
source module up to its canonical quotient basis; a unit-generated relation
submodule returns the zero module. For the dual numbers `QQ[e]/(e^2)` with
`M=A` and sequence `(e)`, the relation space is `QQ e`, the quotient basis is
represented by `1`, the projection is `(a,b) -> a`, and `e` acts by zero on the
one-dimensional quotient. This operation supplies the `H_0` quotient module
with its action and projection.

`homological.koszul.dga.compute` constructs the unital differential graded
algebra `K(f; A)` for a finite dimensional unital commutative `QQ` algebra
`A` and an ordered sequence `f`. Its underlying complex is the regular-module
Koszul complex. The product is the tensor product of multiplication in `A`
and the exterior product on the sequence generators, with basis order given
by algebra coordinates inside lexicographically ordered increasing wedges.
The result retains the algebra, sequence, unit, and sparse nonzero structure
coefficients indexed by both input degrees and basis positions. Repeated wedge
indices have zero product; swapping homogeneous factors gives the graded
commutativity sign. The differential is the Koszul differential and obeys the
graded Leibniz rule for this product.

The DGA operation requires an explicit unit that acts on every basis element
on both sides. It checks commutativity, associativity, the unit equations, and
the full sparse product table before returning a value. The current envelope
uses sequence length at most 6, algebra dimension at most 6, total chain basis
size at most 256, at most 200,000 nonzero product coefficients, at most
2,000,000 admitted exact work units, and at most 8 MiB estimated output.
Admission accounts for product-table growth and exact coefficient sizes before
constructing the regular module, differentials, or products. The operation
returns the DGA structure it constructs; it does not infer regularity of the
sequence or exactness of the augmented complex.

`homological.koszul.exactness_profile.compute` reports the complete homology
dimension profile, whether the supplied complex is acyclic in positive
degrees, and the first nonzero higher class in its retained chain coordinates
when one exists. Degree zero is deliberately excluded from the acyclicity
predicate: `H_0` is the quotient and may be nonzero. The profile is computed
through the admitted exact homology operation, so the same coefficient, work,
and output bounds apply. Acyclicity is a statement about this finite complex;
it does not by itself label the sequence regular under a module convention or
after localization/base change.

`homological.koszul.sequence_permute.compute` uses `new_to_old[j]` to specify
which source entry occupies target position `j`. It constructs the target
sequence in that order and returns source-to-target and inverse chain maps in
every degree. On a basis element `m tensor e_I`, each map fixes the module
coordinate and sends the increasing wedge to the reordered wedge with the sign
of the sorting permutation. The operation checks that the submitted source
differentials are exactly those induced by their retained sequence and module
action, verifies both chain-map equations and inverse maps, and admits the
doubled complexes and map tables before reconstruction. In particular,
transposing two sequence entries acts by `-1` on the top exterior power; equal
sequence elements are still separate ordered generators.

`homological.koszul.unit_contraction.compute` accepts a selected sequence
entry only when it is a unit in the retained finite algebra. It returns that
entry's exact inverse and the degree-raising maps induced by inserting its
generator into each exterior basis vector and acting by the inverse on the
module coordinate. The operation checks `dH + Hd = id` in every degree. Thus
the unaugmented complex is contractible whenever the sequence contains a unit;
this is a property of the supplied finite algebra and module, not a regularity
claim for a sequence in a larger ring.

`homological.koszul.append_zero.compute` appends the zero element as a new,
distinct sequence position. It returns the extended complex together with
degreewise inclusions and projections identifying it with the direct sum of
the original complex and its degree shift. The shifted summand has differential
`-d`, as required by the homological shift convention. The operation checks
both chain-map identities and the direct-sum splitting on every basis vector.
The extended sequence remains within the finite-module length limit.

`homological.koszul.module_map.compute` takes an exact matrix from a based
finite module `M` to another `N` over the same algebra, with rows indexed by
the target basis and columns by the source basis. It checks the module-map
equations `phi A_i = B_i phi` for every algebra basis action, then returns the
source and target complexes on the same ordered sequence and the induced map
`id_(wedge^k) tensor phi` in each degree. The producer checks each chain square
`d_N phi_k = phi_(k-1) d_M`; the operation admits the aggregate wedge maps,
action checks, differential products, and serialized result before expanding
them. This is the finite exact form of the usual functoriality of Koszul
complexes in the coefficient module; see [Hochster's commutative algebra lecture notes](https://dept.math.lsa.umich.edu/~hochster/615W12/615W12.pdf) and the [Stacks Project's functoriality lemma for Koszul complexes](https://stacks.math.columbia.edu/tag/0621).

The returned value binds both complexes and the original module map. Its JSON
decoder checks axes and parent bindings but does not repeat the module-linearity
or chain-square computation. A future consumer of a caller-supplied decoded
map must check the relations it relies on. Current bounds are sequence length
6, module dimensions 8, at most 4,096 aggregate degree-map cells, 2,000,000
estimated exact work units, and 8 MiB estimated output. Coefficient growth in
module-linearity, algebra-action, and chain-square checks is bounded before
exact arithmetic. The coefficient domain remains `QQ`.

`homological.koszul.homology_map.compute` consumes this typed chain map and
reconstructs it from its retained module map, rechecking module-linearity and
every chain square before using it. The operation computes exact homology for
both complexes and applies each chain matrix to the source homology basis. It
then expresses each image in the target's boundary-plus-homology basis and
returns the homology coordinates. Matrices use target homology classes as rows
and source classes as columns; the result retains both full homology values so
those coordinates are interpretable after serialization. This is the ordinary
functorial map `ker(d)/im(d) -> ker(d')/im(d')` induced by a chain map; see the
[Stacks Project definition and functoriality of homology](https://stacks.math.columbia.edu/tag/010V).

The initial induced-map envelope caps the combined source and target chain-basis
count at 8, each reconstructed differential or degree-map rational component at
8 decimal digits, the exact quotient-coordinate work estimate at `2^40`, and
the aggregate retained homology and induced-map output at 8 MiB. These estimates
are computed from the complexes rebuilt from the retained modules and sequence,
after the supplied chain map is reconstructed and checked, and before either
homology elimination or quotient-coordinate expansion. This smaller bound
reflects the additional exact quotient-coordinate solve; standalone
construction and homology keep their larger envelopes. As with the other
decoded Koszul values, shape checks alone do not authenticate producer history:
this operation explicitly checks the module and chain-map relations it uses.
