# Exact finite-sequence autocorrelation

The sequence catalog publishes two bounded exact transforms over a finite real
rational sequence:

- `sequence.autocorrelation.aperiodic.compute` returns coefficients at the
  signed lag axis `-(n-1), ..., 0, ..., n-1`, with no wraparound:

  `c_k = sum_j a_(j+k) a_j` for every pair of indices that remains in the
  sequence.

- `sequence.autocorrelation.cyclic.compute` returns one complete residue axis
  `0, ..., n-1`, with indices interpreted modulo `n`:

  `c_k = sum_(j mod n) a_(j+k mod n) a_j`.

The result includes the explicit `convention` field (`"aperiodic"` or
`"cyclic"`), so the lag axis remains self-describing after transport. The
source sequence is retained in the result. Integer wire entries are canonical
denominator-one rationals; native integer callers may use the finite-integer
carrier directly and receive compact exact integer cells.
Rational cells use `CanonicalRational`. Empty sequences return an empty cell
axis for either convention. Complex coefficients are outside this real-rational
contract; conjugation is therefore not an option hidden behind the operation.

Both operations admit the complete source, multiplication work, exact
coefficient growth, and materialized output before constructing the quadratic
coefficient table. Oversized requests return the operation's typed domain or
resource admission error rather than partially computed rows.

[Number-theory operations](index.md) · [Tool reference](../../tools.md)
