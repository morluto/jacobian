# Strict longest increasing subsequence

[Documentation home](../../../index.md) · [Operation references](../index.md) · [Combinatorics](index.md)

`word.longest_increasing_subsequence.compute` accepts a bounded `FiniteWord`
and returns its exact strict-LIS length, one tuple of zero-based source
positions, the corresponding letters, and the source word. “Strict” means
each selected letter has strictly larger rank in the word's explicit ordered
alphabet than its predecessor; duplicate letters cannot both appear in the
witness. This is the convention for the standalone operation. Ordinary
row-insertion RSK has a different weak/strict correspondence for repeated
letters.

The dynamic program scans every predecessor pair `(j, i)` with `j < i` and
extends a best witness ending at `j` exactly when the alphabet rank at `j` is
smaller than the rank at `i`. Ties keep the first predecessor encountered and
then the earliest endpoint, yielding a deterministic witness. The work bound
is `n(n-1)/2`, with `n <= 500`; the Unicode-scalar payload of the alphabet
and positioned letters is also admitted before the scan. The result validator
checks that positions increase, values replay from those source positions, and
values are strictly ordered. The conservative output bound accounts for the
retained source, the copied witness letters, and the digit width of the index
and length integers.

The source word carries the exact ordered alphabet, and the witness retains
source positions and letters, so no additional target-space or alphabet
representation is required. The result proves witness feasibility; the
operation's complete bounded dynamic program establishes optimality.
