# RSK insertion traces

`tableau.rsk.word.trace.compute` returns the ordinary row-insertion RSK pair
and one insertion event for every source position. Under
`ROW_INSERTION_RSK_V1`, each event records the zero-based row and column of
every bumped cell, the entry removed from that cell, and the terminal cell and
entry added by the insertion. The carried entry at each bump is the source
letter's alphabet rank or the preceding bump's removed entry. The event's
terminal row follows all recorded bump rows.

The source word and exact alphabet are retained with the final pair and the
trace. The empty word returns an empty event tuple and its empty pair. The
compact `tableau.rsk.word.compute` operation remains available when only the
pair is needed.

The request admits at most 500 letters, computes a bound using both the word
length and alphabet size, and preflights row-search work and a conservative
serialized-output estimate before constructing insertion events. A
semistandard tableau over `a` ordered ranks has height at most `a`, so the
maximum number of bumps is bounded by
`sum(min(i, a) for i in range(n))`, not by the square of the full word length.
The result limit is 8 MB.

The path follows Knuth's row insertion: at each row replace the leftmost entry
strictly greater than the carried value and carry the replaced value to the
next row, appending when no such entry exists. See Knuth, [*Permutations,
Matrices, and Generalized Young Tableaux*](https://doi.org/10.2140/pjm.1970.34.709),
§2 and §3.
