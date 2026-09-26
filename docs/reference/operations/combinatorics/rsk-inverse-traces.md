# RSK reverse-insertion traces

`tableau.rsk.inverse_word.trace.compute` reconstructs the unique finite word
represented by a compatible ordinary row-insertion RSK pair and returns one
reverse-insertion event per recording label, in descending label order.
Each event records the removed outer corner and insertion entry, every
reverse bump cell and displaced entry, the emitted alphabet rank as a symbol,
and the partition row lengths after removal. Thus the ledger can be replayed
from the supplied pair until both tableaux are empty.

Under `ROW_INSERTION_RSK_V1`, remove label `n` from the recording tableau and
the matching corner from the insertion tableau. Carry that entry upward by
replacing the rightmost strictly smaller entry in each preceding row. The
entry carried out of the first row is the source letter at position `n`.
Repeating for labels `n-1` through `1` recovers the word. This is the inverse
of Knuth's row insertion; see [*Permutations, Matrices, and Generalized Young
Tableaux*](https://doi.org/10.2140/pjm.1970.34.709), §2 and §3.

Admission checks the pair's cell count, alphabet payload, row-search work,
and conservative serialized result size before reverse insertion. The trace
uses the same 500-cell and 8 MB result limits as forward insertion traces.
The compact `tableau.rsk.inverse_word.compute` operation remains appropriate
when only the reconstructed word is needed.
