# Check an integer partition candidate

`combinatorics.partition.check` classifies one bounded sequence of exact
integers under the canonical convention that the empty sequence is the
partition of zero and nonempty parts are positive and weakly decreasing.

An admitted candidate returns a result whose discriminated `outcome` is either
`PARTITION` or `NOT_A_PARTITION`. A `PARTITION` outcome contains the canonical
`IntegerPartition`, its size and length, its conjugate, and its one-based
Ferrers cells in row-major order. A `NOT_A_PARTITION` outcome contains the
exact source sequence and its first left-to-right obstruction:

- `NONPOSITIVE_PART`, with a zero-based index and the offending value; or
- `INCREASING_ADJACENT_PARTS`, with the zero-based index of the right-hand
  part and the two values that violate weak decrease.

The source sequence is limited to 500 exact JSON-safe integers, and the sum of
its positive entries must be at most 500. Exceeding these supported bounds is
an input/admission error, not a mathematical `NOT_A_PARTITION` result.
Noninteger values are boundary-invalid. The operation checks the raw sequence
length before constructing the typed tuple.

```json
{"parts": [4, 2, 1]}
```

returns the partition `(4, 2, 1)`, its conjugate `(3, 2, 1, 1)`, size 7, and
the seven Ferrers cells. For `[3, 4, 1]`, the result identifies the adjacent
increase at index 1.
