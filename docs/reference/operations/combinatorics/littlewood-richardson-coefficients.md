# Littlewood–Richardson coefficients

`symmetric_function.littlewood_richardson.coefficient.compute` computes the
Schur structure constant

```text
c^outer_{inner, content}
```

as the number of semistandard tableaux of skew shape `outer / inner` and
content `content`. Rows weakly increase from left to right and columns
strictly increase from top to bottom. Its reading word scans each row from
right to left, starting at the top row and moving down. Every prefix must
contain at least as many `i` entries as `i+1` entries for each positive `i`.
This is the reverse-row-reading lattice-word convention stated with the LR
coefficient definition in [On Classical groups detected by the triple tensor
product and the Littlewood–Richardson semigroup](https://link.springer.com/article/10.1007/s40993-016-0049-3), §2.1.

The operation returns one exact coefficient bound to all three canonical
`IntegerPartition` inputs. It returns zero when the inner diagram is not
contained in the outer diagram or when the sizes do not match. For complete
search, each input partition has size at most 8 and the number of distinct
content-word prefixes is at most 100,000. These are operation-specific search
bounds; the shared partition and tableau carriers retain their larger 500-cell
envelope.

For example,

```text
s_(2,1) * s_(2,1) contains 2 s_(3,2,1)
```

so `c^(3,2,1)_(2,1),(2,1) = 2`.

The companion operation `symmetric_function.schur.product.compute` returns
the complete expansion `s_left * s_right = sum_lambda c^lambda_{left,right}
s_lambda`. It enumerates all partitions of the total degree and omits zero
coefficients. Terms are ordered by descending lexicographic partition order.
The complete product is admitted only when the total degree is at most 8;
before candidate generation, admission multiplies the number of degree
partitions by the complete content-prefix bound (capped at 1,000,000 units).
This deliberately small shared envelope keeps every coefficient inside the
existing LR kernel's admitted domain. The operation returns a finite Schur
expansion, not a general symmetric-function carrier.
