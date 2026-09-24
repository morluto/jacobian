# Generate a finite prefix from a polynomial recurrence

`ore.shift.recurrence.generate_finite_prefix.compute` solves a polynomial
coefficient recurrence

```text
sum_i p_i(n) a_(n+i) = 0,  p_i(n) in QQ[n]
```

from exactly `order(operator)` consecutive rational initial values. It returns
the finite sequence and the consecutive indices at which the recurrence was
used. The leading coefficient must be nonzero at every one of those indices.
The operation checks that condition on the complete requested finite interval
before generating any values.

The result is a finite-domain recurrence value. It asserts neither an infinite
sequence nor global P-recursiveness. An infinite-domain value would need a
separate representation with a proof-relevant uniqueness and singular-index
contract.

Generation is bounded by the step and index envelopes, coefficient-evaluation
work, a conservative rational-height recurrence, the exact rational carrier,
and serialized output bytes. Admission precedes recurrence expansion. The
growth estimate is intentionally conservative; requests exceeding it are
rejected even when cancellation could make their actual values small.
