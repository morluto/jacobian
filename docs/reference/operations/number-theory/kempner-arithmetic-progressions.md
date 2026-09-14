# Arithmetic progressions in Kempner digit sets

`number_theory.kempner_set.arithmetic_progression.decide` decides whether the
positive integers whose canonical base-`b` digits all lie in a proper subset
`S` contain a nontrivial progression of the requested fixed arity.

The result is exact for every admitted request. `PROGRESSION_FREE` means that
the complete finite carry-state graph has been exhausted. When a progression
exists, `CONTAINS_PROGRESSION` includes positive `a`, positive `d`, source
indices `(0, ..., k-1)`, and the values `(a, a+d, ..., a+(k-1)d)`.

Canonical numerals have no leading zero; zero digits are checked only while
they are significant. The kernel reads least-significant digits, retaining
carry vectors for all terms and a finite pending-leading-zero flag. BFS first
minimizes the common padded digit length and then chooses the lexicographically
first `(a_digit, d_digit)` path in least-significant-first order.

Admission derives the carry-state count, `b²` digit-pair transitions per
state, predecessor storage, witness digit bound, and serialized result size.
Requests outside that envelope are rejected before search. A deadline or
cancellation interrupts execution; it never becomes a progression-free
answer.

Example request:

```json
{"digit_set": {"base": "3", "allowed_digits": ["1", "2"]}, "arity": "3"}
```

This returns the witness `1, 4, 7` with `a = 1` and `d = 3`.

The digit-family convention follows Walker and Walker, [Arithmetic
Progressions with Restricted Digits](https://doi.org/10.1080/00029890.2020.1682888);
the operation supplies a bounded executable decision for one selected base,
digit subset, and arity rather than claiming a theorem about all such families.

[Number-theory operation index](index.md)
