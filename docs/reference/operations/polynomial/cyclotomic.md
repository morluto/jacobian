# Exact cyclotomic polynomials

`polynomial.cyclotomic.compute` constructs the primitive integer polynomial
`Phi_n(x)` for one positive integer index. The result retains the source index,
the degree `phi(n)`, and a dense `ZZ[x]` polynomial whose coefficients are in
descending-degree order. The `n=1` result is `x - 1`.

The operation admits factorization and divisor work, intermediate coefficient
growth, degree, per-coefficient height, and total exact coefficient digits
before calling SymPy's exact `ZZ` polynomial backend. These are semantic
execution limits, independent of the serialized wire representation. Requests
outside that envelope return a typed resource diagnostic; backend failures
remain backend failures and cannot become a zero polynomial. The shared
request deadline is checked before admission, after admission, after backend
completion, and before result construction.

The returned `IntegerPolynomial` is the same canonical value accepted by the
integer polynomial composition, evaluation, content, primitive-part, and GCD
operations. Consumers that work in `QQ[x]` may promote its integer
coefficients explicitly, preserving the declared variable axis and coefficient
order.

For independent checking, the defining divisor identity is

```text
x^n - 1 = product(Phi_d(x) for d dividing n)
```

and `Phi_n(x)` divides `x^m - 1` exactly when `n` divides `m`.
