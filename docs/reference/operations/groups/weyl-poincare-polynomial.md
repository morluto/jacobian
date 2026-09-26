# Weyl-group Poincare polynomial

`root_system.weyl_poincare_polynomial.compute` returns the exact length
generating polynomial

```text
P_W(q) = sum_{w in W} q^length(w) = product_i (1 + q + ... + q^m_i),
```

where the `m_i` are the Weyl exponents of the finite crystallographic Cartan
datum. Coefficients use the shared dense `ZZ[q]` polynomial carrier in
descending-degree order. Reducible data produce the product polynomial on the
full datum. The parent Cartan matrix is retained with the polynomial.

The operation derives the exponents from positive-root heights and never
enumerates Weyl-group elements. Rank is at most 8, degree at most 120, and work
and serialized output are admitted before coefficient construction. For
validation, the coefficients sum to the Weyl-group order, and the polynomial
is palindromic with degree equal to the number of positive roots.
