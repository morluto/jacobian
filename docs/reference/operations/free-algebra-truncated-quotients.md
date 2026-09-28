# Degree-truncated free-algebra quotients

`free_algebra.two_sided_quotient.truncated_algebra.compute` constructs the
finite-dimensional algebra

```text
QQ<X> / (I + F_{>D})
```

where `I` is a homogeneous two-sided ideal and `F_{>D}` is the ideal spanned by
all words of degree greater than `D`. The operation computes a complete
Gröbner–Shirshov basis through degree `D`, returns the canonical normal words
through that degree, and provides the exact rational multiplication table and
unit. Products of total degree greater than `D` are zero by the explicit
truncation; products at or below `D` are reduced modulo `I`.

This value is a genuine finite-dimensional quotient algebra. It does not
claim that the untruncated quotient `QQ<X>/I` is finite-dimensional. Its
presentation, degree cutoff, basis axis, unit, and multiplication table remain
together through JSON serialization. The retained Gröbner–Shirshov completion
records the reduction presentation needed for a future operation to recheck a
table claim. A quotient by the unit ideal is represented by the zero algebra
with empty basis and zero unit.

The operation computes and establishes the table when it constructs its
result. Deserialization performs structural checks on the axes, support,
grading, unit shape, and connection between the source ideal and retained
completion; it does not rerun completion or prove the multiplication laws.
Consequently, a decoded caller-supplied table is data, not a semantic
certificate. Any operation that consumes such a table must admit and check the
specific relation it relies on (including multiplication/reduction laws when
those are used). This slice publishes the target carrier only; it does not yet
provide a consumer operation or establish general consumer interoperability.

The operation admits normal-word candidates using the quotient-profile
envelope, then checks multiplication-table term cells, relation-boundary
scans, worst-case homogeneous reduction work, and conservative serialized
output size before constructing the table. A resource refusal carries no
partial algebra or quotient conclusion.
