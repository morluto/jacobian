# Characteristic-two polar pairing

[Documentation home](../../../index.md) · [Tool surface](../../tools.md) · [Operation references](../index.md) · [This domain](index.md)

`quadratic_form.characteristic_two.polar_pairing.compute` evaluates the polar pairing for a polynomial quadratic form over a finite field of characteristic two:

```text
B_Q(x,y) = Q(x+y) - Q(x) - Q(y)
```

Square terms cancel in this pairing, while each mixed term contributes in both coordinate orders. The form, both vectors, and the result must share one field presentation and ordered axis; a field of odd characteristic is refused. Support traversal is admitted before expansion. For `Q(x,y)=x^2+xy` over `GF(2)`, the pairing of `(1,0)` with `(0,1)` is `1`.
