# Exact Dirichlet-character parity

`dirichlet_character.parity.compute` returns the exact value \(\chi(-1)\) in
the source character's cyclotomic parent. Since \(-1\) has order at most two
in the unit group, this value is exactly `1` or `-1`; the result labels those
cases `EVEN` and `ODD`, respectively. The exact root and source character are
both retained, so parity is not inferred from a floating approximation.

The operation admits and revalidates the complete source unit-group parent
before evaluating the dual-coordinate homomorphism.
