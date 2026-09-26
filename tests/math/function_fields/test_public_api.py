"""Public native values for function-field operations."""


def test_divisor_and_place_values_are_public_canonical_types() -> None:
    from jacobian.math.function_fields import (
        FunctionFieldDivisor,
        FunctionFieldDivisorTerm,
        FunctionFieldPlace,
    )
    from jacobian.math.function_fields._models import (
        FunctionFieldDivisor as CanonicalDivisor,
        FunctionFieldDivisorTerm as CanonicalTerm,
        FunctionFieldPlace as CanonicalPlace,
    )

    assert FunctionFieldDivisor is CanonicalDivisor
    assert FunctionFieldDivisorTerm is CanonicalTerm
    assert FunctionFieldPlace is CanonicalPlace
