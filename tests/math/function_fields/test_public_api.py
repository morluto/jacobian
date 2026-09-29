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


def test_residue_operation_and_result_are_public() -> None:
    from jacobian.math.function_fields import (
        FunctionFieldResidueResult,
        function_field_place_residue,
    )
    from jacobian.math.function_fields._models import FunctionFieldResidueResult as CanonicalResult
    from jacobian.math.function_fields.operations import (
        function_field_place_residue as canonical_residue,
    )

    assert FunctionFieldResidueResult is CanonicalResult
    assert function_field_place_residue is canonical_residue
