import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.galois._models import (
    QQFieldAutomorphism,
    QQRoot,
    QQSplittingField,
    SplittingFieldResult,
)
from jacobian.math.number_theory.galois.operations import (
    apply_automorphism,
    automorphisms,
    compose_automorphisms,
    splitting_field,
)


def test_quadratic_splitting_field_reconstructs_and_composes_action() -> None:
    result = splitting_field((-2, 0, 1))
    assert result.field.degree == 2
    assert tuple(root.index for root in result.roots) == (0, 1)
    action = automorphisms(result.field).automorphisms[0]
    image = apply_automorphism(action, result.roots[0])
    assert image.index == 1
    identity = compose_automorphisms(action, action)
    assert identity.root_permutation == (0, 1)
    assert result.factor_reconstruction == (-2, 0, 1)
    decoded = SplittingFieldResult.model_validate_json(result.model_dump_json())
    assert automorphisms(decoded.field).automorphisms


def test_degree_six_full_symmetric_group_fits_the_field_carrier() -> None:
    result = splitting_field((-1, -1, 0, 0, 0, 0, 1))

    assert result.field.degree == 720
    assert len(result.field.basis_labels) == 720
    assert tuple(root.index for root in result.roots) == tuple(range(6))
    assert SplittingFieldResult.model_validate_json(result.model_dump_json()) == result


def test_splitting_field_rejects_reducible_source_instead_of_echoing_roots() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        splitting_field((-1, 0, 1))
    assert error.value.errors()[0]["type"] == "galois_theory.polynomial_not_irreducible"


def test_automorphism_consumer_rejects_permutation_outside_source_group() -> None:
    result = splitting_field((1, 0, 0, 0, 1))
    forged = QQFieldAutomorphism.model_construct(
        field=result.field,
        root_permutation=(1, 0, 2, 3),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        apply_automorphism(forged, result.roots[0])
    assert error.value.errors()[0]["type"] == "galois_theory.automorphism_not_in_group"


def test_native_consumers_reject_missing_galois_carrier_fields() -> None:
    missing_degree = QQSplittingField.model_construct(
        source=splitting_field((-2, 0, 1)).field.source,
        basis_labels=("b_0", "b_1"),
        root_labels=("root_0", "root_1"),
    )
    with pytest.raises(OperationDomainValidationError):
        automorphisms(missing_degree)

    missing_automorphism_field = QQFieldAutomorphism.model_construct(
        root_permutation=(0, 1)
    )
    with pytest.raises(OperationDomainValidationError):
        apply_automorphism(
            missing_automorphism_field, splitting_field((-2, 0, 1)).roots[0]
        )


def test_native_consumers_re_admit_forged_root_axis_values() -> None:
    result = splitting_field((-2, 0, 1))
    action = automorphisms(result.field).automorphisms[0]
    forged_permutation = QQFieldAutomorphism.model_construct(
        field=result.field,
        root_permutation=(0, 0),
    )
    with pytest.raises(OperationDomainValidationError) as permutation_error:
        apply_automorphism(forged_permutation, result.roots[0])
    assert (
        permutation_error.value.errors()[0]["type"] == "galois_theory.automorphism_axis"
    )

    forged_root = QQRoot.model_construct(
        field=result.field,
        index=-1,
        multiplicity=1,
    )
    with pytest.raises(OperationDomainValidationError) as root_error:
        apply_automorphism(action, forged_root)
    assert root_error.value.errors()[0]["type"] == "galois_theory.root_axis_mismatch"
