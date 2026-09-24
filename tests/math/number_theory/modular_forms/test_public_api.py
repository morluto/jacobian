"""Exact public native API contract for bounded modular-form spaces."""

from jacobian.math.number_theory import modular_forms


def test_exact_public_api_symbols() -> None:
    assert tuple(modular_forms.__all__) == (
        "LevelOneModularQExpansion",
        "ModularCharacterBasis",
        "ModularCharacterBasisElement",
        "ModularCharacterBasisRequest",
        "ModularCharacterCoordinatesProductRequest",
        "ModularCharacterCoordinatesRequest",
        "ModularCharacterHeckeMatrix",
        "ModularCharacterHeckeMatrixRequest",
        "ModularCharacterHeckeRequest",
        "ModularCharacterQExpansion",
        "ModularFormBasis",
        "ModularFormChangeOfBasisFrame",
        "ModularFormCoordinates",
        "ModularFormEqualityResult",
        "ModularFormFieldQExpansion",
        "ModularFormFramedCoordinates",
        "ModularFormFramedHeckeMatrix",
        "ModularFormHeckeMatrix",
        "ModularFormOperatorImage",
        "ModularFormOperatorImagePrefix",
        "ModularFormSpace",
        "ModularFormSpaceInclusion",
        "ModularQExpansion",
        "formal_q_series_u_operator",
        "formal_q_series_v_operator",
        "level_one_named_q_expansion",
        "modular_character_basis_q_expansions",
        "modular_character_coordinates_hecke",
        "modular_character_coordinates_product",
        "modular_character_coordinates_q_expansion",
        "modular_character_hecke_matrix",
        "modular_form_basis_frame",
        "modular_form_basis_q_expansions",
        "modular_form_coordinates_atkin_lehner",
        "modular_form_coordinates_equal",
        "modular_form_coordinates_extend_field",
        "modular_form_coordinates_from_frame",
        "modular_form_coordinates_hecke",
        "modular_form_coordinates_product",
        "modular_form_coordinates_q_expansion",
        "modular_form_coordinates_to_frame",
        "modular_form_coordinates_transport",
        "modular_form_coordinates_u2",
        "modular_form_coordinates_v2",
        "modular_form_coordinates_v3",
        "modular_form_coordinates_v_degeneracy",
        "modular_form_field_coordinates_q_expansion",
        "modular_form_hecke_matrix",
        "modular_form_hecke_matrix_in_frame",
        "modular_form_operator_image",
        "modular_form_operator_image_q_expansion",
        "modular_form_space_inclusion",
        "named_q_expansion",
        "space_dimension",
        "sturm_bound",
    )
    assert all(hasattr(modular_forms, name) for name in modular_forms.__all__)


def test_q_series_operator_tools_do_not_claim_modular_space_membership() -> None:
    from jacobian.catalog.catalog import Catalog

    catalog = Catalog.open()
    u_prime = catalog.inspect("modular_form.u_operator.apply")
    operations = {
        operation.operation_id
        for operation in catalog.browse(
            namespace="modular_form", limit=100, cursor=None
        ).operations
    }

    assert "modular_form.named.q_expansion.compute" in operations
    assert "modular_form.space.sturm_bound.compute" in operations
    assert "modular_form.hecke.apply" not in operations
    assert "modular_form.u_operator.apply" in operations
    assert u_prime is not None
    assert {"form", "prime"} <= set(u_prime.input_schema["properties"])
    assert {"space", "basis_id", "coordinates"} <= set(
        u_prime.output_schema["properties"]
    )
    assert "modular_form.v_operator.apply" not in operations
    assert "modular_form.formal_q_series.u_operator.compute" in operations
    assert "modular_form.formal_q_series.v_operator.compute" in operations
    assert "modular_form.coordinates.hecke.apply" in operations
    assert "modular_form.coordinates.product.compute" in operations
    assert "modular_form.coordinates.extend_field.compute" in operations
    assert "modular_form.field_coordinates.q_expansion.compute" in operations
    assert "modular_form.hecke_matrix.compute" in operations
    assert "modular_form.coordinates.u2.apply" in operations
    assert "modular_form.coordinates.v2.apply" in operations
    assert "modular_form.coordinates.v3.apply" in operations
    assert "modular_form.coordinates.v_degeneracy.apply" in operations
    assert "modular_form.coordinates.operator_image.compute" in operations
    assert "modular_form.basis_frame.create" in operations
    assert "modular_form.coordinates.to_frame.compute" in operations
    assert "modular_form.coordinates.from_frame.compute" in operations
    assert "modular_form.coordinates.transport.compute" in operations
    assert "modular_form.space.inclusion.compute" in operations
    assert "modular_form.equal.check" in operations
    assert "modular_form.equal.check" in operations
    assert "modular_form.character_coordinates.q_expansion.compute" in operations
    assert "modular_form.character_coordinates.product.compute" in operations
    assert "modular_form.character_hecke_matrix.compute" in operations
    assert "modular_form.character_basis.compute" in operations
