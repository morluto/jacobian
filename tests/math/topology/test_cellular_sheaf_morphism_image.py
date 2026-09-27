"""Exact image sheaves and their canonical factorization."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cellular_sheaves import (
    SheafField,
    SheafMorphismImageRequest,
    SheafMorphismImageResult,
    SheafMorphismResult,
    SheafStalk,
    from_cover_maps,
    image_of_morphism,
    morphism,
)
from jacobian.math.topology.cellular_sheaves._models import CoverRestrictionMatrix


def _q(value: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _sheaf(rank: int = 2, field: SheafField = SheafField.RATIONAL, prime=None):
    complex_ = canonical_complex(("a", "b"), (("a", "b"),))
    faces = tuple(face for group in complex_.faces_by_dimension for face in group.faces)
    covers = tuple(
        (face, coface)
        for coface in faces
        for face in faces
        if len(coface) == len(face) + 1 and set(face) < set(coface)
    )
    result = from_cover_maps(
        complex_,
        field,
        prime,
        tuple(
            SheafStalk(simplex=face, basis=tuple(f"x{i}" for i in range(rank)))
            for face in faces
        ),
        tuple(
            CoverRestrictionMatrix(
                source=face,
                target=coface,
                entries=tuple(
                    tuple(
                        (1 if i == j else 0)
                        if field is SheafField.PRIME_FIELD
                        else _q(1 if i == j else 0)
                        for i in range(rank)
                    )
                    for j in range(rank)
                ),
            )
            for face, coface in covers
        ),
    )
    assert result.sheaf is not None
    return result.sheaf


def _varying_rank_sheaf():
    complex_ = canonical_complex(("a", "b"), (("a", "b"),))
    faces = tuple(face for group in complex_.faces_by_dimension for face in group.faces)
    ranks = {("a",): 8, ("b",): 1, ("a", "b"): 1}
    covers = tuple(
        (face, coface)
        for coface in faces
        for face in faces
        if len(coface) == len(face) + 1 and set(face) < set(coface)
    )
    result = from_cover_maps(
        complex_,
        SheafField.RATIONAL,
        None,
        tuple(
            SheafStalk(simplex=face, basis=tuple(f"x{i}" for i in range(ranks[face])))
            for face in faces
        ),
        tuple(
            CoverRestrictionMatrix(
                source=face,
                target=coface,
                entries=tuple(
                    tuple(_q(int(row == column)) for column in range(ranks[face]))
                    for row in range(ranks[coface])
                ),
            )
            for face, coface in covers
        ),
    )
    assert result.sheaf is not None
    return result.sheaf


def _block_matrix(components, faces, row_rank: int, column_rank: int):
    rows = len(faces) * row_rank
    columns = len(faces) * column_rank
    matrix = [[Fraction(0) for _ in range(columns)] for _ in range(rows)]
    by_cell = dict(components)
    for block, face in enumerate(faces):
        component = by_cell[face]
        for row in range(row_rank):
            for column in range(column_rank):
                entry = component[row][column]
                matrix[block * row_rank + row][block * column_rank + column] = Fraction(
                    entry.num, entry.den
                )
    return tuple(tuple(row) for row in matrix)


def test_image_factorization_reconstructs_non_surjective_map() -> None:
    source, target = _sheaf(), _sheaf()
    components = tuple(
        (cell, ((_q(1), _q(0)), (_q(0), _q(0)))) for cell in source.canonical_face_order
    )
    original = morphism(source, target, components)
    result = image_of_morphism(original)

    assert tuple(len(stalk.basis) for stalk in result.image.stalks) == (1, 1, 1)
    assert result.inclusion.components[0][1] == ((_q(1),), (_q(0),))
    assert result.factor.components[0][1] == ((_q(1), _q(0)),)
    assert all(item.entries == ((_q(1),),) for item in result.image.cover_restrictions)

    # Independent matrix assembly checks inclusion * factor = original at every cell.
    inclusions = dict(result.inclusion.components)
    factors = dict(result.factor.components)
    for cell, matrix in original.components:
        left = tuple(
            tuple(Fraction(entry.num, entry.den) for entry in row)
            for row in inclusions[cell]
        )
        right = tuple(
            tuple(Fraction(entry.num, entry.den) for entry in row)
            for row in factors[cell]
        )
        reconstructed = tuple(
            tuple(
                sum(left[i][k] * right[k][j] for k in range(len(right)))
                for j in range(2)
            )
            for i in range(2)
        )
        assert reconstructed == tuple(
            tuple(Fraction(entry.num, entry.den) for entry in row) for row in matrix
        )
    # Independently assemble the degreewise cochain matrices as block diagonals
    # in simplex order, then check the returned factorization after direct sums.
    for degree in source.complex.faces_by_dimension:
        faces = degree.faces
        original_cochain = _block_matrix(original.components, faces, 2, 2)
        inclusion_cochain = _block_matrix(result.inclusion.components, faces, 2, 1)
        factor_cochain = _block_matrix(result.factor.components, faces, 1, 2)
        product = tuple(
            tuple(
                sum(
                    inclusion_cochain[row][inner] * factor_cochain[inner][column]
                    for inner in range(len(factor_cochain))
                )
                for column in range(len(factor_cochain[0]))
            )
            for row in range(len(inclusion_cochain))
        )
        assert product == original_cochain
    restored = SheafMorphismImageResult.model_validate_json(result.model_dump_json())
    assert restored == result
    malformed = result.model_dump(mode="python")
    malformed["morphism"]["components"] = malformed["morphism"]["components"][:-1]
    with pytest.raises(ValueError, match="morphism components"):
        SheafMorphismImageResult.model_validate(malformed)
    malformed = result.model_dump(mode="python")
    components = list(malformed["morphism"]["components"])
    cell = components[0][0]
    components[0] = (cell, ((_q(1),),))
    malformed["morphism"]["components"] = tuple(components)
    with pytest.raises(ValueError, match="morphism matrices must match"):
        SheafMorphismImageResult.model_validate(malformed)
    malformed = result.model_dump(mode="python")
    malformed["morphism"]["natural"] = False
    with pytest.raises(ValueError, match="factorization morphisms must be natural"):
        SheafMorphismImageResult.model_validate(malformed)

    malformed = result.model_dump(mode="python")
    malformed["image"]["cover_restrictions"] = malformed["image"][
        "cover_restrictions"
    ][:-1]
    malformed["inclusion"]["source"]["cover_restrictions"] = malformed[
        "image"
    ]["cover_restrictions"]
    malformed["factor"]["target"]["cover_restrictions"] = malformed["image"][
        "cover_restrictions"
    ]
    with pytest.raises(ValueError, match="image cover_restrictions"):
        SheafMorphismImageResult.model_validate(malformed)
    tool = next(
        item
        for item in BUILTIN_TOOLS
        if item.operation_id == "cellular_sheaf.morphism.image.compute"
    )
    assert tool.run(SheafMorphismImageRequest(morphism=original)) == result


def test_image_restrictions_are_induced_from_target_sheaf() -> None:
    complex_ = canonical_complex(("a", "b"), (("a", "b"),))
    faces = tuple(face for group in complex_.faces_by_dimension for face in group.faces)
    covers = tuple(
        (face, coface)
        for coface in faces
        for face in faces
        if len(coface) == len(face) + 1 and set(face) < set(coface)
    )

    def rank_one_sheaf(edge_restriction: int):
        result = from_cover_maps(
            complex_,
            SheafField.RATIONAL,
            None,
            tuple(SheafStalk(simplex=face, basis=("x",)) for face in faces),
            tuple(
                CoverRestrictionMatrix(
                    source=face,
                    target=coface,
                    entries=((_q(1 if len(coface) == 1 else edge_restriction),),),
                )
                for face, coface in covers
            ),
        )
        assert result.sheaf is not None
        return result.sheaf

    source = rank_one_sheaf(2)
    target = rank_one_sheaf(3)
    morphism_components = tuple(
        (cell, ((_q(2 if len(cell) == 1 else 3),),))
        for cell in faces
    )
    original = morphism(source, target, morphism_components)
    result = image_of_morphism(original)

    assert all(item.entries == ((_q(3),),) for item in target.cover_restrictions)
    assert all(item.entries == ((_q(2),),) for item in result.image.cover_restrictions)
    assert result.inclusion.natural
    assert result.factor.natural


def test_zero_map_has_zero_image_on_every_face() -> None:
    source, target = _sheaf(rank=1), _sheaf(rank=1)
    original = morphism(
        source,
        target,
        tuple((cell, ((_q(0),),)) for cell in source.canonical_face_order),
    )
    result = image_of_morphism(original)
    assert all(stalk.basis == () for stalk in result.image.stalks)
    assert all(
        restriction.entries == () for restriction in result.image.cover_restrictions
    )
    assert result.inclusion.components[0][1] == ((),)
    assert result.factor.components[0][1] == ()
    assert (
        SheafMorphismImageResult.model_validate_json(result.model_dump_json()) == result
    )


def test_prime_field_image_uses_declared_arithmetic() -> None:
    source = _sheaf(rank=2, field=SheafField.PRIME_FIELD, prime=3)
    target = _sheaf(rank=2, field=SheafField.PRIME_FIELD, prime=3)
    original = morphism(
        source,
        target,
        tuple((cell, ((1, 2), (2, 1))) for cell in source.canonical_face_order),
    )
    result = image_of_morphism(original)
    assert result.image.coefficient_field is SheafField.PRIME_FIELD
    assert result.image.prime == 3
    assert all(len(stalk.basis) == 1 for stalk in result.image.stalks)


def test_image_rejects_non_natural_candidate_even_if_flag_claims_true() -> None:
    source, target = _sheaf(rank=1), _sheaf(rank=1)
    valid = morphism(
        source,
        target,
        tuple((cell, ((_q(1),),)) for cell in source.canonical_face_order),
    )
    components = list(valid.components)
    components[0] = (components[0][0], ((_q(0),),))
    forged = SheafMorphismResult.model_construct(
        source=source,
        target=target,
        components=tuple(components),
        natural=True,
        obstruction=None,
    )
    with pytest.raises(OperationDomainValidationError, match="natural"):
        image_of_morphism(forged)

    malformed_parent = SheafMorphismResult.model_construct(
        source={}, target=target, components=(), natural=True, obstruction=None
    )
    with pytest.raises(OperationDomainValidationError, match="parents"):
        image_of_morphism(malformed_parent)

    malformed = SheafMorphismResult.model_construct(
        source=source,
        target=target,
        components=((source.canonical_face_order[0], (1,)),),
        natural=True,
        obstruction=None,
    )
    with pytest.raises(OperationDomainValidationError, match="simplex, matrix"):
        image_of_morphism(malformed)

    malformed_key = SheafMorphismResult.model_construct(
        source=source,
        target=target,
        components=((('unknown',), ((_q(1),),)),),
        natural=True,
        obstruction=None,
    )
    with pytest.raises(OperationDomainValidationError, match="canonical order"):
        image_of_morphism(malformed_key)


def test_image_decoder_retains_parent_diagram_accounting() -> None:
    source, target = _sheaf(rank=1), _sheaf(rank=1)
    original = morphism(
        source,
        target,
        tuple((cell, ((_q(1),),)) for cell in source.canonical_face_order),
    )
    payload = image_of_morphism(original).model_dump(mode="python")
    for name in ("diamonds", "comparable_pairs"):
        payload["image"][name] += 1
        payload["inclusion"]["source"][name] += 1
        payload["factor"]["target"][name] += 1
    with pytest.raises(ValueError, match="parent"):
        SheafMorphismImageResult.model_validate(payload)


def test_image_decoder_binds_cover_paths_across_the_factorization() -> None:
    source, target = _sheaf(rank=1), _sheaf(rank=1)
    original = morphism(
        source,
        target,
        tuple((cell, ((_q(1),),)) for cell in source.canonical_face_order),
    )
    payload = image_of_morphism(original).model_dump(mode="python")
    for sheaf in (
        payload["image"],
        payload["inclusion"]["source"],
        payload["factor"]["target"],
    ):
        sheaf["cover_restrictions"][0]["cover_path"] = ()
    with pytest.raises(ValueError, match=r"cover_restrictions.*axes"):
        SheafMorphismImageResult.model_validate(payload)


def test_image_decoder_rejects_matching_but_incomplete_parent_diagrams() -> None:
    source, target = _sheaf(rank=1), _sheaf(rank=1)
    original = morphism(
        source,
        target,
        tuple((cell, ((_q(1),),)) for cell in source.canonical_face_order),
    )
    payload = image_of_morphism(original).model_dump(mode="python")

    def remove_restrictions(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                if key == "cover_restrictions" and nested:
                    value[key] = nested[1:]
                else:
                    remove_restrictions(nested)
        elif isinstance(value, list):
            for nested in value:
                remove_restrictions(nested)

    remove_restrictions(payload)
    with pytest.raises(ValueError, match="complete parent diagram"):
        SheafMorphismImageResult.model_validate(payload)


def test_image_rejects_excess_component_count_before_resolving_axes(monkeypatch) -> None:
    from jacobian.math.topology.cellular_sheaves import morphism_image

    source, target = _sheaf(rank=1), _sheaf(rank=1)
    valid = morphism(
        source,
        target,
        tuple((cell, ((_q(1),),)) for cell in source.canonical_face_order),
    )
    malformed = SheafMorphismResult.model_construct(
        source=source,
        target=target,
        components=valid.components * 100,
        natural=True,
        obstruction=None,
    )
    monkeypatch.setattr(
        morphism_image,
        "_resolve_component_key",
        lambda *_args: pytest.fail("oversized components were traversed"),
    )

    with pytest.raises(OperationDomainValidationError, match="one morphism component"):
        image_of_morphism(malformed)


def test_image_rejects_oversized_component_before_resource_scan(monkeypatch) -> None:
    from jacobian.math.topology.cellular_sheaves import morphism_image

    source, target = _sheaf(rank=1), _sheaf(rank=1)
    valid = morphism(
        source,
        target,
        tuple((cell, ((_q(1),),)) for cell in source.canonical_face_order),
    )
    components = list(valid.components)
    cell, _matrix = components[0]
    components[0] = (cell, ((_q(1),), (_q(0),)))
    malformed = SheafMorphismResult.model_construct(
        source=source,
        target=target,
        components=tuple(components),
        natural=True,
        obstruction=None,
    )

    def resource_scan_must_not_run(*args, **kwargs):
        raise AssertionError("resource scan ran before component shape validation")

    monkeypatch.setattr(
        morphism_image, "_admit_morphism_resources", resource_scan_must_not_run
    )
    with pytest.raises(OperationDomainValidationError, match="stalk axes"):
        image_of_morphism(malformed)


def test_induced_restriction_work_uses_both_image_ranks() -> None:
    from jacobian.math.topology.cellular_sheaves.morphism_image import (
        _induced_restriction_work,
    )

    sheaf = _varying_rank_sheaf()
    ranks = {stalk.simplex: len(stalk.basis) for stalk in sheaf.stalks}
    restriction = next(
        item
        for item in sheaf.cover_restrictions
        if item.source == ("a",) and item.target == ("a", "b")
    )
    assert _induced_restriction_work(restriction, ranks, ranks) == 82
    identity = morphism(
        sheaf,
        sheaf,
        tuple(
            (
                cell,
                tuple(
                    tuple(_q(int(row == column)) for column in range(ranks[cell]))
                    for row in range(ranks[cell])
                ),
            )
            for cell in sheaf.canonical_face_order
        ),
    )
    image = image_of_morphism(identity).image
    assert {stalk.simplex: len(stalk.basis) for stalk in image.stalks} == ranks
