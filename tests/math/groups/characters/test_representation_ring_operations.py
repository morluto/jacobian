def test_cyclic_group_of_order_eleven_is_admitted_for_tensor_product() -> None:
    """The published contract covers cyclic groups through order 60.

    Two dimensionally wrong terms in the tensor-product work estimate rejected
    this input: the pairing term multiplied by `order` on top of the per-entry
    charge `(order + 4 * dimension**2)`, and `predicted_digits` added
    `(order - 1)` digits when summing `order` rationals of width `d` grows the
    numerator by at most `len(str(order))`. Together they charged 243M units
    against a 50M cap for a product that takes about 20ms.
    """
    from jacobian.math.groups._models import (
        GroupConjugacyClassesResult,
        PermutationGroup,
    )
    from jacobian.math.groups.characters._models import CharacterRingElement
    from jacobian.math.groups.characters.operations import character_table
    from jacobian.math.groups.characters.representation_ring_operations import (
        character_tensor_product,
    )
    from jacobian.math.groups.operations import group_conjugacy_classes

    order = 11
    generator = (*range(1, order), 0)
    group = PermutationGroup(degree=order, generators=(generator,))
    classes = group_conjugacy_classes(order, [list(generator)])
    partition = GroupConjugacyClassesResult._from_kernel(
        group, tuple(tuple(tuple(element) for element in cls) for cls in classes)
    )
    table = character_table(partition)
    trivial = CharacterRingElement(
        table=table, irreducible_multiplicities=(1,) + (0,) * (len(table.rows) - 1)
    )

    result = character_tensor_product(trivial, trivial)

    assert result.irreducible_multiplicities == (1,) + (0,) * (len(table.rows) - 1)
