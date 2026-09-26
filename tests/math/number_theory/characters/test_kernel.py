import pytest
from pydantic import ValidationError

from jacobian.math.number_theory.characters import (
    character_group,
    dirichlet_character_group_enumerate,
    dirichlet_character_kernel,
)
from jacobian.math.number_theory.characters.values import (
    DirichletCharacter,
    DirichletCharacterKernel,
)


def test_character_kernels_are_complete_subgroups_with_exact_indices():
    for modulus in (1, 3, 5, 8, 12, 15):
        group = character_group(modulus)
        for coordinates in dirichlet_character_group_enumerate(group).coordinates:
            character = DirichletCharacter(group=group, coordinates=coordinates)
            kernel = dirichlet_character_kernel(character)

            assert kernel.character == character
            assert kernel.index * len(kernel.residues) == group.character_count
            assert kernel.residues == tuple(sorted(kernel.residues))
            assert modulus == 1 or all(
                left * right % modulus in kernel.residues
                for left in kernel.residues
                for right in kernel.residues
            )
            assert (
                DirichletCharacterKernel.model_validate_json(kernel.model_dump_json())
                == kernel
            )


def test_kernel_value_rejects_missing_or_extra_residues():
    group = character_group(5)
    character = DirichletCharacter(group=group, coordinates=(2,))
    kernel = dirichlet_character_kernel(character)

    with pytest.raises(ValidationError, match="kernel index"):
        DirichletCharacterKernel(
            character=character,
            residues=kernel.residues[:-1],
            index=kernel.index,
        )
