"""Private FLINT adapters for exact regular-language matrices."""


def accepted_word_count(
    matrix: tuple[tuple[int, ...], ...],
    initial_state: int,
    accepting_states: tuple[int, ...],
    word_length: int,
) -> int:
    """Return the selected row sum of ``matrix ** word_length`` exactly."""

    from flint import fmpz_mat

    powered = fmpz_mat(matrix) ** word_length
    return sum(int(powered[initial_state, target]) for target in accepting_states)


__all__ = ["accepted_word_count"]
