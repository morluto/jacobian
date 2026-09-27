"""The finite simplicial-set carrier includes its initial object."""

from jacobian.math.topology.simplicial_sets.operations import from_tables


def test_all_empty_degree_prefix_is_the_initial_simplicial_set_and_roundtrips():
    for max_degree in range(5):
        value = from_tables(
            max_degree=max_degree,
            sets=tuple(() for _ in range(max_degree + 1)),
            face_maps=tuple(
                tuple(() for _ in range(degree + 1))
                for degree in range(1, max_degree + 1)
            ),
            degeneracy_maps=tuple(
                tuple(() for _ in range(degree + 1)) for degree in range(max_degree)
            ),
        )
        assert value.status == "SIMPLICIAL_SET"
        assert value.simplicial_set is not None
        assert value.simplicial_set.total_simplices == 0
        assert value.simplicial_set.sets == tuple(() for _ in range(max_degree + 1))
        assert (
            value.simplicial_set.model_validate_json(
                value.simplicial_set.model_dump_json()
            )
            == value.simplicial_set
        )


def test_empty_degree_prefix_can_be_truncated_without_inventing_simplices():
    source = from_tables(
        3,
        ((), (), (), ()),
        (((), ()), ((), (), ()), ((), (), (), ())),
        (((),), ((), ()), ((), (), ())),
    )
    assert source.simplicial_set is not None
    assert source.simplicial_set.total_simplices == 0
