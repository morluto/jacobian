from jacobian.math.graphs import rooted_trees


def test_exact_rooted_tree_public_api() -> None:
    expected = (
        "RootedTreeFinePartition",
        "RootedTreeFinePartitionConstructed",
        "RootedTreeNotATree",
        "RootedTreeShrub",
        "construct_fine_partition",
    )

    # The module's declared __all__ is the source of truth: every
    # advertised name must stay exported, but adding one is allowed.
    assert set(expected) <= set(rooted_trees.__all__)
    # The subset assertion above cannot see a newly added name, so the
    # private-name rule is checked over the complete __all__.
    assert all(not name.startswith("_") for name in rooted_trees.__all__)
    assert len(rooted_trees.__all__) == len(set(rooted_trees.__all__))
    assert all(hasattr(rooted_trees, name) for name in rooted_trees.__all__)
    assert not hasattr(rooted_trees, "RootedTreeFinePartitionRequest")
