from jacobian.math.graphs import rooted_trees


def test_exact_rooted_tree_public_api() -> None:
    expected = (
        "RootedTreeFinePartition",
        "RootedTreeFinePartitionConstructed",
        "RootedTreeNotATree",
        "RootedTreeShrub",
        "construct_fine_partition",
    )

    assert tuple(rooted_trees.__all__) == expected
    assert len(rooted_trees.__all__) == len(set(rooted_trees.__all__))
    assert all(hasattr(rooted_trees, name) for name in rooted_trees.__all__)
    assert not hasattr(rooted_trees, "RootedTreeFinePartitionRequest")
