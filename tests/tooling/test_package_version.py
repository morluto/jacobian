from importlib.metadata import version

import jacobian


def test_package_version_matches_distribution_metadata() -> None:
    assert jacobian.__version__ == version("jacobian")
