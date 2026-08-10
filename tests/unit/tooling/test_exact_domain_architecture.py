"""Architecture ratchet for the verified-domain fixture seam."""

from __future__ import annotations

from pathlib import Path

from tools.check_test_architecture import check_test_architecture


def test_domain_conftest_cannot_copy_exact_domain_install_recipe(
    tmp_path: Path,
) -> None:
    conftest = tmp_path / "tests" / "domain" / "matrix" / "conftest.py"
    conftest.parent.mkdir(parents=True)
    conftest.write_text(
        "from jacobian.exact_domain_checkers import install_exact_domain_verification\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "topology.toml").write_text(
        'version = 1\n[[lanes]]\nname = "domain"\ntier = "domain"\n'
        'paths = ["tests/domain"]\n',
        encoding="utf-8",
    )

    report = check_test_architecture(tmp_path)

    assert any(item.code == "exact-domain-install-recipe" for item in report.violations)


def test_non_root_pytest_plugins_are_rejected(tmp_path: Path) -> None:
    conftest = tmp_path / "tests" / "composition" / "conftest.py"
    conftest.parent.mkdir(parents=True)
    conftest.write_text(
        'pytest_plugins = ("tests.support.runtime_templates",)\n',
        encoding="utf-8",
    )
    (tmp_path / "tests" / "topology.toml").write_text(
        'version = 1\n[[lanes]]\nname = "composition"\ntier = "composition"\n'
        'paths = ["tests/composition"]\n',
        encoding="utf-8",
    )

    report = check_test_architecture(tmp_path)

    assert any(item.code == "non-root-pytest-plugins" for item in report.violations)
