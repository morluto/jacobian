"""The dead-test-helper gate must resolve names the way Python does.

``make test-dead-code`` is the only thing enforcing this invariant, so a
checker that under-reports lets dead helpers accumulate invisibly again. The
probes below are the specific shapes that a flat module-level name set gets
wrong: a local parameter or an attribute access that happens to share a
helper's name resolves elsewhere, so neither is a reference to it.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from tools.check_test_dead_helpers import Violation, _check_file, check

SHADOWED_BY_PARAMETER = '''\
"""A parameter that shares a module-level name is not a reference to it."""

value = 1


def test_shadowing(value: int) -> None:
    assert value
'''

SHADOWED_BY_ATTRIBUTE = '''\
"""An attribute access never resolves to a module-level binding."""


def helper() -> int:
    return 2


def test_attribute() -> None:
    class Holder:
        helper = None

    assert Holder().helper is None
'''

STRING_LITERAL_IS_NOT_A_REFERENCE = '''\
"""A string with the helper name does not call or load the helper."""


def helper() -> int:
    return 2


def test_string_value_is_not_a_reference() -> None:
    message = "helper"
    assert message == "helper"
'''

COLLECTED_CLASS_REFERENCES_ITS_BASE = '''\
"""The class base is evaluated in the module scope."""


class Base:
    pass


class TestCase(Base):
    pass
'''

LIVE_HELPERS = '''\
"""Live helpers, fixtures, and pytest collection must all survive."""

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def live_fixture():
    return 7


def live_helper() -> int:
    return 3


def test_fixture_argument(live_fixture: int) -> None:
    assert live_fixture == 7


@pytest.mark.usefixtures("live_fixture")
def test_fixture_by_name() -> None:
    assert live_helper() == 3


@pytest.fixture
def fixture_used_only_by_marker():
    return 8


@pytest.mark.usefixtures("fixture_used_only_by_marker")
def test_fixture_used_only_by_marker() -> None:
    pass


def test_nested_scope_rebinds() -> None:
    def helper() -> int:
        return helper  # noqa: F821 - deliberately the global, not itself

    assert helper() == 3
'''


def _write(tmp_path: Path, name: str, body: str) -> Path:
    directory = tmp_path / "tests"
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _reported(tmp_path: Path, name: str, body: str) -> set[str]:
    path = _write(tmp_path, name, body)
    return {
        violation.name
        for violation in _check_file(tmp_path, path)
        if isinstance(violation, Violation)
    }


def test_parameter_shadowing_does_not_rescue_a_dead_variable(tmp_path: Path) -> None:
    assert "value" in _reported(tmp_path, "test_shadow.py", SHADOWED_BY_PARAMETER)


def test_attribute_access_does_not_rescue_a_dead_function(tmp_path: Path) -> None:
    assert "helper" in _reported(tmp_path, "test_shadow.py", SHADOWED_BY_ATTRIBUTE)


def test_unrelated_string_literals_do_not_rescue_a_dead_function(
    tmp_path: Path,
) -> None:
    assert "helper" in _reported(
        tmp_path, "test_strings.py", STRING_LITERAL_IS_NOT_A_REFERENCE
    )


def test_local_assignment_inside_statement_does_not_rescue_global_helper(
    tmp_path: Path,
) -> None:
    body = """\
    def helper() -> int:
        return 1

    def test_local_shadow() -> None:
        if True:
            helper = 2
        assert helper == 2
    """
    assert "helper" in _reported(tmp_path, "test_local_shadow.py", body)


def test_directly_parametrized_argument_does_not_rescue_fixture(
    tmp_path: Path,
) -> None:
    body = """\
    import pytest

    @pytest.fixture
    def item():
        return 1

    @pytest.mark.parametrize("item", [2])
    def test_direct_value(item):
        assert item == 2
    """
    assert "item" in _reported(tmp_path, "test_direct_param.py", body)


def test_aliased_fixture_name_resolves_to_its_definition(tmp_path: Path) -> None:
    body = """\
    import pytest

    @pytest.fixture(name="item")
    def _item():
        return 1

    def test_fixture_argument(item):
        assert item == 1
    """
    assert _reported(tmp_path, "test_fixture_alias.py", body) == set()


def test_collected_class_base_is_scanned_in_module_scope(tmp_path: Path) -> None:
    assert (
        _reported(tmp_path, "test_class_base.py", COLLECTED_CLASS_REFERENCES_ITS_BASE)
        == set()
    )


def test_live_helpers_fixtures_and_pytest_collection_are_reported_clean(
    tmp_path: Path,
) -> None:
    assert _reported(tmp_path, "test_live.py", LIVE_HELPERS) == set()


def test_waiver_comment_exempts_a_deliberate_helper(tmp_path: Path) -> None:
    body = (
        '"""Deliberate."""\n\n\n'
        "def kept() -> int:  # dead-code: exercised through an external harness\n"
        "    return 1\n"
    )
    assert _reported(tmp_path, "test_waived.py", body) == set()


def test_check_reports_a_summary_for_the_whole_tree() -> None:
    report = check(Path(__file__).resolve().parents[2])

    assert report.files_scanned > 0
    assert report.render().startswith("test-dead-code:")
    assert report.failed is bool(report.violations)


@pytest.mark.parametrize("body", [SHADOWED_BY_PARAMETER, SHADOWED_BY_ATTRIBUTE])
def test_unparseable_module_is_skipped_rather_than_judged(
    tmp_path: Path, body: str
) -> None:
    # A file that does not parse cannot be judged, so the gate skips it instead
    # of claiming a verdict it never reached.
    path = _write(tmp_path, "test_broken.py", body)
    path.write_text(path.read_text(encoding="utf-8") + "\ndef (\n", encoding="utf-8")

    assert _check_file(tmp_path, path) == ()
