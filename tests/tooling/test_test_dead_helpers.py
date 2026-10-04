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
from tools.check_test_dead_helpers import Violation, _check_file, _test_files, check

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


def test_comprehension_iterator_loads_before_target_binding(tmp_path: Path) -> None:
    body = """\
    helper = [1]

    def test_comprehension() -> None:
        assert [helper for helper in helper] == [1]
    """
    assert _reported(tmp_path, "test_comprehension.py", body) == set()


def test_class_comprehension_body_uses_scope_surrounding_class(
    tmp_path: Path,
) -> None:
    body = """\
    def helper() -> int:
        return 1

    class TestCase:
        helper = 2
        values = [helper() for _ in (0,)]

        def test_values(self):
            assert self.values == [1]
    """
    assert _reported(tmp_path, "test_class_comprehension.py", body) == set()


def test_class_compound_statement_updates_bindings_in_order(tmp_path: Path) -> None:
    body = """\
    def helper():
        return 1

    class TestCase:
        if True:
            helper = 2
            value = helper
    """
    assert "helper" in _reported(tmp_path, "test_class_compound.py", body)


@pytest.mark.parametrize(
    "suite",
    [
        "for _ in (0,):\n    helper = 2\n    value = helper",
        "with context():\n    helper = 2\n    value = helper",
        "try:\n    helper = 2\n    value = helper\nexcept Exception:\n    pass",
        "match 1:\n    case _:\n        helper = 2\n        value = helper",
    ],
)
def test_class_bindings_update_within_each_compound_suite(
    tmp_path: Path, suite: str
) -> None:
    body = "def helper():\n    return 1\n\nclass TestCase:\n" + textwrap.indent(
        suite, "    "
    )
    assert "helper" in _reported(tmp_path, "test_compound_suite.py", body)


def test_false_class_branch_preserves_global_helper_fallback(tmp_path: Path) -> None:
    body = """\
    def helper():
        return 1

    class TestCase:
        if False:
            helper = 2
        value = helper()

        def test_value(self):
            assert self.value == 1
    """
    assert _reported(tmp_path, "test_class_fallback.py", body) == set()


def test_nested_class_does_not_close_over_outer_class_bindings(
    tmp_path: Path,
) -> None:
    body = """\
    def helper():
        return 1

    class TestOuter:
        helper = 2

        class Inner:
            value = helper()

        def test_inner(self):
            assert self.Inner.value == 1
    """
    assert _reported(tmp_path, "test_nested_class.py", body) == set()


def test_lambda_parameter_shadows_module_helper(tmp_path: Path) -> None:
    body = """\
    helper = 1
    callback = lambda helper: helper

    def test_callback() -> None:
        assert callback(2) == 2
    """
    assert "helper" in _reported(tmp_path, "test_lambda.py", body)


def test_destructuring_assignments_are_checked_individually(tmp_path: Path) -> None:
    body = """\
    helper, used = (1, 2)

    def test_used() -> None:
        assert used == 2
    """
    assert _reported(tmp_path, "test_destructure.py", body) == {"helper"}


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


def test_unused_fixture_dependency_does_not_make_either_fixture_live(
    tmp_path: Path,
) -> None:
    body = """\
    import pytest

    @pytest.fixture
    def inner():
        return 1

    @pytest.fixture
    def outer(inner):
        return inner
    """
    assert _reported(tmp_path, "test_fixture_dependency.py", body) == {
        "inner",
        "outer",
    }


def test_unused_dynamic_fixture_dependency_does_not_make_inner_live(
    tmp_path: Path,
) -> None:
    body = """\
    import pytest

    @pytest.fixture
    def inner():
        return 1

    @pytest.fixture
    def outer(request):
        return request.getfixturevalue("inner")
    """
    assert _reported(tmp_path, "test_dynamic_fixture_dependency.py", body) == {
        "inner",
        "outer",
    }


def test_fixture_import_alias_resolves_to_its_definition(tmp_path: Path) -> None:
    body = """\
    from pytest import fixture as fx

    @fx
    def item():
        return 1

    def test_fixture_argument(item):
        assert item == 1
    """
    assert _reported(tmp_path, "test_fixture_import_alias.py", body) == set()


def test_aliased_parametrize_marks_fixture_parameter_as_direct(
    tmp_path: Path,
) -> None:
    body = """\
    import pytest as pt

    @pt.fixture
    def item():
        return 1

    @pt.mark.parametrize("item", [2])
    def test_direct_value(item):
        assert item == 2
    """
    assert _reported(tmp_path, "test_alias_parametrize.py", body) == {"item"}


def test_fixture_lookup_by_name_resolves_to_its_definition(tmp_path: Path) -> None:
    body = """\
    import pytest

    @pytest.fixture(name="item")
    def _item():
        return 1

    def test_dynamic_fixture(request):
        assert request.getfixturevalue("item") == 1
    """
    assert _reported(tmp_path, "test_getfixturevalue.py", body) == set()


def test_pytest_lifecycle_hooks_are_not_dead_helpers(tmp_path: Path) -> None:
    body = """\
    def setup_module(module):
        module.ready = True

    def pytest_generate_tests(metafunc):
        pass
    """
    assert _reported(tmp_path, "test_lifecycle.py", body) == set()


def test_unittest_module_lifecycle_hooks_are_not_dead_helpers(
    tmp_path: Path,
) -> None:
    body = """\
    def setUpModule():
        pass

    def tearDownModule():
        pass
    """
    assert _reported(tmp_path, "test_unittest_lifecycle.py", body) == set()


def test_all_default_pytest_test_function_prefixes_are_collected(
    tmp_path: Path,
) -> None:
    body = """\
    def helper():
        return 1

    def testThing():
        assert helper() == 1

    def test():
        assert helper() == 1
    """
    assert _reported(tmp_path, "test_function_prefix.py", body) == set()


def test_overwritten_module_binding_is_reported_even_when_name_is_used(
    tmp_path: Path,
) -> None:
    body = """\
    helper = 1
    helper = 2

    def test_helper():
        assert helper == 2
    """
    path = _write(tmp_path, "test_overwritten.py", body)
    violations = _check_file(tmp_path, path)
    assert [(item.name, item.line) for item in violations] == [("helper", 1)]


def test_earlier_module_binding_used_before_reassignment_is_live(
    tmp_path: Path,
) -> None:
    body = """\
    import pytest

    CASES = (1,)

    @pytest.mark.parametrize("value", CASES)
    def test_value(value):
        assert value

    CASES = (2,)
    """
    path = _write(tmp_path, "test_binding_order.py", body)
    violations = _check_file(tmp_path, path)
    assert [(item.name, item.line) for item in violations] == [("CASES", 9)]


def test_eager_module_call_resolves_deferred_load_before_reassignment(
    tmp_path: Path,
) -> None:
    body = """\
    def helper():
        return 1

    def use():
        return helper()

    VALUE = use()
    helper = None

    def test_value():
        assert VALUE == 1
    """
    path = _write(tmp_path, "test_eager_call.py", body)
    violations = _check_file(tmp_path, path)
    assert [(item.name, item.line) for item in violations] == [("helper", 8)]


def test_nested_eager_calls_keep_the_binding_used_during_import(
    tmp_path: Path,
) -> None:
    body = """\
    def helper():
        return 1

    def inner():
        return helper()

    def outer():
        return inner()

    VALUE = outer()
    helper = None

    def test_value():
        assert VALUE == 1
    """
    path = _write(tmp_path, "test_nested_eager_call.py", body)
    violations = _check_file(tmp_path, path)
    assert [(item.name, item.line) for item in violations] == [("helper", 11)]


def test_forward_defined_eager_call_chain_uses_import_time_bindings(
    tmp_path: Path,
) -> None:
    body = """\
    def first():
        return second()

    def second():
        return third()

    def third():
        return 1

    VALUE = first()
    third = None

    def test_value():
        assert VALUE == 1
    """
    path = _write(tmp_path, "test_forward_eager_chain.py", body)
    violations = _check_file(tmp_path, path)
    assert [(item.name, item.line) for item in violations] == [("third", 11)]


def test_unused_helper_class_does_not_rescue_method_dependencies(
    tmp_path: Path,
) -> None:
    body = """\
    def helper():
        return 1

    class Support:
        def value(self):
            return helper()
    """
    assert _reported(tmp_path, "test_helper_class.py", body) == {"Support", "helper"}


def test_multiline_assignment_reads_the_previous_binding(tmp_path: Path) -> None:
    body = """\
    helper = 1
    helper = (
        helper + 1
    )

    def test_helper():
        assert helper == 2
    """
    assert _reported(tmp_path, "test_multiline_binding.py", body) == set()


def test_default_pytest_filename_pattern_is_scanned(tmp_path: Path) -> None:
    _write(tmp_path, "sample_test.py", "helper = 1\n")

    assert [path.name for path in _test_files(tmp_path)] == ["sample_test.py"]


def test_pattern_capture_shadows_module_helper(tmp_path: Path) -> None:
    body = """\
    helper = 1

    def test_capture():
        match 2:
            case helper:
                assert helper == 2
    """
    assert "helper" in _reported(tmp_path, "test_pattern_capture.py", body)


def test_walrus_in_comprehension_binds_containing_scope(tmp_path: Path) -> None:
    body = """\
    helper = 1
    values = (2,)

    def test_walrus():
        [(helper := value) for value in values]
        assert helper == 2
    """
    assert "helper" in _reported(tmp_path, "test_walrus.py", body)


def test_class_method_resolves_same_named_module_helper(tmp_path: Path) -> None:
    body = """\
    def helper() -> int:
        return 3

    class TestCase:
        def helper(self) -> int:
            return helper()

        def test_helper(self) -> None:
            assert self.helper() == 3
    """
    assert _reported(tmp_path, "test_method_scope.py", body) == set()


def test_autouse_fixture_is_referenced_implicitly(tmp_path: Path) -> None:
    body = """\
    import pytest

    @pytest.fixture(autouse=True)
    def setup_environment():
        return 1
    """
    assert _reported(tmp_path, "test_autouse.py", body) == set()


def test_pytest_plugins_is_a_pytest_owned_module_declaration(tmp_path: Path) -> None:
    body = """\
    pytest_plugins = ("tests.plugin",)
    """
    assert _reported(tmp_path, "test_plugin.py", body) == set()


def test_keyword_parametrize_argnames_do_not_request_fixtures(
    tmp_path: Path,
) -> None:
    body = """\
    import pytest

    @pytest.fixture
    def item():
        return 1

    @pytest.mark.parametrize(argnames="item", argvalues=[2])
    def test_item(item):
        assert item == 2
    """
    assert _reported(tmp_path, "test_keyword_parametrize.py", body) == {"item"}


def test_class_parametrize_argnames_do_not_request_method_fixtures(
    tmp_path: Path,
) -> None:
    body = """\
    import pytest

    @pytest.fixture
    def item():
        return 1

    @pytest.mark.parametrize("item", [2])
    class TestCase:
        def test_item(self, item):
            assert item == 2
    """
    assert _reported(tmp_path, "test_class_parametrize.py", body) == {"item"}


def test_nested_test_named_method_is_not_collected_from_local_class(
    tmp_path: Path,
) -> None:
    body = """\
    import pytest

    @pytest.fixture
    def unused():
        return 1

    def test_outer():
        class Helper:
            def test_nested(self, unused):
                assert unused == 1
    """
    assert _reported(tmp_path, "test_local_class.py", body) == {"unused"}


def test_empty_class_loop_preserves_module_helper_fallback(tmp_path: Path) -> None:
    body = """\
    def helper():
        return 1

    class TestCase:
        for _ in ():
            helper = 2
        value = helper()

        def test_value(self):
            assert self.value == 1
    """
    assert _reported(tmp_path, "test_class_loop_fallback.py", body) == set()


def test_nested_function_load_depends_on_its_outer_helper(tmp_path: Path) -> None:
    body = """\
    def helper():
        return 1

    def outer():
        def inner():
            return helper()
        return inner
    """
    assert _reported(tmp_path, "test_nested_helper.py", body) == {"outer", "helper"}


def test_fixture_alias_can_use_a_module_string_constant(tmp_path: Path) -> None:
    body = """\
    import pytest

    FIXTURE_NAME = "item"

    @pytest.fixture(name=FIXTURE_NAME)
    def item_fixture():
        return 1

    def test_item(item):
        assert item == 1
    """
    assert _reported(tmp_path, "test_fixture_constant_alias.py", body) == set()


def test_dynamic_fixture_alias_is_kept_live_conservatively(tmp_path: Path) -> None:
    body = """\
    import pytest

    def fixture_name():
        return "item"

    @pytest.fixture(name=fixture_name())
    def item_fixture():
        return 1
    """
    assert _reported(tmp_path, "test_fixture_dynamic_alias.py", body) == set()


def test_fixture_method_dependencies_keep_module_fixtures_live(tmp_path: Path) -> None:
    body = """\
    import pytest

    @pytest.fixture
    def inner():
        return 1

    class TestCase:
        @pytest.fixture
        def outer(self, inner):
            return inner

        def test_value(self, outer):
            assert outer == 1
    """
    assert _reported(tmp_path, "test_fixture_method.py", body) == set()


def test_module_callable_alias_is_collected_as_a_test(tmp_path: Path) -> None:
    body = """\
    def check():
        pass

    test_alias = check
    """
    assert _reported(tmp_path, "test_callable_test_alias.py", body) == set()


def test_constant_parametrize_argnames_do_not_request_fixtures(
    tmp_path: Path,
) -> None:
    body = """\
    import pytest

    ARG = "item"

    @pytest.fixture
    def item():
        return 1

    @pytest.mark.parametrize(ARG, [2])
    def test_item(item):
        assert item == 2
    """
    assert _reported(tmp_path, "test_constant_parametrize.py", body) == {"item"}


def test_class_try_handler_preserves_module_helper_fallback(tmp_path: Path) -> None:
    body = """\
    def helper():
        return 1

    class TestCase:
        try:
            raise ValueError
            helper = 2
        except ValueError:
            pass
        value = helper()

        def test_value(self):
            assert self.value == 1
    """
    assert _reported(tmp_path, "test_class_try_fallback.py", body) == set()


def test_class_body_bindings_apply_in_execution_order(tmp_path: Path) -> None:
    body = """\
    def helper() -> int:
        return 3

    class TestCase:
        value = helper()
        helper = 2
    """
    assert _reported(tmp_path, "test_class_order.py", body) == set()


def test_self_recursion_does_not_rescue_unused_helper(tmp_path: Path) -> None:
    body = """\
    def helper() -> int:
        return helper()
    """
    assert _reported(tmp_path, "test_self_recursive.py", body) == {"helper"}


def test_mutually_recursive_helpers_without_a_root_are_dead(tmp_path: Path) -> None:
    body = """\
    def first():
        return second()

    def second():
        return first()
    """
    assert _reported(tmp_path, "test_recursive_cycle.py", body) == {"first", "second"}


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


def test_waiver_comment_inside_body_does_not_exempt_definition(
    tmp_path: Path,
) -> None:
    body = """\
    def kept() -> int:
        # dead-code: unrelated comment inside the body
        return 1
    """
    assert _reported(tmp_path, "test_body_comment.py", body) == {"kept"}


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
