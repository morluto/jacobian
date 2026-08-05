from __future__ import annotations

import pytest

from jacobian.worker_environment import worker_environment


def test_worker_environment_does_not_forward_parent_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JACOBIAN_TEST_SECRET", "do-not-forward")
    monkeypatch.setenv("HTTPS_PROXY", "http://secret.invalid")

    environment = worker_environment()

    assert "JACOBIAN_TEST_SECRET" not in environment
    assert "HTTPS_PROXY" not in environment
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["TZ"] == "UTC"


def test_worker_environment_omits_ambient_path_pythonpath_and_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/ambient/host/path")
    monkeypatch.setenv("PYTHONPATH", "/ambient/host/pythonpath")
    monkeypatch.setenv("HOME", "/ambient/host/home")

    environment = worker_environment()

    assert "PATH" not in environment
    assert "PYTHONPATH" not in environment
    assert "HOME" not in environment


def test_worker_environment_omits_proxies_credentials_and_arbitrary_host_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid")
    monkeypatch.setenv("https_proxy", "http://proxy.invalid")
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ambient-credential")
    monkeypatch.setenv("JACOBIAN_ARBITRARY_HOST_VAR", "ambient-value")
    monkeypatch.setenv("TMPDIR", "/ambient/host/tmp")
    monkeypatch.setenv("LC_CTYPE", "C")

    environment = worker_environment()

    assert "HTTP_PROXY" not in environment
    assert "https_proxy" not in environment
    assert "NO_PROXY" not in environment
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert "JACOBIAN_ARBITRARY_HOST_VAR" not in environment
    assert "TMPDIR" not in environment
    assert "LC_CTYPE" not in environment


def test_path_prefix_builds_toolchain_only_path_without_ambient_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/ambient/host/path")

    environment = worker_environment(path_prefix="/authorized/toolchain/bin")

    assert environment["PATH"] == "/authorized/toolchain/bin"
    assert "/ambient/host/path" not in environment["PATH"]


def test_extra_variables_explicitly_opts_host_values_back_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ELAN_HOME", "/authorized/elan")
    monkeypatch.setenv("JACOBIAN_TEST_SECRET", "do-not-forward")

    environment = worker_environment(extra_variables=("ELAN_HOME",))

    assert environment["ELAN_HOME"] == "/authorized/elan"
    assert "JACOBIAN_TEST_SECRET" not in environment


def test_overrides_take_precedence_over_defaults_and_path_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/ambient/host/path")

    environment = worker_environment(
        path_prefix="/authorized/toolchain/bin",
        overrides={"PATH": "/override/bin", "LANG": "C"},
    )

    assert environment["PATH"] == "/override/bin"
    assert environment["LANG"] == "C"
    assert environment["LC_ALL"] == "C.UTF-8"
    assert environment["TZ"] == "UTC"


def test_locale_sets_lang_and_lc_all() -> None:
    environment = worker_environment(locale="C")

    assert environment["LANG"] == "C"
    assert environment["LC_ALL"] == "C"
