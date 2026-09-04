from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pytest

import jacobian
from jacobian.process import (
    run_bounded_process,
    run_bounded_worker_dialogue,
    worker_environment,
)


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
    monkeypatch.setenv("JACOBIAN_OPTED_IN_VARIABLE", "/authorized/value")
    monkeypatch.setenv("JACOBIAN_TEST_SECRET", "do-not-forward")

    environment = worker_environment(extra_variables=("JACOBIAN_OPTED_IN_VARIABLE",))

    assert environment["JACOBIAN_OPTED_IN_VARIABLE"] == "/authorized/value"
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


@pytest.mark.parametrize("dialogue", [False, True])
def test_python_worker_imports_host_package_without_site_or_ambient_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dialogue: bool
) -> None:
    # -S removes editable-install/site-packages assistance, exposing source-only
    # startup failures even when pytest itself runs in an installed environment.
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "untrusted"))
    monkeypatch.setenv("JACOBIAN_TEST_SECRET", "do-not-forward")
    environment = worker_environment()
    command = [
        sys.executable,
        "-S",
        "-c",
        "import json, os, jacobian; "
        "print(json.dumps([jacobian.__file__, dict(os.environ)]), flush=True)",
    ]
    if dialogue:
        completed = run_bounded_worker_dialogue(
            command,
            lambda child: child.read_until(b"\n", frame_limit=8192),
            absolute_deadline=time.monotonic() + 10,
            environment=environment,
            stdout_limit=8192,
            stderr_limit=8192,
            cwd=str(tmp_path),
        )
        output = completed.value
    else:
        result = run_bounded_process(
            command,
            input_bytes=b"",
            timeout_seconds=10,
            environment=environment,
            stdout_limit=8192,
            stderr_limit=8192,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0, result.stderr
        assert not result.timed_out
        output = result.stdout
    package_file, child_environment = json.loads(output)
    assert Path(package_file).resolve() == Path(jacobian.__file__).resolve()
    assert child_environment["PYTHONPATH"] == str(
        Path(jacobian.__file__).resolve().parent.parent
    )
    assert "JACOBIAN_TEST_SECRET" not in child_environment
    assert "HOME" not in child_environment
    assert "PATH" not in child_environment
    assert "PYTHONPATH" not in environment


@pytest.mark.parametrize("pythonpath", ["", "/explicit/package/path"])
def test_python_worker_preserves_explicit_pythonpath(pythonpath: str) -> None:
    environment = worker_environment(overrides={"PYTHONPATH": pythonpath})
    result = run_bounded_process(
        [sys.executable, "-S", "-c", "import os; print(os.environ['PYTHONPATH'])"],
        input_bytes=b"",
        timeout_seconds=10,
        environment=environment,
        stdout_limit=8192,
        stderr_limit=8192,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.decode().rstrip("\n") == pythonpath
    assert environment["PYTHONPATH"] == pythonpath


def test_non_python_command_does_not_receive_package_path() -> None:
    executable = shutil.which("env")
    if executable is None:
        pytest.skip("env executable is unavailable")
    environment = worker_environment()
    result = run_bounded_process(
        [executable],
        input_bytes=b"",
        timeout_seconds=10,
        environment=environment,
        stdout_limit=8192,
        stderr_limit=8192,
    )
    assert result.returncode == 0, result.stderr
    assert dict(line.split("=", 1) for line in result.stdout.decode().splitlines()) == (
        environment
    )
