"""Owner-local CI policy tests split from test_ci_execution_policy.py."""

from __future__ import annotations

import argparse
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from pre_commit.clientlib import load_config
from pre_commit.lang_base import hook_cmd
from pre_commit.parse_shebang import normalize_cmd

ROOT = Path(__file__).parents[2]


def _pull_request_trigger(workflow_path: str) -> str:
    workflow = (ROOT / workflow_path).read_text(encoding="utf-8")
    return workflow.split("  pull_request:", 1)[1].split("  merge_group:", 1)[0]


def test_local_hook_commands_have_parseable_entrypoints_and_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(str(ROOT / ".pre-commit-config.yaml"))
    hooks = [
        hook
        for repo in config["repos"]
        if repo["repo"] == "local"
        for hook in repo["hooks"]
    ]
    assert hooks

    class ArgumentsParsedError(Exception):
        pass

    original_parse_args = argparse.ArgumentParser.parse_args

    def stop_after_parse(
        parser: argparse.ArgumentParser,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> None:
        original_parse_args(parser, args, namespace)
        raise ArgumentsParsedError

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop_after_parse)

    for hook in hooks:
        command = hook_cmd(hook["entry"], hook["args"])
        normalize_cmd(command)
        if hook["id"] == "jacobian-pre-push":
            assert command == ("make", "lint", "typecheck")
            continue

        assert command[:4] == ("uv", "run", "--locked", "python"), hook["id"]
        script_index = 4
        assert script_index < len(command), hook["id"]
        script = (ROOT / command[script_index]).resolve()
        assert script.is_relative_to(ROOT) and script.is_file(), hook["id"]

        namespace = runpy.run_path(str(script))
        main = namespace.get("main")
        assert callable(main), hook["id"]
        monkeypatch.setattr(
            sys,
            "argv",
            [str(script), *command[script_index + 1 :]],
        )
        with pytest.raises(ArgumentsParsedError):
            main()
