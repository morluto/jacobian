"""Generate the packaged inline operation index at wheel build."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        del version
        src = Path(self.root) / "src"
        dest = src / "jacobian" / "data" / "inline_index.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(src))
        try:
            from jacobian.package_index import write_package_index

            write_package_index(dest)
        except ImportError:
            if not dest.is_file():
                raise RuntimeError(
                    "package index is missing and Jacobian cannot be imported "
                    "in the isolated build environment"
                ) from None
        force_include = build_data.setdefault("force_include", {})
        force_include[str(dest)] = "jacobian/data/inline_index.json"
