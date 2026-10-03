"""Record the live error code behind every ``pytest.raises(..., match=...)`` site.

Loading this file as a pytest plugin (``-p tools.match_recorder``) replaces
``pytest.raises`` with a recorder that captures, for each call site, the
exception class and the code the owner actually publishes. The companion
``rewrite_match_pins.py`` then rewrites a site only when a *specific* code was
observed, so a code-less or generic rejection keeps its message match instead of
silently becoming ``pytest.raises(ValueError)``.

Emits JSON keyed by ``"<file>:<line>"`` when ``MATCH_RECORDER_OUT`` is set.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from tools.match_records import merge_record

_OUT = os.environ.get("MATCH_RECORDER_OUT")
RECORD: dict[str, dict[str, object]] = {}
_original_raises = None

# Codes too coarse to distinguish the guards they cover. A site whose only
# observed code is one of these keeps its message match.
GENERIC_CODES = frozenset(
    {
        "value_error",
        "polynomial.invariant",
        "probability.model_invariant",
        "polynomial.multivariate_contract",
        "assertion_error",
    }
)


class _RecordingRaises:
    """Wrap ``pytest.raises`` and capture what the owner actually raised."""

    def __init__(self, expected: object, *args: object, **kwargs: object) -> None:
        self._expected = expected
        self._match = kwargs.get("match", args[1] if len(args) > 1 else None)
        self._key = _site_key()
        if _original_raises is None:
            raise RuntimeError("match recorder is not configured")
        self._ctx = _original_raises(expected, *args, **kwargs)  # type: ignore[call-overload]
        self.excinfo = None

    def __enter__(self) -> object:
        self.excinfo = self._ctx.__enter__()
        return self.excinfo

    def __exit__(self, exc_type: object, exc: object, tb: object) -> object:
        accepted = self._ctx.__exit__(exc_type, exc, tb)
        if exc is not None:
            record: dict[str, object] = {
                "expected": getattr(self._expected, "__name__", str(self._expected)),
                "match": self._match,
                "cls": type(exc).__name__,
                "code": None,
            }
            errors = getattr(exc, "errors", None)
            if callable(errors):
                try:
                    reported = errors()
                except Exception:
                    reported = None
                # Owner errors may return a list or a tuple of mappings.
                if isinstance(reported, (list, tuple)) and len(reported) == 1:
                    first = reported[0]
                    if isinstance(first, dict):
                        record["code"] = first.get("type")
            reason = getattr(exc, "reason", None)
            if record["code"] is None and isinstance(reason, str):
                record["reason"] = reason
            if self._key:
                RECORD[self._key] = merge_record(RECORD.get(self._key), record)
        return accepted


def _site_key() -> str | None:
    """Identify the ``pytest.raises`` call site from its caller frame."""

    try:
        frame = sys._getframe(3)
    except ValueError:
        return None
    filename = frame.f_code.co_filename
    return f"{os.path.relpath(filename)}:{frame.f_lineno}"


def pytest_sessionfinish(session, exitstatus):
    if not _OUT:
        return
    serialisable = {
        key: value for key, value in RECORD.items() if value.get("match") is not None
    }
    with Path(_OUT).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(serialisable) + "\n")


def _recording_raises(expected_exception=None, *args, **kwargs):
    """Preserve pytest's function form and optional keyword-only forms."""

    if args or "func" in kwargs:
        if _original_raises is None:
            raise RuntimeError("match recorder is not configured")
        return _original_raises(expected_exception, *args, **kwargs)
    return _RecordingRaises(expected_exception, *args, **kwargs)


def pytest_configure(config):
    global _original_raises
    _original_raises = pytest.raises
    pytest.raises = _recording_raises


def pytest_unconfigure(config):
    global _original_raises
    if _original_raises is not None:
        pytest.raises = _original_raises
        _original_raises = None
