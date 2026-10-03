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

_OUT = os.environ.get("MATCH_RECORDER_OUT")
RECORD: dict[str, dict[str, object]] = {}
_original_raises = pytest.raises

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
        self._ctx = _original_raises(expected, *args, **kwargs)  # type: ignore[call-overload]
        self.excinfo = None

    def __enter__(self) -> object:
        self.excinfo = self._ctx.__enter__()
        return self.excinfo

    def __exit__(self, exc_type: object, exc: object, tb: object) -> object:
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
                if isinstance(reported, (list, tuple)) and reported:
                    first = reported[0]
                    if isinstance(first, dict):
                        record["code"] = first.get("type")
            reason = getattr(exc, "reason", None)
            if record["code"] is None and isinstance(reason, str):
                record["reason"] = reason
            if self._key:
                RECORD[self._key] = merge_record(RECORD.get(self._key), record)
        return self._ctx.__exit__(exc_type, exc, tb)


def merge_record(
    previous: dict[str, object] | None, record: dict[str, object]
) -> dict[str, object]:
    """Union two observations of one site, keeping code-less executions.

    A site executed more than once may raise different codes (parametrized
    cases, loops, or separate xdist workers). Every observed code is kept so
    the rewriter can refuse an ambiguous single-code assert. ``None`` is an
    explicit member meaning "this execution raised no owner code" and must
    survive the merge: dropping it is what let a coded observation hide an
    earlier code-less one.

    Observed ``match`` texts are unioned the same way. A parametrized site
    asserting two different messages behind one owner code is guarded only by
    its wording, so retaining the first message would let the rewriter delete
    the sole distinction between those guards.
    """

    if previous is None:
        return record
    observed = previous.get("codes")
    codes: set[str | None] = set(observed) if isinstance(observed, list) else set()
    codes.add(_as_code(previous.get("code")))
    codes.add(_as_code(record.get("code")))
    merged = dict(previous)
    merged["codes"] = sorted(codes, key=lambda item: (item is not None, item or ""))
    merged["code"] = next(iter(codes)) if len(codes) == 1 else None
    matches = _observed_matches(previous, record)
    if matches:
        merged["matches"] = matches
        if previous.get("match") is not None:
            merged["match"] = previous["match"]
    return merged


def _observed_matches(
    previous: dict[str, object], record: dict[str, object]
) -> list[str]:
    """Return every ``match`` text observed for one site, sorted and unique."""

    matches: set[str] = set()
    for source in (previous, record):
        seen = source.get("matches")
        if isinstance(seen, list):
            matches.update(item for item in seen if isinstance(item, str) and item)
        match = source.get("match")
        if isinstance(match, str) and match:
            matches.add(match)
    return sorted(matches)


def _as_code(value: object) -> str | None:
    """Normalise a recorded code; ``None`` marks a code-less execution.

    A site that raises a code-less error in one parametrization and a coded
    error in another cannot be pinned to a single assert, so a code-less
    execution must survive the merge as an explicit ``None``.
    """

    return value if isinstance(value, str) and value else None


def _site_key() -> str | None:
    """Identify the ``pytest.raises`` call site from its caller frame."""

    try:
        frame = sys._getframe(2)
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


pytest.raises = _RecordingRaises
