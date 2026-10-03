"""Pure helpers for merging pytest.raises recorder observations."""

from __future__ import annotations


def merge_record(
    previous: dict[str, object] | None, record: dict[str, object]
) -> dict[str, object]:
    """Union observations of one site, keeping code-less executions."""

    if previous is None:
        return record
    codes: set[str | None] = set()
    for source in (previous, record):
        observed = source.get("codes")
        if isinstance(observed, list):
            codes.update(_as_code(item) for item in observed)
        codes.add(_as_code(source.get("code")))
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
    """Return every nonempty match text from both observations."""

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
    """Normalize a code; ``None`` marks a code-less execution."""

    return value if isinstance(value, str) and value else None
