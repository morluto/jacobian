from __future__ import annotations

from pathlib import Path

from tools.check_architecture import check_architecture


def _write_source(
    root: Path,
    source: str,
    relative: str = "src/jacobian/adapter.py",
) -> None:
    path = root / relative
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")


def test_architecture_rejects_direct_canonical_rational_text_conversions(
    tmp_path: Path,
) -> None:
    source = (
        "def convert(value):\n"
        "    numerator = int(value.num)\n"
        "    denominator = int(value.den, 10)\n"
        "    text = str(value.num)\n"
        "    return numerator, denominator, text\n"
    )
    _write_source(tmp_path, source)
    _write_source(tmp_path, source, "src/jacobian_checkers/checker.py")

    violations = [
        violation
        for violation in check_architecture(tmp_path).violations
        if violation.code == "unsafe-canonical-conversion"
    ]

    assert len(violations) == 6
    assert {violation.line for violation in violations} == {2, 3, 4}
    assert {violation.path for violation in violations} == {
        "src/jacobian/adapter.py",
        "src/jacobian_checkers/checker.py",
    }
    assert all("canonical conversion API" in item.message for item in violations)


def test_architecture_accepts_canonical_conversion_api(tmp_path: Path) -> None:
    _write_source(
        tmp_path,
        "def convert(value):\n"
        "    fraction = value.as_fraction()\n"
        "    return fraction.numerator, fraction.denominator\n",
    )

    report = check_architecture(tmp_path)

    assert all(
        violation.code != "unsafe-canonical-conversion"
        for violation in report.violations
    )
