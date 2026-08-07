"""Architecture checks for digit-limit-safe exact rational results."""

from __future__ import annotations

from pathlib import Path

from tools.check_architecture import check_architecture


def _write_source(root: Path, source: str) -> None:
    path = root / "src/jacobian/domain.py"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")


def test_direct_rational_result_component_formatting_is_rejected(
    tmp_path: Path,
) -> None:
    _write_source(
        tmp_path,
        "result = CanonicalRational(\n"
        "    num=str(value.numerator),\n"
        "    den=str(value.denominator),\n"
        ")\n"
        'payload = {"num": str(value.p), "den": str(value.q)}\n'
        "nested = CanonicalRational(\n"
        "    num=str(int(value.p)),\n"
        "    den=str(int(value.q)),\n"
        ")\n"
        'formatted = {"num": format(value.numerator), '
        '"den": "{}".format(value.denominator)}\n'
        'text = f"{value.numerator}/{value.denominator}"\n'
        "decimal = format(value.numerator)\n"
        'ratio = "{}".format(value.denominator)\n',
    )

    report = check_architecture(tmp_path)

    violations = [
        violation
        for violation in report.violations
        if violation.code == "unsafe-canonical-rational-output"
    ]
    assert len(violations) == 12


def test_canonical_integer_formatter_is_accepted(tmp_path: Path) -> None:
    _write_source(
        tmp_path,
        "result = CanonicalRational(\n"
        "    num=format_canonical_integer(value.numerator),\n"
        "    den=format_canonical_integer(value.denominator),\n"
        ")\n"
        'text = f"{format_canonical_integer(value.numerator)}/'
        '{format_canonical_integer(value.denominator)}"\n'
        'decimal = "{}".format(format_canonical_integer(value.numerator))\n',
    )

    report = check_architecture(tmp_path)

    assert not any(
        violation.code == "unsafe-canonical-rational-output"
        for violation in report.violations
    )


def test_unrelated_short_attribute_interpolation_is_accepted(tmp_path: Path) -> None:
    _write_source(tmp_path, 'message = f"{point.p},{point.q}"\n')

    report = check_architecture(tmp_path)

    assert not any(
        violation.code == "unsafe-canonical-rational-output"
        for violation in report.violations
    )
