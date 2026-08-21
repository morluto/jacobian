from jacobian._exact import CanonicalRational
from jacobian.math._rational_height import RationalHeight, sum_heights


def _height(num: str, den: str) -> RationalHeight:
    return RationalHeight.from_canonical(CanonicalRational(num=num, den=den))


def test_product_and_quotient_cover_unreduced_component_growth() -> None:
    left = _height("999", "97")
    right = _height("103", "9999")

    assert left.product(right) == RationalHeight(6, 6)
    assert left.quotient(right) == RationalHeight(7, 5)


def test_sum_uses_a_product_common_denominator_bound() -> None:
    result = sum_heights((_height("1", "32"), _height("1", "125")))

    assert result == RationalHeight(5, 5)
