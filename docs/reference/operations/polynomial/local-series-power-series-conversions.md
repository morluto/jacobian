# Power-series and Laurent conversions

`local_series.from_power_series.compute` embeds the exact #1713
`TruncatedSeries` value in `TruncatedLaurentWindow` at finite center zero. It
preserves the variable and exclusive precision. Known leading zeros are
normalized into `valuation_lower`; an all-zero prefix remains the canonical
zero window.

`local_series.to_power_series.compute` converts back only at finite center
zero, with a common precision from 1 through 512. It fills the known
nonnegative coefficient positions into #1713's dense carrier. If a known
coefficient at a negative exponent is nonzero, the result has status
`HAS_NEGATIVE_EXPONENTS` and records the first such exponent instead of
truncating the pole. A zero prefix with explicitly zero negative slots converts
to the zero power series. Infinity and nonzero finite centers have no implicit
map to #1713's `QQ[[x]]/(x^N)` parent and are rejected.

Both directions retain the variable name and precision. They admit order at
most 512 and coefficient components at most 256 decimal digits before copying
the dense tuple.
