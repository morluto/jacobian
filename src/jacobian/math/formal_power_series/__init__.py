"""Supported exact truncated formal-power-series API."""

from jacobian.math.formal_power_series._models import TruncatedSeries
from jacobian.math.formal_power_series._operations import (
    compute_add as add,
)
from jacobian.math.formal_power_series._operations import (
    compute_compose as compose,
)
from jacobian.math.formal_power_series._operations import (
    compute_derivative as derivative,
)
from jacobian.math.formal_power_series._operations import (
    compute_divide as divide,
)
from jacobian.math.formal_power_series._operations import (
    compute_from_polynomial as from_polynomial,
)
from jacobian.math.formal_power_series._operations import (
    compute_identity_check as identity_check,
)
from jacobian.math.formal_power_series._operations import (
    compute_integral as integral_zero_constant,
)
from jacobian.math.formal_power_series._operations import (
    compute_inverse as inverse,
)
from jacobian.math.formal_power_series._operations import (
    compute_multiply as multiply,
)
from jacobian.math.formal_power_series._operations import (
    compute_power as power,
)
from jacobian.math.formal_power_series._operations import (
    compute_reversion as reversion,
)
from jacobian.math.formal_power_series._operations import (
    compute_scalar_multiply as scalar_multiply,
)
from jacobian.math.formal_power_series._operations import (
    compute_subtract as subtract,
)
from jacobian.math.formal_power_series._operations import (
    compute_to_polynomial as to_polynomial,
)
from jacobian.math.formal_power_series._operations import (
    compute_truncate as truncate,
)

__all__ = [
    "TruncatedSeries",
    "add",
    "compose",
    "derivative",
    "divide",
    "from_polynomial",
    "identity_check",
    "integral_zero_constant",
    "inverse",
    "multiply",
    "power",
    "reversion",
    "scalar_multiply",
    "subtract",
    "to_polynomial",
    "truncate",
]
