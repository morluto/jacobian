"""Minimal deterministic environment for local worker subprocesses.

No host variable is forwarded by default.  ``PATH``, ``PYTHONPATH``, ``HOME``,
proxy variables, credential variables, locale variables, and arbitrary host
variables are all absent unless a caller explicitly authorizes them through
*extra_variables* or *overrides*.  *path_prefix* constructs ``PATH`` solely
from the supplied authorized directories; the ambient host ``PATH`` is never
appended.
"""

from __future__ import annotations

import os

_DEFAULT_LOCALE = "C.UTF-8"


def worker_environment(
    *,
    extra_variables: tuple[str, ...] = (),
    overrides: dict[str, str] | None = None,
    path_prefix: str | None = None,
    locale: str = _DEFAULT_LOCALE,
) -> dict[str, str]:
    """Return a deterministic worker environment with no ambient host leakage.

    The returned mapping contains only the deterministic defaults
    (``PYTHONHASHSEED``, ``PYTHONDONTWRITEBYTECODE``, ``TZ``, ``LANG``,
    ``LC_ALL``).  *extra_variables* explicitly opts specific host variables
    back in, copied from the host environment only when present; this is an
    explicit caller authorization, not a default.  *overrides* take
    precedence over both the defaults and any opted-in host values.
    *path_prefix* becomes the whole ``PATH`` (a toolchain-only path); the
    ambient host ``PATH`` is never appended.  *locale* sets ``LANG`` and
    ``LC_ALL`` (default ``C.UTF-8``).
    """

    environment: dict[str, str] = {
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TZ": "UTC",
        "LANG": locale,
        "LC_ALL": locale,
    }
    for name in extra_variables:
        value = os.environ.get(name)
        if value is not None:
            environment[name] = value
    if path_prefix:
        environment["PATH"] = path_prefix
    if overrides:
        environment.update(overrides)
    return environment
