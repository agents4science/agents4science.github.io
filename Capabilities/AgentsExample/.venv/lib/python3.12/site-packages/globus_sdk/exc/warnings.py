from __future__ import annotations

import warnings


class RemovedInV5Warning(DeprecationWarning):
    """
    This warning indicates that a feature or usage was detected which will be
    unsupported in globus-sdk version 5.

    Users are encouraged to resolve these warnings when possible.
    """


def warn_deprecated(message: str, stacklevel: int = 2) -> None:
    warnings.warn(message, RemovedInV5Warning, stacklevel=stacklevel)
