from __future__ import annotations

import sys
import typing as t

__all__ = (
    "Consent",
    "ConsentTree",
    "ConsentForest",
    "ConsentParseError",
    "ConsentTreeConstructionError",
)

# legacy aliases
# (when accessed, these will emit deprecation warnings in a future release)
if t.TYPE_CHECKING:
    from globus_sdk.scopes.consents import (
        Consent,
        ConsentForest,
        ConsentParseError,
        ConsentTree,
        ConsentTreeConstructionError,
    )
else:

    def __getattr__(name: str) -> t.Any:
        import globus_sdk.scopes.consents as consents_module
        from globus_sdk.exc import warn_deprecated

        warn_deprecated(
            "'globus_sdk.experimental.consents' has been renamed to "
            "'globus_sdk.scopes.consents'. "
            f"Importing '{name}' from `globus_sdk.experimental` is deprecated."
        )
        value = getattr(consents_module, name, None)
        if value is None:
            raise AttributeError(f"module {__name__} has no attribute {name}")
        setattr(sys.modules[__name__], name, value)
        return value
