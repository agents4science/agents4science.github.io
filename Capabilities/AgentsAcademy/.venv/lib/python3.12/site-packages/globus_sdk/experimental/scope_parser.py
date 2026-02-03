"""
Scope parsing has been moved out of experimental into the main SDK.
This module will be removed in a future release but is maintained in the interim for
  backwards compatibility.
"""

from __future__ import annotations

import sys
import typing as t

__all__ = (
    "Scope",
    "ScopeParseError",
    "ScopeCycleError",
)

if t.TYPE_CHECKING:
    from globus_sdk.scopes import Scope, ScopeCycleError, ScopeParseError
else:

    def __getattr__(name: str) -> t.Any:
        import globus_sdk.scopes as scopes_module
        from globus_sdk.exc import warn_deprecated

        warn_deprecated(
            "'globus_sdk.experimental.scope_parser' has been merged into "
            "'globus_sdk.scopes'. "
            f"Importing '{name}' from `globus_sdk.experimental` is deprecated. "
            f"Use `globus_sdk.scopes.{name}` instead."
        )

        value = getattr(scopes_module, name, None)
        if value is None:
            raise AttributeError(f"module {__name__} has no attribute {name}")
        setattr(sys.modules[__name__], name, value)
        return value
