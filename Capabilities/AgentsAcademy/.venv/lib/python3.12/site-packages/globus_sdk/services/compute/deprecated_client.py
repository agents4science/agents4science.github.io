from __future__ import annotations

import sys
import typing as t

from globus_sdk.exc import warn_deprecated

from .client import ComputeClientV2

__all__ = ("ComputeClient",)

if t.TYPE_CHECKING:

    class ComputeClient(ComputeClientV2):
        pass

else:

    def __getattr__(name: str) -> t.Any:
        if name == "ComputeClient":
            warn_deprecated(
                "'globus_sdk.ComputeClient' is deprecated and will be removed "
                "in the future. Prefer 'globus_sdk.ComputeClientV2'."
            )

            class ComputeClient(ComputeClientV2):
                """A deprecated alias for 'globus_sdk.ComputeClientV2'."""

            setattr(sys.modules[__name__], name, ComputeClient)
            return ComputeClient

        raise AttributeError(f"module {__name__} has no attribute {name}")
