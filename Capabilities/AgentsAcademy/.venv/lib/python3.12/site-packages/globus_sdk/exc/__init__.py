import importlib
import sys
import typing as t

from .api import ErrorSubdocument, GlobusAPIError
from .base import GlobusError, GlobusSDKUsageError, ValidationError
from .err_info import (
    AuthorizationParameterInfo,
    ConsentRequiredInfo,
    ErrorInfo,
    ErrorInfoContainer,
)
from .warnings import RemovedInV4Warning, warn_deprecated

__all__ = (
    "GlobusError",
    "GlobusSDKUsageError",
    "ValidationError",
    "GlobusAPIError",
    "ErrorSubdocument",
    "NetworkError",
    "GlobusTimeoutError",
    "GlobusConnectionTimeoutError",
    "GlobusConnectionError",
    "convert_request_exception",
    "ErrorInfo",
    "ErrorInfoContainer",
    "AuthorizationParameterInfo",
    "ConsentRequiredInfo",
    "RemovedInV4Warning",
    "warn_deprecated",
)

# imports from `globus_sdk.exc.convert` are done lazily
#
# this ensures that we do not eagerly import `requests` when attempting to use SDK
# components which do not need it, but which do need errors (e.g., RemovedInV4Warning)
# and we avoid paying the performance penalty for importing the relevant dependencies
if t.TYPE_CHECKING:
    from .convert import (
        GlobusConnectionError,
        GlobusConnectionTimeoutError,
        GlobusTimeoutError,
        NetworkError,
        convert_request_exception,
    )
else:
    _LAZY_IMPORT_TABLE = {
        "convert": {
            "GlobusConnectionError",
            "GlobusConnectionTimeoutError",
            "GlobusTimeoutError",
            "NetworkError",
            "convert_request_exception",
        }
    }

    def __getattr__(name: str) -> t.Any:
        for modname, items in _LAZY_IMPORT_TABLE.items():
            if name in items:
                mod = importlib.import_module("." + modname, __name__)
                value = getattr(mod, name)
                setattr(sys.modules[__name__], name, value)
                return value

        raise AttributeError(f"module {__name__} has no attribute {name}")
