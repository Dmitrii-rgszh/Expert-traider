"""Compatibility shim for the deprecated ``multipart`` import path.

FastAPI/Starlette still import ``multipart`` (the legacy name of the
``python-multipart`` package). Recent versions of the upstream library emit a
``PendingDeprecationWarning`` when this happens. Placing this shim earlier in the
``sys.path`` re-exports the modern package without triggering the warning.
"""
from python_multipart import *  # type: ignore

from python_multipart import (  # type: ignore
	__all__ as _ALL,
	__author__ as _AUTHOR,
	__copyright__ as _COPYRIGHT,
	__license__ as _LICENSE,
	__version__ as _VERSION,
)

__all__ = _ALL  # type: ignore
__author__ = _AUTHOR  # type: ignore
__copyright__ = _COPYRIGHT  # type: ignore
__license__ = _LICENSE  # type: ignore
__version__ = _VERSION  # type: ignore
