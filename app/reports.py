"""Stable API for workbook persistence/formatting adapter."""

import sys

from .infrastructure import workbooks as _implementation

sys.modules[__name__] = _implementation
