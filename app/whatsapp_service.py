"""Compatibility module alias; intake use cases are in application.ingestion."""

import sys

from .application import ingestion as _implementation

sys.modules[__name__] = _implementation
