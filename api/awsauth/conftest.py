"""Prevent pytest from importing the parent api/__init__.py.

The parent api/ package has heavy dependencies (pydantic, libs) that are not
needed for the end-to-end tests which only use the requests library.  When
those dependencies are missing we pre-register a stub 'api' module so that
Python's import machinery does not attempt to execute api/__init__.py.

When running from the full repo with all dependencies installed (e.g. CI
lint-and-test), the real api package imports fine and the stub is never used.
"""

import sys
import types

try:
    import api  # noqa: F401  – succeeds when full deps are installed
except (ImportError, ModuleNotFoundError):
    sys.modules.setdefault("api", types.ModuleType("api"))
