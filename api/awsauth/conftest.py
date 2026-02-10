"""Prevent pytest from importing the parent api/__init__.py.

The parent api/ package has heavy dependencies (pydantic, libs) that are not
needed for the end-to-end tests which only use the requests library.  By
pre-registering a stub 'api' module we stop Python's import machinery from
executing api/__init__.py.
"""
import sys
import types

sys.modules.setdefault("api", types.ModuleType("api"))
