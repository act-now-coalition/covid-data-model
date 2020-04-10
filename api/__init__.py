from typing import Dict, Iterator, Type, Tuple, List
import pathlib
import importlib
import sys
import logging

import pydantic

_logger = logging.getLogger(__name__)

SCHEMAS_PATH = pathlib.Path(__file__).parent / "schemas"



def load_public_schemas() -> List[Type[pydantic.BaseModel]]:
    """Loads all pydantic Schemas for export.

    If a class is prefixed with a '_' (i.e. _MySchema), it will not be exported
    as a public schema.

    Performs a bit of python magic to load all modules in the api/ folder and
    find all subclasses of the base model.

    Returns: List of api schema specifications.
    """
    # Importing all python files in api/ to collect all schemas for public api.
    root = pathlib.Path(__file__).parent
    for path in root.rglob("*.py"):
        relative = path.relative_to(root.parent)
        module_name = str(relative).replace("\\", "/").replace("/", ".")[:-3]

        spec = importlib.util.spec_from_file_location(module_name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            _logger.error(f"Failed to import module at {path}")
            raise

    schemas = []
    for subclass in pydantic.BaseModel.__subclasses__():

        if subclass.__module__.startswith('pydantic'):
            # Skip any pydantic models.
            continue

        if subclass.__name__.startswith('_'):
            _logger.debug(f"Skipping private model: {subclass}")
            continue
        schemas.append(subclass)

    return schemas
