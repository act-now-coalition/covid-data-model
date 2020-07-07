from typing import Dict, Iterator, Type, Tuple, List
import pathlib
import importlib
import sys
import logging

import pydantic
from libs import base_model

_logger = logging.getLogger(__name__)

SCHEMAS_PATH = pathlib.Path(__file__).parent / "schemas"


def find_public_model_classes() -> List[Type[pydantic.BaseModel]]:
    """Finds all model classes (i.e. that derive from pydantic.BasModel) for export.

    If a class is prefixed with a '_' (i.e. _MySchema), it will not be exported
    as a public schema.

    Performs a bit of python magic to load all modules in the in the `api/` folder
    and find all subclasses of `pydantic.BaseModel`.

    Returns: List of api model classes.
    """
    # Find all python files in api/ and load them as global modules.
    # This allows us to then enumerate pydantic.BaseModel.__subclasses__
    # to find all our model classes.
    root = pathlib.Path(__file__).parent
    for path in root.rglob("*.py"):
        relative = path.relative_to(root.parent)
        module_name = str(relative).replace("\\", "/").replace("/", ".")[:-3]

        spec = importlib.util.spec_from_file_location(module_name, str(path))
        mod = importlib.util.module_from_spec(spec)
        # If a module has already been imported don't reimport.
        if spec.name not in sys.modules:
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)

    model_classes = []
    for subclass in base_model.BaseModel.__subclasses__():
        # Skip any internal pydantic models.
        if subclass.__module__.startswith("pydantic"):
            continue
        if subclass.__name__.startswith("_"):
            _logger.debug(f"Skipping private model: {subclass}")
            continue
        model_classes.append(subclass)

    return model_classes
