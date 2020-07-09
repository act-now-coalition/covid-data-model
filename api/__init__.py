from typing import Dict, Iterator, Type, Tuple, List
import pathlib
import importlib
import sys
import logging

from libs import base_model

_logger = logging.getLogger(__name__)

SCHEMAS_PATH = pathlib.Path(__file__).parent / "schemas"


def _get_subclasses_recursively(cls) -> Iterator[Type]:
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _get_subclasses_recursively(subclass)


def find_public_model_classes() -> List[Type[base_model.APIBaseModel]]:
    """Finds all model classes (i.e. that derive from base_model.APIBaseModel) for export.

    Performs a bit of python magic to load all modules in the in the `api/` folder
    and find all subclasses of `base_model.APIBaseModel`.

    Returns: List of api model classes.
    """
    # Find all python files in api/ and load them as global modules.
    # This allows us to then enumerate base_model.APIBaseModel.__subclasses__
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

    # Calling `base_model.APIBaseModel.__subclasses__()` only returns direct subclasses.
    # To find all classes that may inherit from subclasses of APIBaseModel, we need to
    # recursively get subclasses.
    for subclass in _get_subclasses_recursively(base_model.APIBaseModel):
        model_classes.append(subclass)

    return model_classes
