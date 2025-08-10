# commands/__init__.py
import pkgutil
import inspect
import importlib
from .base import Command

__all__ = ['Command']

# Discover and export all Command subclasses
for _, module_name, _ in pkgutil.iter_modules(__path__):
    if module_name not in ['__init__', 'base']:
        # Dynamically import the module
        module = importlib.import_module(f".{module_name}", package=__name__)
        # Find all Command subclasses in the module
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, Command) and obj is not Command:
                __all__.append(name)
