#!/usr/bin/env python3

import importlib.util
import sys
from types import ModuleType
from typing import Optional


def lazy_load_module_if_available(name: str) -> Optional[ModuleType]:
    """Import a module if available, otherwise return None."""
    # If it's already loaded, no need to re-import
    if name in sys.modules:
        return sys.modules[name]

    # Can we find the module?
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    if not hasattr(spec, "loader") or spec.loader is None:
        return None

    # Lazy load, and put in sys.modules
    module = importlib.util.module_from_spec(spec)
    loader = importlib.util.LazyLoader(spec.loader)
    loader.exec_module(module)
    sys.modules[name] = module
    return module
