import sys

import pytest


@pytest.fixture(autouse=True)
def _isolate_module_stubs():
    # Several tests install lightweight stubs (diffusers, yaml, ...) into
    # sys.modules. Drop those stubs afterwards so they never leak into later
    # tests, while keeping real modules imported meanwhile (torchvision cannot
    # be imported twice in one process).
    before = dict(sys.modules)
    yield
    for name, module in list(sys.modules.items()):
        if module is before.get(name) or getattr(module, "__file__", None):
            continue
        if name in before:
            sys.modules[name] = before[name]
        else:
            del sys.modules[name]
