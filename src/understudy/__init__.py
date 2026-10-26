"""The name this project had before 0.5.0, kept for one release so that code importing
`understudy` and `python -m understudy.cli` keep working. Use `ditto_code`."""
import importlib
import sys

from ditto_code import __version__  # noqa: F401

for _name in ("claude", "ladder", "memory", "proxy"):
    _module = importlib.import_module(f"ditto_code.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module

from . import cli  # noqa: E402  (a real module, so `python -m understudy.cli` runs)
