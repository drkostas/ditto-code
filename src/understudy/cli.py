"""`python -m understudy.cli`, the way the proxy was started before 0.5.0. Use `ditto_code.cli`."""
from ditto_code.cli import main, old_name  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(old_name())
