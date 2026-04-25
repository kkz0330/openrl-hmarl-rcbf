from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


_OPENRL_BAD_OPTION_NAMES = {
    "actor_lr",
    "auto_alph",
    "alpha_value",
    "alpha_lr",
}


@contextmanager
def _patched_jsonargparse_add_argument() -> Iterator[None]:
    import jsonargparse

    original_add_argument = jsonargparse.ArgumentParser.add_argument

    def patched_add_argument(self, *args, **kwargs):
        if args and isinstance(args[0], str) and args[0] in _OPENRL_BAD_OPTION_NAMES:
            args = (f"--{args[0]}",) + args[1:]
        return original_add_argument(self, *args, **kwargs)

    jsonargparse.ArgumentParser.add_argument = patched_add_argument
    try:
        yield
    finally:
        jsonargparse.ArgumentParser.add_argument = original_add_argument


def create_openrl_config_parser():
    from openrl.configs.config import create_config_parser as create_parser

    with _patched_jsonargparse_add_argument():
        return create_parser()


def parse_openrl_default_config():
    return create_openrl_config_parser().parse_args([])
