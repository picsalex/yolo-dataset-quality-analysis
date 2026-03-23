"""Pipeline step decorators."""

import functools
import logging
import time

from yolo_scout.utils.logger import logger

_BLUE = "\033[34m"
_BOLD_WHITE = "\033[1;37m"
_LIGHT_GRAY = "\033[37m"
_RESET = "\033[0m"

_registry: list = []
_counter: list = [0]
_total: list = [0]


def step(fn=None, *, name=None, level=logging.INFO):
    """Decorator that times and logs a pipeline step. Supports @step and @step(name='...', level=logging.DEBUG)."""

    def decorator(f):
        label = name or f.__name__.replace("_", " ")
        _registry.append((f, level))

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not logger.isEnabledFor(level):
                return f(*args, **kwargs)
            _counter[0] += 1
            index = f"{_LIGHT_GRAY}({_counter[0]}/{_total[0]}){_RESET}"
            logger.log(level, f"\n{_BLUE}Step{_RESET} {_BOLD_WHITE}{label}{_RESET} {index} {_BLUE}has started.{_RESET}")
            t0 = time.perf_counter()
            result = f(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            logger.log(
                level,
                f"{_BLUE}Step{_RESET} {_BOLD_WHITE}{label}{_RESET} {_BLUE}has finished in{_RESET} {_LIGHT_GRAY}{elapsed:.3f}s.{_RESET}",
            )
            return result

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


def pipeline(fn):
    """Decorator that initialises the step counter before running a pipeline."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _total[0] = sum(1 for _, lvl in _registry if logger.isEnabledFor(lvl))
        _counter[0] = 0
        return fn(*args, **kwargs)

    return wrapper
