"""Error handling utilities for solvers and CLI"""

from contextlib import contextmanager
from typing import Callable, Type, Tuple, Any, Optional
import time

from sts_solver.exceptions import (
    STSException,
    InvalidInstanceError,
    SolverNotFoundError,
    SolverTimeoutError,
    InvalidSolutionError,
    ConfigurationError,
)


@contextmanager
def handle_solver_errors(rethrow: bool = False):
    """
    Context manager to normalize solver exceptions.

    Usage:
        with handle_solver_errors():
            ... solver code ...
    """
    try:
        yield
    except SolverTimeoutError:
        # pass through as-is
        if rethrow:
            raise
    except (InvalidInstanceError, InvalidSolutionError, ConfigurationError, SolverNotFoundError) as e:
        if rethrow:
            raise
        # Swallow known exceptions to allow graceful CLI handling
    except Exception as e:  # noqa: BLE001
        if rethrow:
            raise STSException(str(e)) from e


def retry_on_failure(
    exceptions: Tuple[Type[BaseException], ...] = (STSException,),
    retries: int = 0,
    delay: float = 0.0,
    backoff: float = 1.0,
):
    """Retry decorator with optional backoff for transient failures."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args, **kwargs):
            _retries = retries
            _delay = delay
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if _retries <= 0:
                        raise
                    if _delay > 0:
                        time.sleep(_delay)
                    _retries -= 1
                    _delay *= max(backoff, 1.0)

        return wrapper

    return decorator
