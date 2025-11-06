"""CLI utilities for validation and common helpers"""

from ..exceptions import InvalidInstanceError


def validate_instance_size(n: int) -> None:
    if n % 2 != 0:
        raise InvalidInstanceError("Number of teams must be even")
    if n < 4:
        raise InvalidInstanceError("Number of teams must be at least 4")
