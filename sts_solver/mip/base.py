"""Base classes and common helpers for MIP solvers."""

from abc import abstractmethod
from typing import Any, Optional

from ..base_solver import BaseSolver


class MIPBaseSolver(BaseSolver):
    """
    Base class for all MIP solvers.

    Provides common attributes and hooks (weeks, periods, backend).
    Concrete implementations can use OR-Tools, CBC, SCIP, etc.
    
    Note: OR-Tools MIP solvers run in C++ and don't respond to Ctrl+C
    during solving. Use the timeout parameter to limit solving time.
    """

    def __init__(self, n: int, timeout: int = 300, optimization: bool = False, backend: Optional[str] = None):
        super().__init__(n, timeout, optimization)
        self.backend = backend
        self.weeks = n - 1
        self.periods = n // 2

    @abstractmethod
    def _build_model(self) -> Any:  # pragma: no cover - provided by subclasses
        return None

    # _solve_model is inherited abstract from BaseSolver
