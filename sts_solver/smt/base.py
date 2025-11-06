"""Base classes and common helpers for SMT solvers (Z3/CVC5)."""

from abc import abstractmethod
from typing import Any

from ..base_solver import BaseSolver


class SMTBaseSolver(BaseSolver):
    """Base class for all SMT solvers (Z3/CVC5 variants)."""

    def __init__(self, n: int, timeout: int = 300, optimization: bool = False, use_tactics: bool = False):
        super().__init__(n, timeout, optimization)
        self.use_tactics = use_tactics
        self.weeks = n - 1
        self.periods = n // 2

    @abstractmethod
    def _build_model(self) -> Any:  # pragma: no cover - provided by subclasses
        return None
