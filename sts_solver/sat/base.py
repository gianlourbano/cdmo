"""SAT base solver abstraction for STS models (pure Boolean encodings)."""

from typing import Any, Dict
from ..base_solver import BaseSolver, SolverMetadata


class SATBaseSolver(BaseSolver):
    """Base class for SAT encodings of STS.

    Provides convenience attributes for weeks and periods. Concrete subclasses
    implement _build_model to return a (solver, state) tuple and _solve_model
    to extract a STSSolution.
    """

    def __init__(self, n: int, timeout: int = 300, optimization: bool = False):
        super().__init__(n, timeout, optimization)
        self.weeks = n - 1
        self.periods = n // 2

    @classmethod
    def get_metadata(cls) -> SolverMetadata:  # pragma: no cover - overridden in subclasses
        return SolverMetadata(name="sat-base", approach="SAT", version="1.0")
