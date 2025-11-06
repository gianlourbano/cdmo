"""Base solver interface for all STS solver implementations"""

from abc import ABC, abstractmethod
from typing import Optional, Any
from dataclasses import dataclass
import time

from .utils.solution_format import STSSolution
from .exceptions import InvalidInstanceError


@dataclass
class SolverMetadata:
    """Metadata about a solver implementation"""
    name: str
    approach: str  # CP, SAT, SMT, MIP
    version: str
    supports_optimization: bool = False
    max_recommended_size: Optional[int] = None
    description: str = ""


class BaseSolver(ABC):
    """
    Abstract base class for all STS solvers.
    Provides input validation, basic timing, and a consistent API.
    """

    def __init__(self, n: int, timeout: int = 300, optimization: bool = False):
        self.n = n
        self.timeout = timeout
        self.optimization = optimization
        self.start_time: Optional[float] = None
        self._validate_instance()

    def _validate_instance(self) -> None:
        if self.n % 2 != 0:
            raise InvalidInstanceError("Number of teams must be even")
        if self.n < 4:
            raise InvalidInstanceError("Number of teams must be at least 4")
        if self.timeout <= 0:
            raise InvalidInstanceError("Timeout must be positive")

    @abstractmethod
    def _build_model(self) -> Any:
        """Build solver-specific model or data."""
        pass

    @abstractmethod
    def _solve_model(self, model: Any) -> STSSolution:
        """Solve the built model and return a STSSolution."""
        pass

    def solve(self) -> STSSolution:
        """Main solving method with timing and basic error handling."""
        self.start_time = time.time()
        try:
            model = self._build_model()
            return self._solve_model(model)
        except Exception:
            elapsed = int(time.time() - self.start_time) if self.start_time else 0
            return STSSolution(
                time=min(elapsed, self.timeout),
                optimal=False,
                obj=None,
                sol=[]
            )

    @property
    def elapsed_time(self) -> int:
        if self.start_time is None:
            return 0
        return int(time.time() - self.start_time)

    @property
    def is_timeout(self) -> bool:
        return self.elapsed_time >= self.timeout

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> SolverMetadata:
        """Return metadata about this solver."""
        pass
