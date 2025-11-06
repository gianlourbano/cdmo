"""CP unified registry bridge: exposes MiniZinc backends as registry solvers."""
from typing import Any

from ..base_solver import BaseSolver, SolverMetadata
from ..registry import registry
from ..utils.solution_format import STSSolution
from .solver import solve_cp


class _DelegatingCPSolver(BaseSolver):
    _backend: str = "gecode"
    _name: str = "gecode"

    def _build_model(self) -> Any:
        return None

    def _solve_model(self, model: Any) -> STSSolution:
        # Pass through backend name to CP solver
        return solve_cp(self.n, self._backend, self.timeout, self.optimization)

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name=cls._name,
            approach="CP",
            version="1.0",
            supports_optimization=True,
            description=f"MiniZinc backend: {cls._backend}"
        )


@registry.register("CP", "gecode")
class CPGeocodeSolver(_DelegatingCPSolver):
    _backend = "gecode"
    _name = "gecode"


@registry.register("CP", "chuffed")
class CPChuffedSolver(_DelegatingCPSolver):
    _backend = "chuffed"
    _name = "chuffed"
