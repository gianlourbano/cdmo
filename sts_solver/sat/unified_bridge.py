"""SAT unified registry bridge: exposes SAT formulations as registry solvers."""
from typing import Any

from ..base_solver import BaseSolver, SolverMetadata
from ..registry import registry
from ..utils.solution_format import STSSolution
from .solver import solve_sat


class _DelegatingSATFormulation(BaseSolver):
    _formulation: str = "baseline"
    _name: str = "baseline"

    def _build_model(self) -> Any:
        return None

    def _solve_model(self, model: Any) -> STSSolution:
        # Pass formulation name via solver_name
        return solve_sat(self.n, self._name, self.timeout, self.optimization)

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name=cls._name,
            approach="SAT",
            version="1.0",
            supports_optimization=False,
            description=f"SAT formulation: {cls._name}"
        )


@registry.register("SAT", "baseline")
class SATBaselineSolver(_DelegatingSATFormulation):
    _formulation = "baseline"
    _name = "baseline"


@registry.register("SAT", "pairwise")
class SATPairwiseSolver(_DelegatingSATFormulation):
    _formulation = "pairwise"
    _name = "pairwise"
