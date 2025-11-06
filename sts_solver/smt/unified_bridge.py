"""Wrapper classes to expose current SMT implementations via the unified registry."""
from typing import Any, Optional, Callable

from .base import SMTBaseSolver
from ..base_solver import SolverMetadata
from ..registry import registry
from ..utils.solution_format import STSSolution

# Ensure SMT solvers are registered in their own registry via import side effects
from . import solver as _smt_loader  # noqa: F401
from .registry import get_solver as _get_smt_solver


class _DelegatingSMTSolver(SMTBaseSolver):
    _formulation: str = "baseline"

    def _build_model(self) -> Any:
        return None

    def _solve_model(self, model: Any) -> STSSolution:
        func = _get_smt_solver(self._formulation)
        assert func is not None
        # func signature: (n: int, solver_name: Optional[str], timeout: int, optimization: bool)
        return func(self.n, None, self.timeout, self.optimization)

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name=cls._formulation,
            approach="SMT",
            version="1.0",
            supports_optimization=True,  # SMT variants generally accept the flag
            description=f"Delegates to SMT formulation: {cls._formulation}"
        )


@registry.register("SMT", "baseline")
class SMTBaselineSolver(_DelegatingSMTSolver):
    _formulation = "baseline"


@registry.register("SMT", "optimized")
class SMTOptimizedSolver(_DelegatingSMTSolver):
    _formulation = "optimized"


@registry.register("SMT", "compact")
class SMTCompactSolver(_DelegatingSMTSolver):
    _formulation = "compact"


@registry.register("SMT", "presolve")
class SMTPresolveSolver(_DelegatingSMTSolver):
    _formulation = "presolve"


@registry.register("SMT", "presolve_cvc5")
class SMTPresolveCVC5Solver(_DelegatingSMTSolver):
    _formulation = "presolve_cvc5"


@registry.register("SMT", "presolve_2")
class SMTPresolve2Solver(_DelegatingSMTSolver):
    _formulation = "presolve_2"


@registry.register("SMT", "presolve_3")
class SMTPresolve3Solver(_DelegatingSMTSolver):
    _formulation = "presolve_3"


@registry.register("SMT", "presolve_symmetry")
class SMTPresolveSymmetrySolver(_DelegatingSMTSolver):
    _formulation = "presolve_symmetry"
