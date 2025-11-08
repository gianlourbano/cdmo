"""Wrapper classes to expose current SMT implementations via the unified registry."""
from typing import Any, Optional, Callable

from .base import SMTBaseSolver
from ..base_solver import SolverMetadata
from ..registry import registry
from ..utils.solution_format import STSSolution

# Ensure SMT solvers are registered in their own registry via import side effects
from . import solver as _smt_loader  # noqa: F401
from .registry import get_solver as _get_smt_solver
from .baseline_class import SMTBaselineNativeSolver
from .optimized_class import SMTOptimizedNativeSolver
from .compact_class import SMTCompactNativeSolver
from .presolve_class import SMTPresolveNativeSolver
from .presolve2_class import SMTPresolve2NativeSolver
from .presolve3_class import SMTPresolve3NativeSolver
from .presolve_symmetry_class import SMTPresolveSymmetryNativeSolver
from .presolve_cvc5_class import SMTPresolveCVC5NativeSolver


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
class SMTBaselineSolver(SMTBaselineNativeSolver):
    pass


@registry.register("SMT", "optimized")
class SMTOptimizedSolver(SMTOptimizedNativeSolver):
    pass


@registry.register("SMT", "compact")
class SMTCompactSolver(SMTCompactNativeSolver):
    pass


@registry.register("SMT", "presolve")
class SMTPresolveSolver(SMTPresolveNativeSolver):
    pass


@registry.register("SMT", "presolve_cvc5")
class SMTPresolveCVC5Solver(SMTPresolveCVC5NativeSolver):
    pass


@registry.register("SMT", "presolve_2")
class SMTPresolve2Solver(SMTPresolve2NativeSolver):
    pass


@registry.register("SMT", "presolve_3")
class SMTPresolve3Solver(SMTPresolve3NativeSolver):
    pass


@registry.register("SMT", "presolve_symmetry")
class SMTPresolveSymmetrySolver(SMTPresolveSymmetryNativeSolver):
    pass
