"""Wrapper classes to expose current MIP implementations via the unified registry."""
from typing import Any, Optional, Callable

from .base import MIPBaseSolver
from ..base_solver import SolverMetadata
from ..registry import registry
from ..utils.solution_format import STSSolution
from .solver import (
    solve_mip_match,
    solve_mip_flow,
    solve_mip_pulp,
    solve_mip_presolve,
)
from .compact import MIPCompactNativeSolver
from .optimized import MIPOptimizedNativeSolver
from .standard import MIPStandardNativeSolver


class _DelegatingMIPSolver(MIPBaseSolver):
    _delegate: Optional[Callable[[int, Optional[str], int, bool], STSSolution]] = None
    _name = "mip-delegate"
    _supports_opt = False
    _max_n = None  # type: Optional[int]

    def _build_model(self) -> Any:
        return None

    def _solve_model(self, model: Any) -> STSSolution:
        assert self._delegate is not None
        delegate = self._delegate
        return delegate(self.n, None, self.timeout, self.optimization)

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name=cls._name,
            approach="MIP",
            version="1.0",
            supports_optimization=cls._supports_opt,
            max_recommended_size=cls._max_n,
            description=f"Delegates to existing MIP implementation: {cls._name}"
        )


@registry.register("MIP", "standard")
class MIPStandardSolver(MIPStandardNativeSolver):
    pass


@registry.register("MIP", "optimized")
class MIPOptimizedSolver(MIPOptimizedNativeSolver):
    pass


@registry.register("MIP", "compact")
class MIPCompactSolver(MIPCompactNativeSolver):
    pass


@registry.register("MIP", "match")
class MIPMatchSolver(_DelegatingMIPSolver):
    _delegate = staticmethod(solve_mip_match)
    _name = "match"
    _supports_opt = False


@registry.register("MIP", "flow")
class MIPFlowSolver(_DelegatingMIPSolver):
    _delegate = staticmethod(solve_mip_flow)
    _name = "flow"
    _supports_opt = False


@registry.register("MIP", "pulp")
class MIPPulpSolver(_DelegatingMIPSolver):
    _delegate = staticmethod(solve_mip_pulp)
    _name = "pulp"
    _supports_opt = False


@registry.register("MIP", "presolve")
class MIPPresolveSolver(_DelegatingMIPSolver):
    _delegate = staticmethod(solve_mip_presolve)
    _name = "presolve"
    _supports_opt = False
