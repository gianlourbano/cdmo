"""Unified registration of all class-based MIP formulations."""

from ..registry import registry
from .standard import MIPStandardNativeSolver as _Std
from .optimized import MIPOptimizedNativeSolver as _Opt
from .compact import MIPCompactNativeSolver as _Compact
from .ortools_simple_match import MIPMatchSolver as _Match
from .ortools_match_based import MIPMatchCompactSolver as _MatchCompact
from .ortools_compact import MIPFlowSolver as _Flow
from .ortools_presolve import MIPPresolveSolver as _Presolve


# Register each model once. Select formulation via `--model` and backend via
# `--solver` (e.g., `--model optimized --solver CBC`). Defaults are applied
# by each class when backend is omitted.

@registry.register("MIP", "standard")
class MIPStandardSolver(_Std):
    """Standard MIP formulation (default backend: CBC)"""
    pass


@registry.register("MIP", "optimized")
class MIPOptimizedSolver(_Opt):
    """Optimized MIP formulation with symmetry breaking (default backend: CBC)"""
    pass


@registry.register("MIP", "compact")
class MIPCompactSolver(_Compact):
    """Compact MIP formulation with fewer variables (default backend: CBC)"""
    pass


@registry.register("MIP", "match")
class MIPMatch(_Match):
    """Simplified match-based formulation (default backend: SCIP)"""
    pass


@registry.register("MIP", "match_compact")
class MIPMatchCompact(_MatchCompact):
    """True compact match-based formulation (default backend: SCIP)"""
    pass


@registry.register("MIP", "flow")
class MIPFlow(_Flow):
    """Multi-commodity flow formulation (default backend: SCIP)"""
    pass


@registry.register("MIP", "presolve")
class MIPPresolve(_Presolve):
    """Presolve formulation with precomputed round-robin (default backend: SCIP)"""
    pass
