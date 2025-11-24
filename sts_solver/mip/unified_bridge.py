"""Unified registration of all class-based MIP formulations."""

from ..registry import registry
from .standard import MIPStandardNativeSolver as _Std
from .optimized import MIPOptimizedNativeSolver as _Opt
from .compact import MIPCompactNativeSolver as _Compact
from .ortools_simple_match import MIPMatchSolver as _Match
from .ortools_match_based import MIPMatchCompactSolver as _MatchCompact
from .ortools_compact import MIPFlowSolver as _Flow
from .ortools_presolve import MIPPresolveSolver as _Presolve
from .pulp import MIPPulpSolver as _Pulp


@registry.register("MIP", "standard")
class MIPStandardSolver(_Std):
    pass


@registry.register("MIP", "optimized")
class MIPOptimizedSolver(_Opt):
    pass


@registry.register("MIP", "compact")
class MIPCompactSolver(_Compact):
    pass


@registry.register("MIP", "match")
class MIPMatch(_Match):
    pass


@registry.register("MIP", "match_compact")
class MIPMatchCompact(_MatchCompact):
    pass


@registry.register("MIP", "flow")
class MIPFlow(_Flow):
    pass


@registry.register("MIP", "presolve")
class MIPPresolve(_Presolve):
    pass


@registry.register("MIP", "pulp")
class MIPPulp(_Pulp):
    pass
