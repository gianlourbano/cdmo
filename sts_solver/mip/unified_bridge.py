"""Unified registration of all class-based MIP formulations."""

from ..registry import registry
from .standard import MIPStandardNativeSolver as _Std
from .ortools_presolve import MIPPresolveSolver as _Presolve


# Register each model once. Select formulation via `--model` and backend via
# `--solver` (e.g., `--model optimized --solver CBC`). Defaults are applied
# by each class when backend is omitted.

@registry.register("MIP", "standard")
class MIPStandardSolver(_Std):
    """Standard MIP formulation (default backend: CBC)"""
    pass

@registry.register("MIP", "presolve")
class MIPPresolve(_Presolve):
    """Presolve formulation with precomputed round-robin (default backend: SCIP)"""
    pass