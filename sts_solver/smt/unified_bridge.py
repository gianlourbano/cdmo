"""Expose SMT class-based implementations via the unified registry."""
from ..registry import registry
from .baseline_class import SMTBaselineNativeSolver
from .optimized_class import SMTOptimizedNativeSolver
from .compact_class import SMTCompactNativeSolver
from .presolve_class import SMTPresolveNativeSolver
from .presolve2_class import SMTPresolve2NativeSolver
from .presolve3_class import SMTPresolve3NativeSolver
from .presolve_symmetry_class import SMTPresolveSymmetryNativeSolver


@registry.register("SMT", "baseline")
class SMTBaselineSolver(SMTBaselineNativeSolver):
    pass


# @registry.register("SMT", "optimized")
# class SMTOptimizedSolver(SMTOptimizedNativeSolver):
#     pass


# @registry.register("SMT", "compact")
# class SMTCompactSolver(SMTCompactNativeSolver):
#     pass


# @registry.register("SMT", "presolve")
# class SMTPresolveSolver(SMTPresolveNativeSolver):
#     pass


@registry.register("SMT", "presolve_2")
class SMTPresolve2Solver(SMTPresolve2NativeSolver):
    pass


# @registry.register("SMT", "presolve_3")
# class SMTPresolve3Solver(SMTPresolve3NativeSolver):
#     pass


# @registry.register("SMT", "presolve_symmetry")
# class SMTPresolveSymmetrySolver(SMTPresolveSymmetryNativeSolver):
#     pass
