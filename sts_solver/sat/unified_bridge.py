"""SAT unified registry bridge: exposes SAT formulations as registry solvers."""
from ..registry import registry
from .baseline_class import SATBaselineNativeSolver
from .pairwise_class import SATPairwiseNativeSolver


@registry.register("SAT", "baseline")
class SATBaselineSolver(SATBaselineNativeSolver):
    """Registry wrapper for baseline native SAT solver."""
    pass

@registry.register("SAT", "pairwise")
class SATPairwiseSolver(SATPairwiseNativeSolver):
    """Registry wrapper for pairwise native SAT solver."""
    pass
