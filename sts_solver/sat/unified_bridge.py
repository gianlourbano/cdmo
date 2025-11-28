"""SAT unified registry bridge: exposes SAT formulations as registry solvers."""
from ..registry import registry

from .vanilla_heule_class import SATVanillaHeuleSolver
from .vanilla_totalizer_class import SATVanillaTotalizerSolver
from .vanilla_sequential_class import SATVanillaSequentialSolver
from .vanilla_pairwise_class import SATVanillaPairwiseSolver

from sts_solver.sat.pairwise_sb_class import SATPairwiseSBSolver
from sts_solver.sat.pairwise_sb_dt_class import SATPairwiseSBDTSolver
from sts_solver.sat.pairwise_dt_class import SATPairwiseDTSolver


@registry.register("SAT", "vanilla_heule")
class SATHeuleVanillaSolver(SATVanillaHeuleSolver):
    """Registry wrapper for Heule vanilla native SAT solver."""
    pass

@registry.register("SAT", "vanilla_sequential")
class SATVanillaSequentialSolver(SATVanillaSequentialSolver):
    """Registry wrapper for vanilla sequential native SAT solver."""
    pass

@registry.register("SAT", "vanilla_pairwise")
class SATVanillaPairwiseSolver(SATVanillaPairwiseSolver):
    """Registry wrapper for vanilla pairwise native SAT solver."""
    pass

@registry.register("SAT", "vanilla_totalizer")
class SATVanillaTotalizerSolver(SATVanillaTotalizerSolver):
    """Registry wrapper for vanilla totalizer native SAT solver."""
    pass

@registry.register("SAT", "pairwise_sb")
class SATPairwiseSBSolver(SATPairwiseSBSolver):
    """Registry wrapper for pairwise with symmetry breaking native SAT solver."""
    pass

@registry.register("SAT", "pairwise_sb_dt")
class SATPairwiseSBDTSolver(SATPairwiseSBDTSolver):
    """Registry wrapper for pairwise with symmetry breaking and deficient teams native SAT solver."""
    pass

@registry.register("SAT", "pairwise_dt")
class SATPairwiseDTSolver(SATPairwiseDTSolver):
    """Registry wrapper for pairwise with deficient teams native SAT solver."""
    pass
