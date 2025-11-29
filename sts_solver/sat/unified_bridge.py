"""SAT unified registry bridge: exposes SAT formulations as registry solvers."""
from ..registry import registry

from .vanilla_heule_class import SATVanillaHeuleSolver as _SATVanillaHeuleSolver
from .vanilla_totalizer_class import SATVanillaTotalizerSolver as _SATVanillaTotalizerSolver
from .vanilla_sequential_class import SATVanillaSequentialSolver as _SATVanillaSequentialSolver
from .vanilla_pairwise_class import SATVanillaPairwiseSolver as _SATVanillaPairwiseSolver

from sts_solver.sat.pairwise_sb_class import SATPairwiseSBSolver as _SATPairwiseSBSolver
from sts_solver.sat.pairwise_sb_dt_class import SATPairwiseSBDTSolver as _SATPairwiseSBDTSolver
from sts_solver.sat.pairwise_dt_class import SATPairwiseDTSolver as _SATPairwiseDTSolver


@registry.register("SAT", "vanilla_heule")
class SATHeuleVanillaSolver(_SATVanillaHeuleSolver):
    """Registry wrapper for Heule vanilla native SAT solver."""
    pass

@registry.register("SAT", "vanilla_sequential")
class SATVanillaSequentialSolver(_SATVanillaSequentialSolver):
    """Registry wrapper for vanilla sequential native SAT solver."""
    pass

@registry.register("SAT", "vanilla_pairwise")
class SATVanillaPairwiseSolver(_SATVanillaPairwiseSolver):
    """Registry wrapper for vanilla pairwise native SAT solver."""
    pass

@registry.register("SAT", "vanilla_totalizer")
class SATVanillaTotalizerSolver(_SATVanillaTotalizerSolver):
    """Registry wrapper for vanilla totalizer native SAT solver."""
    pass

@registry.register("SAT", "pairwise_sb")
class SATPairwiseSBSolver(_SATPairwiseSBSolver):
    """Registry wrapper for pairwise with symmetry breaking native SAT solver."""
    pass

@registry.register("SAT", "pairwise_sb_dt")
class SATPairwiseSBDTSolver(_SATPairwiseSBDTSolver):
    """Registry wrapper for pairwise with symmetry breaking and deficient teams native SAT solver."""
    pass

@registry.register("SAT", "pairwise_dt")
class SATPairwiseDTSolver(_SATPairwiseDTSolver):
    """Registry wrapper for pairwise with deficient teams native SAT solver."""
    pass
