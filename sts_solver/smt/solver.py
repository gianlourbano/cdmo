"""
SMT solver dispatcher for STS using Z3 with multiple formulations
"""

from typing import Optional
from ..utils.solution_format import STSSolution
from .registry import get_all_solvers, get_registered_solvers

# Import solver modules (decorators will auto-register)
from . import z3_baseline
from . import z3_optimized  
from . import z3_compact
from . import z3_with_presolve
from . import cvc5_presolve

def solve_smt(
    n: int, 
    solver_name: Optional[str] = None, 
    timeout: int = 300,
    optimization: bool = False
) -> STSSolution:
    """
    Solve STS using SMT with Z3 - Multiple formulation options
    
    Args:
        n: Number of teams
        solver_name: SMT variant (baseline, optimized, compact, tactics, presolve, or z3)
        timeout: Timeout in seconds
        optimization: Whether to use optimization version
        
    Returns:
        STSSolution object with results
    """
    
    # Parse solver name to extract formulation
    formulation = "baseline"
    
    if solver_name:
        formulation = solver_name.lower()
        if formulation == "z3":
            formulation = "baseline"  # Default Z3 is baseline
    
    # Use registry to find and call the appropriate solver
    solvers = get_all_solvers()
    if formulation in solvers:
        return solvers[formulation](n, solver_name, timeout, optimization)
    else:
        # Fallback to baseline if formulation not found
        return solvers["baseline"](n, solver_name, timeout, optimization)