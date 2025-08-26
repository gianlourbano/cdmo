"""
SMT solver dispatcher for STS using Z3 with multiple formulations
"""

from typing import Optional
from ..utils.solution_format import STSSolution
from .z3_baseline import solve_smt as solve_smt_baseline
from .z3_optimized import solve_smt_optimized
from .z3_compact import solve_smt_compact
from .z3_tactics import solve_smt_tactics


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
        solver_name: SMT variant (baseline, optimized, compact, tactics, or z3)
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
    
    # Choose formulation based on solver name
    if formulation == "optimized":
        return solve_smt_optimized(n, solver_name, timeout, optimization)
    elif formulation == "compact":
        return solve_smt_compact(n, solver_name, timeout, optimization)
    elif formulation == "tactics":
        return solve_smt_tactics(n, solver_name, timeout, optimization)
    else:
        # Baseline formulation (default)
        return solve_smt_baseline(n, solver_name, timeout, optimization)