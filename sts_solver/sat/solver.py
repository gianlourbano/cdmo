"""
Pure SAT solvers dispatcher for the STS problem.
"""

from typing import Optional
from ..utils.solution_format import STSSolution

from .z3_sat_baseline import solve_sat_baseline
from .z3_sat_pairwise import solve_sat_pairwise


def solve_sat(
    n: int, 
    solver_name: Optional[str] = None, 
    timeout: int = 300,
    optimization: bool = False
) -> STSSolution:
    """
    Solve STS using a pure SAT encoding with Z3, with multiple formulation options.
    
    This dispatcher parses the solver_name to select the appropriate SAT encoding.
    
    Args:
        n: Number of teams.
        solver_name: SAT formulation name (e.g., "baseline", "pairwise").
        timeout: Timeout in seconds.
        optimization: Whether to use optimization (not applicable for pure SAT).
        
    Returns:
        STSSolution object with results.
    """
        
    # Default formulation if none is provided.
    formulation = "baseline"
    actual_solver = solver_name
    
    if solver_name and "-" in solver_name:
        parts = solver_name.split("-", 1)
        formulation = parts[0].lower()
        actual_solver = parts[1].upper()
        
    if formulation == "pairwise":
        return solve_sat_pairwise(n, actual_solver, timeout, optimization)
    else:
        return solve_sat_baseline(n, actual_solver, timeout, optimization)