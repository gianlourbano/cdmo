"""CLI utilities for validation and common helpers"""

from typing import List
from ..exceptions import InvalidInstanceError
from ..registry import registry


def validate_instance_size(n: int) -> None:
    if n % 2 != 0:
        raise InvalidInstanceError("Number of teams must be even")
    if n < 4:
        raise InvalidInstanceError("Number of teams must be at least 4")


def get_solvers_for_approach(approach: str) -> List[str]:
    """
    Return the list of solver names to benchmark for an approach.
    
    Uses the unified registry to get all registered solvers for the approach.
    This ensures consistency with the list-solvers command.
    """
    approach = approach.upper()
    
    # Ensure all solver modules are imported (triggers registration)
    if approach == "CP":
        import sts_solver.cp.unified_bridge  # noqa: F401
    elif approach == "SAT":
        import sts_solver.sat.unified_bridge  # noqa: F401
    elif approach == "SMT":
        import sts_solver.smt.unified_bridge  # noqa: F401
    elif approach == "MIP":
        import sts_solver.mip.unified_bridge  # noqa: F401
    
    # Get solvers from registry
    solvers_dict = registry.list_solvers(approach=approach)
    solver_list = solvers_dict.get(approach, [])
    
    return sorted(solver_list)
