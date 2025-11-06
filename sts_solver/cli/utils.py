"""CLI utilities for validation and common helpers"""

from ..exceptions import InvalidInstanceError


def validate_instance_size(n: int) -> None:
    if n % 2 != 0:
        raise InvalidInstanceError("Number of teams must be even")
    if n < 4:
        raise InvalidInstanceError("Number of teams must be at least 4")


def get_solvers_for_approach(approach: str):
    """Return the list of solver names to benchmark for an approach."""
    approach = approach.upper()
    if approach == "CP":
        return ["gecode", "chuffed"]
    if approach == "SAT":
        return ["z3"]
    if approach == "SMT":
        # Keep parity with legacy CLI
        return ["baseline", "optimized", "compact", "tactics"]
    if approach == "MIP":
        from ..mip.ortools_solver import get_available_ortools_solvers
        ortools_solvers = get_available_ortools_solvers()

        mip_solvers = []
        for s in ortools_solvers:
            mip_solvers.append(s)
        for s in ortools_solvers:
            mip_solvers.append(f"optimized-{s}")
        for s in ortools_solvers:
            mip_solvers.append(f"compact-{s}")
        for s in ortools_solvers:
            mip_solvers.append(f"match-{s}")
        return mip_solvers
    return []
