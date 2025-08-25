"""
OR-Tools MIP solver for STS problem

This module provides an efficient Mixed-Integer Programming implementation
using Google OR-Tools based on the user's improved model structure.
"""

import time
from typing import Optional, List, Tuple
from ortools.linear_solver import pywraplp

from ..utils.solution_format import STSSolution


def solve_sts_ortools(
    n: int,
    solver_name: Optional[str] = None,
    timeout: int = 300,
    optimization: bool = False
) -> STSSolution:
    """Solve STS using OR-Tools MIP solver"""
    
    if n % 2 != 0:
        return STSSolution(
            time=0,
            optimal=False,
            obj=None,
            sol=[]
        )
    
    start_time = time.time()
    
    try:
        # Create solver
        if solver_name and solver_name.upper() == "SCIP":
            solver = pywraplp.Solver.CreateSolver("SCIP")
        elif solver_name and solver_name.upper() == "GUROBI":
            solver = pywraplp.Solver.CreateSolver("GUROBI")
        elif solver_name and solver_name.upper() == "CPLEX":
            solver = pywraplp.Solver.CreateSolver("CPLEX")
        elif solver_name and solver_name.upper() == "GLOP":
            # GLOP is LP-only
            return STSSolution(
                time=0,
                optimal=False,
                obj=None,
                sol=[]
            )
        else:
            # Default to CBC
            solver = pywraplp.Solver.CreateSolver("CBC")
        
        if not solver:
            return STSSolution(
                time=0,
                optimal=False,
                obj=None,
                sol=[]
            )
        
        solver.SetNumThreads(1)
        solver.SetTimeLimit(timeout * 1000)

        # Parameters
        teams = list(range(1, n + 1))
        weeks = list(range(1, n))
        periods = list(range(1, n // 2 + 1))

        # Variables
        match_vars = {}
        for i in teams:
            for j in teams:
                if i == j: 
                    continue
                for w in weeks:
                    for p in periods:
                        match_vars[i, j, w, p] = solver.BoolVar(f'match_{i}_{j}_{w}_{p}')

        # Constraints
        
        # Every team plays every other team exactly once
        for i in teams:
            for j in teams:
                if i < j:
                    solver.Add(
                        solver.Sum(match_vars[i, j, w, p] + match_vars[j, i, w, p] 
                                   for w in weeks for p in periods) == 1,
                        f"meet_once_{i}_{j}"
                    )

        # Every team plays exactly once per week
        for k in teams:
            for w in weeks:
                solver.Add(
                    solver.Sum(match_vars[k, j, w, p] for j in teams if k != j for p in periods) +
                    solver.Sum(match_vars[i, k, w, p] for i in teams if i != k for p in periods) == 1,
                    f"one_game_per_week_{k}_{w}"
                )

        # Each slot has exactly one game
        for w in weeks:
            for p in periods:
                solver.Add(
                    solver.Sum(match_vars[i, j, w, p] for i in teams for j in teams if i != j) == 1,
                    f"one_game_per_slot_{w}_{p}"
                )

        # At most twice in same period
        for k in teams:
            for p in periods:
                solver.Add(
                    solver.Sum(match_vars[k, j, w, p] for j in teams if k != j for w in weeks) +
                    solver.Sum(match_vars[i, k, w, p] for i in teams if i != k for w in weeks) <= 2,
                    f"period_limit_{k}_{p}"
                )

        # Objective function
        if optimization:
            home_games = {k: solver.Sum(match_vars[k, j, w, p] for j in teams if k!=j for w in weeks for p in periods) for k in teams}
            away_games = {k: solver.Sum(match_vars[i, k, w, p] for i in teams if i!=k for w in weeks for p in periods) for k in teams}
            
            deviation = {k: solver.NumVar(0, solver.infinity(), f'dev_{k}') for k in teams}

            for k in teams:
                solver.Add(deviation[k] >= home_games[k] - away_games[k])
                solver.Add(deviation[k] >= away_games[k] - home_games[k])

            solver.Minimize(solver.Sum(deviation[k] for k in teams))

        # Solve
        status = solver.Solve()
        elapsed_time = int(time.time() - start_time)

        # Extract results
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # Build the schedule matrix
            schedule = [[None for _ in weeks] for _ in periods]
            
            for (i, j, w, p), var in match_vars.items():
                if var.solution_value() > 0.5:
                    schedule[p-1][w-1] = [i, j]

            # Convert to required format
            sol = []
            for p_idx in range(len(periods)):
                period_games = []
                for w_idx in range(len(weeks)):
                    game = schedule[p_idx][w_idx]
                    if game:
                        period_games.append(game)
                    else:
                        # Fallback
                        period_games.append([1, 2])
                sol.append(period_games)

            optimal = (status == pywraplp.Solver.OPTIMAL)
            obj_value = None
            if optimization and solver.Objective():
                obj_value = int(solver.Objective().Value())
            
            return STSSolution(
                time=elapsed_time,
                optimal=optimal,
                obj=obj_value,
                sol=sol
            )
        else:
            # No solution
            return STSSolution(
                time=elapsed_time if elapsed_time < timeout else timeout,
                optimal=False,
                obj=None,
                sol=[]
            )
            
    except Exception as e:
        elapsed_time = int(time.time() - start_time)
        if elapsed_time >= timeout:
            elapsed_time = timeout
            
        return STSSolution(
            time=elapsed_time,
            optimal=False,
            obj=None,
            sol=[]
        )


def get_available_ortools_solvers() -> List[str]:
    """Get list of available OR-Tools MIP solvers (excludes LP-only solvers)"""
    available = []
    
    # Only include true MIP solvers, exclude GLOP (LP-only)
    mip_solvers_to_check = ["CBC", "SCIP", "GUROBI", "CPLEX"]
    
    for solver_name in mip_solvers_to_check:
        try:
            solver = pywraplp.Solver.CreateSolver(solver_name)
            if solver:
                available.append(solver_name)
        except:
            continue
    
    return available