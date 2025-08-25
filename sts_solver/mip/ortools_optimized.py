"""
Optimized OR-Tools MIP solver for STS problem

This module provides an optimized version with symmetry breaking,
aggressive presolving, and problem-specific enhancements for better scalability.
"""

import time
from typing import Optional, List
from ortools.linear_solver import pywraplp

from ..utils.solution_format import STSSolution


def solve_sts_ortools_optimized(
    n: int,
    solver_name: Optional[str] = None,
    timeout: int = 300,
    optimization: bool = False
) -> STSSolution:
    """
    Solve STS using optimized OR-Tools MIP solver with advanced techniques
    
    Args:
        n: Number of teams
        solver_name: OR-Tools solver to use (SCIP, CBC, GUROBI, CPLEX)
        timeout: Timeout in seconds
        optimization: Whether to use optimization version
        
    Returns:
        STSSolution object with results
    """
    
    if n % 2 != 0:
        return STSSolution(time=0, optimal=False, obj=None, sol=[])
    
    start_time = time.time()
    
    try:
        # Create solver
        if solver_name and solver_name.upper() == "SCIP":
            solver = pywraplp.Solver.CreateSolver("SCIP")
        elif solver_name and solver_name.upper() == "GUROBI":
            solver = pywraplp.Solver.CreateSolver("GUROBI")
        elif solver_name and solver_name.upper() == "CPLEX":
            solver = pywraplp.Solver.CreateSolver("CPLEX")
        else:
            solver = pywraplp.Solver.CreateSolver("CBC")
        
        if not solver:
            return STSSolution(time=0, optimal=False, obj=None, sol=[])
        
        # Set basic solver parameters
        solver.SetNumThreads(1)
        solver.SetTimeLimit(timeout * 1000)
        
        # Configure solver-specific options
        if solver_name and solver_name.upper() == "CBC":
            try:
                solver.SetSolverSpecificParametersAsString("presolve=on cuts=on heuristics=on")
            except:
                pass  # Continue if parameter setting fails
        elif solver_name and solver_name.upper() == "SCIP":
            # SCIP parameters
            try:
                solver.SetSolverSpecificParametersAsString("presolving/maxrounds = -1")
            except:
                pass
            try:
                solver.SetSolverSpecificParametersAsString("separating/maxrounds = -1")  
            except:
                pass
        
        # Problem parameters
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
        
        # Constraint 1: Every team plays every other team exactly once
        for i in teams:
            for j in teams:
                if i < j:
                    solver.Add(
                        solver.Sum(match_vars[i, j, w, p] + match_vars[j, i, w, p] 
                                   for w in weeks for p in periods) == 1,
                        f"meet_once_{i}_{j}"
                    )

        # Constraint 2: Every team plays exactly once per week
        for k in teams:
            for w in weeks:
                solver.Add(
                    solver.Sum(match_vars[k, j, w, p] for j in teams if k != j for p in periods) +
                    solver.Sum(match_vars[i, k, w, p] for i in teams if i != k for p in periods) == 1,
                    f"one_game_per_week_{k}_{w}"
                )

        # Constraint 3: Each slot (week, period) has exactly one game
        for w in weeks:
            for p in periods:
                solver.Add(
                    solver.Sum(match_vars[i, j, w, p] for i in teams for j in teams if i != j) == 1,
                    f"one_game_per_slot_{w}_{p}"
                )

        # Constraint 4: Every team plays at most twice in the same period
        for k in teams:
            for p in periods:
                solver.Add(
                    solver.Sum(match_vars[k, j, w, p] for j in teams if k != j for w in weeks) +
                    solver.Sum(match_vars[i, k, w, p] for i in teams if i != k for w in weeks) <= 2,
                    f"period_limit_{k}_{p}"
                )

        # Symmetry breaking
        
        # Fix team 1's schedule
        solver.Add(match_vars[1, 2, 1, 1] == 1, "sym_team1_week1_p1")
        
        # Order opponents
        if n >= 6:
            # Team 1 plays against teams 2, 3, 4, ... in weeks 1, 2, 3, ...
            for w_idx, w in enumerate(weeks[:min(4, len(weeks))]):
                if w_idx + 2 <= n:
                    opponent = w_idx + 2
                    solver.Add(
                        solver.Sum(match_vars[1, opponent, w, p] + match_vars[opponent, 1, w, p] 
                                 for p in periods) == 1,
                        f"sym_team1_week{w}_opponent{opponent}"
                    )
        
        # Period symmetry
        solver.Add(
            solver.Sum(match_vars[1, j, 1, 1] for j in teams if j != 1) +
            solver.Sum(match_vars[i, 1, 1, 1] for i in teams if i != 1) == 1,
            "sym_team1_period1_week1"
        )
        
        # Week ordering
        for p in periods[:min(2, len(periods))]:
            for w in weeks[:-1]:  # All weeks except last
                # Get team indices playing in this period
                home_teams_w = [match_vars[i, j, w, p] * i for i in teams for j in teams if i != j]
                home_teams_w_plus_1 = [match_vars[i, j, w+1, p] * i for i in teams for j in teams if i != j]
                
                # Simplified ordering
        
        # Additional constraints
        
        # Home/away balance
        for k in teams:
            home_games_k = solver.Sum(match_vars[k, j, w, p] 
                                    for j in teams if k != j 
                                    for w in weeks for p in periods)
            away_games_k = solver.Sum(match_vars[i, k, w, p] 
                                    for i in teams if i != k 
                                    for w in weeks for p in periods)
            
            # Each team plays exactly n-1 games total
            solver.Add(home_games_k + away_games_k == n - 1, f"total_games_{k}")
            
            # Balance home/away games
            if not optimization and n > 4:
                solver.Add(home_games_k >= (n-1)//2 - 1, f"min_home_{k}")
                solver.Add(home_games_k <= (n-1)//2 + 1, f"max_home_{k}")
        
        # Additional cuts for larger problems
        if n >= 10:
            # Period constraints
            for p in periods:
                total_games_in_period = solver.Sum(match_vars[i, j, w, p] 
                                                 for i in teams for j in teams if i != j
                                                 for w in weeks)
                solver.Add(total_games_in_period == (n-1), f"period_{p}_total_games")

        # Objective
        if optimization:
            home_games = {k: solver.Sum(match_vars[k, j, w, p] for j in teams if k!=j for w in weeks for p in periods) for k in teams}
            away_games = {k: solver.Sum(match_vars[i, k, w, p] for i in teams if i!=k for w in weeks for p in periods) for k in teams}
            
            deviation = {k: solver.NumVar(0, n-1, f'dev_{k}') for k in teams}

            for k in teams:
                solver.Add(deviation[k] >= home_games[k] - away_games[k])
                solver.Add(deviation[k] >= away_games[k] - home_games[k])

            solver.Minimize(solver.Sum(deviation[k] for k in teams))

        # Solve strategy
        if n >= 12:
            # Two-phase solving for larger instances
            original_time_limit = timeout * 1000
            solver.SetTimeLimit(min(60000, original_time_limit // 3))  # Use 1/3 time for quick search
            
            status = solver.Solve()
            
            if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
                # Use full time if quick solve failed
                solver.SetTimeLimit(original_time_limit)
                status = solver.Solve()
        else:
            status = solver.Solve()
        
        elapsed_time = int(time.time() - start_time)

        # Extract solution
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
                        period_games.append([1, 2])  # Fallback
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
            return STSSolution(
                time=elapsed_time if elapsed_time < timeout else timeout,
                optimal=False,
                obj=None,
                sol=[]
            )
            
    except Exception as e:
        elapsed_time = int(time.time() - start_time)
        return STSSolution(
            time=min(elapsed_time, timeout),
            optimal=False,
            obj=None,
            sol=[]
        )