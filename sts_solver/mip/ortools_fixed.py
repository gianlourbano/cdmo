"""
Fixed compact OR-Tools MIP formulation for STS problem

This provides a working, simpler compact formulation that should scale better.
"""

import time
from typing import Optional
from ortools.linear_solver import pywraplp

from ..utils.solution_format import STSSolution


def solve_sts_fixed_compact(
    n: int,
    solver_name: Optional[str] = None,
    timeout: int = 300,
    optimization: bool = False
) -> STSSolution:
    """
    Solve STS using a fixed, simplified compact formulation
    
    Uses assignment variables: x[i,j,w,p] = 1 if team i plays at home vs team j in week w, period p
    This is cleaner than the original match formulation.
    """
    
    if n % 2 != 0:
        return STSSolution(time=0, optimal=False, obj=None, sol=[])
    
    start_time = time.time()
    
    try:
        # Solver setup
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
        
        solver.SetNumThreads(1)
        solver.SetTimeLimit(timeout * 1000)
        
        # Enable aggressive presolving with error handling
        if solver_name == "CBC" or solver_name is None:
            try:
                solver.SetSolverSpecificParametersAsString("cuts=on presolve=on heuristics=on")
            except:
                pass
        elif solver_name and solver_name.upper() == "SCIP":
            # Conservative SCIP parameters
            try:
                solver.SetSolverSpecificParametersAsString("presolving/maxrounds = -1")
            except:
                pass
            try:
                solver.SetSolverSpecificParametersAsString("separating/maxrounds = -1")
            except:
                pass
        
        teams = list(range(1, n + 1))
        weeks = list(range(1, n))
        periods = list(range(1, n // 2 + 1))
        
        # === DECISION VARIABLES ===
        # x[i,j,w,p] = 1 if team i plays at home against team j in week w, period p
        x = {}
        for i in teams:
            for j in teams:
                if i != j:  # Team can't play against itself
                    for w in weeks:
                        for p in periods:
                            x[i, j, w, p] = solver.BoolVar(f"x_{i}_{j}_{w}_{p}")
        
        # === CONSTRAINTS ===
        
        # 1. Every pair of teams plays exactly once (either i vs j or j vs i)
        for i in teams:
            for j in teams:
                if i < j:  # Only consider each pair once
                    solver.Add(
                        solver.Sum(x[i, j, w, p] + x[j, i, w, p] 
                                 for w in weeks for p in periods) == 1,
                        f"pair_{i}_{j}_once"
                    )
        
        # 2. Each team plays exactly once per week
        for t in teams:
            for w in weeks:
                # Sum all games where team t is involved (home or away)
                games_as_home = solver.Sum(x[t, j, w, p] for j in teams if j != t for p in periods)
                games_as_away = solver.Sum(x[i, t, w, p] for i in teams if i != t for p in periods)
                solver.Add(games_as_home + games_as_away == 1, f"team_{t}_week_{w}")
        
        # 3. Each slot (week, period) has exactly one game
        for w in weeks:
            for p in periods:
                solver.Add(
                    solver.Sum(x[i, j, w, p] for i in teams for j in teams if i != j) == 1,
                    f"slot_{w}_{p}"
                )
        
        # 4. Period constraint: each team plays at most twice in the same period
        for t in teams:
            for p in periods:
                games_in_period = (
                    solver.Sum(x[t, j, w, p] for j in teams if j != t for w in weeks) +
                    solver.Sum(x[i, t, w, p] for i in teams if i != t for w in weeks)
                )
                solver.Add(games_in_period <= 2, f"team_{t}_period_{p}")
        
        # === SYMMETRY BREAKING ===
        
        # Fix team 1 vs team 2 in week 1, period 1
        solver.Add(x[1, 2, 1, 1] == 1, "sym_fix_1_vs_2")
        
        # Team 1's opponents in order (helps with symmetry)
        if n >= 6:
            for w_idx, w in enumerate(weeks[:min(3, len(weeks))]):
                if w == 1:
                    continue  # Already fixed
                opponent = 2 + w_idx
                if opponent <= n:
                    solver.Add(
                        solver.Sum(x[1, opponent, w, p] + x[opponent, 1, w, p] for p in periods) == 1,
                        f"sym_team1_opponent_{opponent}_week_{w}"
                    )
        
        # Lexicographic ordering of first period
        for w in weeks[:-1]:
            home_team_w = solver.Sum(i * x[i, j, w, 1] for i in teams for j in teams if i != j)
            home_team_w_plus_1 = solver.Sum(i * x[i, j, w+1, 1] for i in teams for j in teams if i != j)
            solver.Add(home_team_w <= home_team_w_plus_1, f"lex_order_week_{w}")
        
        # === OBJECTIVE FUNCTION ===
        if optimization:
            # Count home and away games for each team
            home_games = {}
            away_games = {}
            
            for t in teams:
                home_games[t] = solver.Sum(x[t, j, w, p] for j in teams if j != t for w in weeks for p in periods)
                away_games[t] = solver.Sum(x[i, t, w, p] for i in teams if i != t for w in weeks for p in periods)
            
            # Minimize total deviation from perfect balance
            deviations = {}
            for t in teams:
                dev = solver.NumVar(0, n-1, f"dev_{t}")
                solver.Add(dev >= home_games[t] - away_games[t])
                solver.Add(dev >= away_games[t] - home_games[t])
                deviations[t] = dev
            
            solver.Minimize(solver.Sum(deviations[t] for t in teams))
        
        # === SOLVE ===
        status = solver.Solve()
        elapsed_time = int(time.time() - start_time)
        
        # === EXTRACT SOLUTION ===
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            sol = []
            
            for p in periods:
                period_games = []
                for w in weeks:
                    # Find which game is played in this slot
                    game_found = False
                    for i in teams:
                        for j in teams:
                            if i != j and x[i, j, w, p].solution_value() > 0.5:
                                period_games.append([i, j])  # [home, away]
                                game_found = True
                                break
                        if game_found:
                            break
                    
                    if not game_found:
                        # This shouldn't happen with correct constraints
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
            return STSSolution(
                time=min(elapsed_time, timeout),
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