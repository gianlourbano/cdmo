"""
Simplified match-based MIP formulation that's more MIP-friendly
"""

import time
from typing import Optional
from ortools.linear_solver import pywraplp

from ..utils.solution_format import STSSolution


def solve_sts_simple_match(
    n: int,
    solver_name: Optional[str] = None,
    timeout: int = 300,
    optimization: bool = False
) -> STSSolution:
    """
    Solve STS using simplified match-based formulation
    
    Variables:
    - x[t1,t2,w,p] = 1 if teams t1,t2 play in week w, period p (with t1 < t2)
    - h[t1,t2,w,p] = 1 if t1 plays at home vs t2 in week w, period p
    
    This is still more compact than schedule[w,p,slot] approach
    """
    
    if n % 2 != 0:
        return STSSolution(time=0, optimal=False, obj=None, sol=[])
    
    start_time = time.time()
    weeks = n - 1
    periods = n // 2
    
    try:
        # Solver initialization
        if solver_name and solver_name.upper() == "SCIP":
            solver = pywraplp.Solver.CreateSolver("SCIP")
        elif solver_name and solver_name.upper() == "GUROBI":
            solver = pywraplp.Solver.CreateSolver("GUROBI")
        else:
            solver = pywraplp.Solver.CreateSolver("CBC")
        
        if not solver:
            return STSSolution(time=0, optimal=False, obj=None, sol=[])
        
        solver.set_time_limit(timeout * 1000)
        
        # === VARIABLES ===
        # x[t1,t2,w,p] = 1 if teams t1,t2 play in week w, period p (t1 < t2)
        x = {}
        # h[t1,t2,w,p] = 1 if t1 plays at home vs t2 in week w, period p  
        h = {}
        
        for t1 in range(1, n + 1):
            for t2 in range(t1 + 1, n + 1):
                for w in range(weeks):
                    for p in range(periods):
                        x[t1, t2, w, p] = solver.BoolVar(f"x_{t1}_{t2}_{w}_{p}")
                        h[t1, t2, w, p] = solver.BoolVar(f"h_{t1}_{t2}_{w}_{p}")
                        
                        # h can only be 1 if x is 1
                        solver.Add(h[t1, t2, w, p] <= x[t1, t2, w, p])
        
        num_match_vars = n * (n-1) // 2
        total_vars = num_match_vars * weeks * periods * 2  # x and h variables
        baseline_vars = n * weeks * periods * 2  # schedule[w,p,slot] variables
        
        print(f"Match-based variables: {total_vars} vs Baseline: {baseline_vars}")
        
        # === CONSTRAINTS ===
        
        # 1. Each pair plays exactly once
        for t1 in range(1, n + 1):
            for t2 in range(t1 + 1, n + 1):
                total_games = []
                for w in range(weeks):
                    for p in range(periods):
                        total_games.append(x[t1, t2, w, p])
                solver.Add(solver.Sum(total_games) == 1, f"once_total_{t1}_{t2}")
        
        # 2. Each team plays exactly once per week
        for t in range(1, n + 1):
            for w in range(weeks):
                weekly_games = []
                for t2 in range(1, n + 1):
                    if t2 != t:
                        for p in range(periods):
                            if t < t2:
                                weekly_games.append(x[t, t2, w, p])
                            else:
                                weekly_games.append(x[t2, t, w, p])
                solver.Add(solver.Sum(weekly_games) == 1, f"once_per_week_{t}_{w}")
        
        # 3. Each period in each week has exactly one match
        for w in range(weeks):
            for p in range(periods):
                period_games = []
                for t1 in range(1, n + 1):
                    for t2 in range(t1 + 1, n + 1):
                        period_games.append(x[t1, t2, w, p])
                solver.Add(solver.Sum(period_games) == 1, f"one_per_period_{w}_{p}")
        
        # 4. Each team plays at most twice in the same period
        for t in range(1, n + 1):
            for p in range(periods):
                period_appearances = []
                for t2 in range(1, n + 1):
                    if t2 != t:
                        for w in range(weeks):
                            if t < t2:
                                period_appearances.append(x[t, t2, w, p])
                            else:
                                period_appearances.append(x[t2, t, w, p])
                solver.Add(solver.Sum(period_appearances) <= 2, f"at_most_twice_{t}_{p}")
        
        # Symmetry breaking
        if n >= 2:
            solver.Add(x[1, 2, 0, 0] == 1, "fix_first_match")
            solver.Add(h[1, 2, 0, 0] == 1, "fix_first_home")
        
        # === SOLVE ===
        print(f"Solving with {solver.NumConstraints()} constraints, {solver.NumVariables()} variables")
        status = solver.Solve()
        elapsed_time = int(time.time() - start_time)
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # Extract solution
            sol = [[] for _ in range(periods)]
            
            # Find which match is in each slot
            for w in range(weeks):
                for p in range(periods):
                    for t1 in range(1, n + 1):
                        for t2 in range(t1 + 1, n + 1):
                            if x[t1, t2, w, p].solution_value() > 0.5:
                                # This match happens in week w, period p
                                if h[t1, t2, w, p].solution_value() > 0.5:
                                    home_team, away_team = t1, t2
                                else:
                                    home_team, away_team = t2, t1
                                
                                # Ensure sol[p] has enough weeks
                                while len(sol[p]) <= w:
                                    sol[p].append([0, 0])
                                
                                sol[p][w] = [home_team, away_team]
                                break
            
            # Fill missing slots
            for p in range(periods):
                while len(sol[p]) < weeks:
                    sol[p].append([0, 0])
            
            is_optimal = (status == pywraplp.Solver.OPTIMAL)
            
            return STSSolution(
                time=elapsed_time,
                optimal=is_optimal,
                obj=None if not optimization else calculate_balance_objective(sol, n),
                sol=sol
            )
        else:
            return STSSolution(
                time=timeout if elapsed_time >= timeout else elapsed_time,
                optimal=False,
                obj=None,
                sol=[]
            )
            
    except Exception as e:
        elapsed_time = int(time.time() - start_time)
        print(f"Exception in simple match-based MIP: {e}")
        return STSSolution(
            time=elapsed_time,
            optimal=False,
            obj=None,
            sol=[]
        )


def calculate_balance_objective(sol, n):
    """Calculate home/away balance objective (lower is better)"""
    home_counts = [0] * (n + 1)
    away_counts = [0] * (n + 1)
    
    for period_games in sol:
        for home, away in period_games:
            if home > 0 and away > 0:
                home_counts[home] += 1
                away_counts[away] += 1
    
    total_imbalance = sum(abs(home_counts[t] - away_counts[t]) for t in range(1, n + 1))
    return total_imbalance