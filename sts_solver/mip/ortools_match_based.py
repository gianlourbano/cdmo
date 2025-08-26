"""
True compact (match-based) OR-Tools MIP formulation for STS problem

Based on the highly successful SMT compact approach, this uses match-based variables
instead of schedule-based variables for better scalability.
"""

import time
from typing import Optional
from ortools.linear_solver import pywraplp

from ..utils.solution_format import STSSolution


def solve_sts_match_based(
    n: int,
    solver_name: Optional[str] = None,
    timeout: int = 300,
    optimization: bool = False
) -> STSSolution:
    """
    Solve STS using true compact match-based formulation (like SMT compact)
    
    This formulation uses only 3 variables per match:
    - match_week[t1,t2] = which week teams t1,t2 play (integer)
    - match_period[t1,t2] = which period teams t1,t2 play (integer)
    - match_t1_home[t1,t2] = 1 if t1 plays at home vs t2, 0 otherwise (boolean)
    
    Total variables: 3 * n*(n-1)/2 instead of n*(n-1)*periods*weeks
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
        # Only 3 variables per unique team pair (much more compact!)
        match_week = {}
        match_period = {}  
        match_t1_home = {}
        
        for t1 in range(1, n + 1):
            for t2 in range(t1 + 1, n + 1):
                match_key = (t1, t2)
                match_week[match_key] = solver.IntVar(0, weeks - 1, f"week_{t1}_{t2}")
                match_period[match_key] = solver.IntVar(0, periods - 1, f"period_{t1}_{t2}")
                match_t1_home[match_key] = solver.BoolVar(f"t1_home_{t1}_{t2}")
        
        print(f"Created {len(match_week) * 3} total variables for {n} teams")
        print(f"  vs {n * weeks * periods * 2} variables in schedule-based approach")
        
        # === CONSTRAINTS ===
        
        # Constraint 1: Each team plays exactly once per week
        for t in range(1, n + 1):
            for w in range(weeks):
                weekly_matches = []
                for t2 in range(1, n + 1):
                    if t2 != t:
                        if t < t2:
                            match_key = (t, t2)
                        else:
                            match_key = (t2, t)
                        
                        # This match contributes to week w for team t
                        weekly_matches.append(match_week[match_key] == w)
                
                solver.Add(solver.Sum(weekly_matches) == 1, f"once_per_week_t{t}_w{w}")
        
        # Constraint 2: Each period in each week has exactly one match
        # This is complex in MIP - we need auxiliary variables for logical AND
        week_period_match = {}  # week_period_match[w,p,t1,t2] = 1 if match (t1,t2) is in week w, period p
        
        for w in range(weeks):
            for p in range(periods):
                for t1 in range(1, n + 1):
                    for t2 in range(t1 + 1, n + 1):
                        match_key = (t1, t2)
                        wp_match_var = solver.BoolVar(f"wp_match_{w}_{p}_{t1}_{t2}")
                        week_period_match[w, p, match_key] = wp_match_var
                        
                        # wp_match_var = 1 iff (match_week[match_key] == w AND match_period[match_key] == p)
                        week_indicator = solver.BoolVar(f"week_ind_{w}_{t1}_{t2}")
                        period_indicator = solver.BoolVar(f"period_ind_{p}_{t1}_{t2}")
                        
                        # week_indicator = 1 iff match_week == w
                        solver.Add(week_indicator * weeks >= match_week[match_key] - w + 1)
                        solver.Add(week_indicator * weeks <= match_week[match_key] - w + weeks)
                        solver.Add((1 - week_indicator) * weeks >= w - match_week[match_key] + 1)
                        
                        # period_indicator = 1 iff match_period == p  
                        solver.Add(period_indicator * periods >= match_period[match_key] - p + 1)
                        solver.Add(period_indicator * periods <= match_period[match_key] - p + periods)
                        solver.Add((1 - period_indicator) * periods >= p - match_period[match_key] + 1)
                        
                        # wp_match_var = week_indicator AND period_indicator
                        solver.Add(wp_match_var <= week_indicator)
                        solver.Add(wp_match_var <= period_indicator)
                        solver.Add(wp_match_var >= week_indicator + period_indicator - 1)
                
                # Exactly one match in each (week, period) slot
                slot_matches = [week_period_match[w, p, (t1, t2)] for t1 in range(1, n + 1) for t2 in range(t1 + 1, n + 1)]
                solver.Add(solver.Sum(slot_matches) == 1, f"one_match_per_slot_w{w}_p{p}")
        
        # Constraint 3: Each team plays at most twice in the same period
        for t in range(1, n + 1):
            for p in range(periods):
                period_games = []
                for t2 in range(1, n + 1):
                    if t2 != t:
                        if t < t2:
                            match_key = (t, t2)
                        else:
                            match_key = (t2, t)
                        
                        # This match contributes to period p for team t
                        period_games.append(match_period[match_key] == p)
                
                solver.Add(solver.Sum(period_games) <= 2, f"at_most_twice_t{t}_p{p}")
        
        # Symmetry breaking: fix first match
        if n >= 2:
            first_match = (1, 2)
            solver.Add(match_week[first_match] == 0, "fix_first_week")
            solver.Add(match_period[first_match] == 0, "fix_first_period") 
            solver.Add(match_t1_home[first_match] == 1, "fix_first_home")
        
        # === SOLVE ===
        print(f"Solving with {solver.NumConstraints()} constraints, {solver.NumVariables()} variables")
        status = solver.Solve()
        elapsed_time = int(time.time() - start_time)
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # Extract solution and convert to required format
            sol = [[] for _ in range(periods)]
            
            for t1 in range(1, n + 1):
                for t2 in range(t1 + 1, n + 1):
                    match_key = (t1, t2)
                    week = int(match_week[match_key].solution_value())
                    period = int(match_period[match_key].solution_value())
                    t1_home = bool(match_t1_home[match_key].solution_value())
                    
                    if t1_home:
                        home_team, away_team = t1, t2
                    else:
                        home_team, away_team = t2, t1
                    
                    # Ensure sol[period] has enough weeks
                    while len(sol[period]) <= week:
                        sol[period].append([0, 0])
                    
                    sol[period][week] = [home_team, away_team]
            
            # Fill any missing slots
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
        print(f"Exception in match-based MIP solver: {e}")
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
            if home > 0 and away > 0:  # Valid game
                home_counts[home] += 1
                away_counts[away] += 1
    
    total_imbalance = sum(abs(home_counts[t] - away_counts[t]) for t in range(1, n + 1))
    return total_imbalance