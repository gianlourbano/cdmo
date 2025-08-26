"""
Compact SMT solver for STS using Z3 with reduced variables and direct encoding
"""

import time
from z3 import *
from typing import Optional

from ..utils.solution_format import STSSolution


def solve_smt_compact(
    n: int, 
    solver_name: Optional[str] = None, 
    timeout: int = 300,
    optimization: bool = False
) -> STSSolution:
    """
    Solve STS using SMT with Z3 - Compact version with fewer variables
    
    This version uses:
    - Direct match variables instead of schedule slots
    - Boolean variables for team matchups
    - More efficient constraint encoding
    """
    
    weeks = n - 1
    periods = n // 2
    total_matches = n * (n - 1) // 2
    
    start_time = time.time()
    
    # Create Z3 solver
    s = Solver()
    s.set("timeout", timeout * 1000)
    
    try:
        # Primary variables: match[t1][t2] = (week, period, is_t1_home)
        # where t1 < t2 and is_t1_home indicates if t1 plays at home
        matches = {}
        match_week = {}
        match_period = {}
        match_t1_home = {}  # Boolean: True if t1 plays at home against t2
        
        for t1 in range(1, n + 1):
            for t2 in range(t1 + 1, n + 1):
                match_key = (t1, t2)
                match_week[match_key] = Int(f"week_{t1}_{t2}")
                match_period[match_key] = Int(f"period_{t1}_{t2}")
                match_t1_home[match_key] = Bool(f"home_{t1}_{t2}")
                
                # Domain constraints
                s.add(match_week[match_key] >= 0)
                s.add(match_week[match_key] < weeks)
                s.add(match_period[match_key] >= 0)
                s.add(match_period[match_key] < periods)
        
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
                        
                        weekly_matches.append(If(match_week[match_key] == w, 1, 0))
                
                s.add(Sum(weekly_matches) == 1)
        
        # Constraint 2: Each period in each week has exactly the right number of matches
        for w in range(weeks):
            for p in range(periods):
                period_matches = []
                for t1 in range(1, n + 1):
                    for t2 in range(t1 + 1, n + 1):
                        match_key = (t1, t2)
                        period_matches.append(
                            If(And(match_week[match_key] == w, match_period[match_key] == p), 1, 0)
                        )
                
                s.add(Sum(period_matches) == 1)  # Exactly one match per period
        
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
                        
                        period_games.append(If(match_period[match_key] == p, 1, 0))
                
                s.add(Sum(period_games) <= 2)
        
        # Symmetry breaking
        # Fix the first match: team 1 vs team 2 in week 0, period 0, team 1 at home
        if n >= 2:
            first_match = (1, 2)
            s.add(match_week[first_match] == 0)
            s.add(match_period[first_match] == 0)
            s.add(match_t1_home[first_match] == True)
        
        # Additional symmetry breaking: order matches in first week by team numbers
        first_week_matches = []
        for t1 in range(1, n + 1):
            for t2 in range(t1 + 1, n + 1):
                match_key = (t1, t2)
                first_week_matches.append((match_key, t1))
        
        # Sort first week matches by team number and assign periods in order
        first_week_matches.sort(key=lambda x: x[1])
        for i, (match_key, _) in enumerate(first_week_matches[:periods]):
            if match_key != (1, 2):  # Skip the already fixed match
                s.add(Implies(match_week[match_key] == 0, match_period[match_key] == i))
        
        # Solve
        result = s.check()
        
        if result == sat:
            model = s.model()
            elapsed_time = int(time.time() - start_time)
            
            # Extract solution and convert to required format
            sol = [[] for _ in range(periods)]
            
            for t1 in range(1, n + 1):
                for t2 in range(t1 + 1, n + 1):
                    match_key = (t1, t2)
                    week = model.eval(match_week[match_key]).as_long()
                    period = model.eval(match_period[match_key]).as_long()
                    t1_home = model.eval(match_t1_home[match_key])
                    
                    if is_true(t1_home):
                        home_team, away_team = t1, t2
                    else:
                        home_team, away_team = t2, t1
                    
                    # Ensure sol[period] has enough weeks
                    while len(sol[period]) <= week:
                        sol[period].append([0, 0])
                    
                    sol[period][week] = [home_team, away_team]
            
            # Fill any missing slots (should not happen with correct constraints)
            for p in range(periods):
                while len(sol[p]) < weeks:
                    sol[p].append([0, 0])
            
            return STSSolution(
                time=elapsed_time,
                optimal=True,
                obj=None if not optimization else calculate_balance_objective(sol, n),
                sol=sol
            )
        else:
            return STSSolution(
                time=timeout,
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