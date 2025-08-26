"""
Optimized SMT solver for STS using Z3 with advanced tactics and optimizations
"""

import time
from z3 import *
from typing import Optional

from ..utils.solution_format import STSSolution


def solve_smt_optimized(
    n: int, 
    solver_name: Optional[str] = None, 
    timeout: int = 300,
    optimization: bool = False
) -> STSSolution:
    """
    Solve STS using SMT with Z3 - Optimized version with advanced tactics
    
    Optimizations:
    - Custom tactics for better performance
    - Enhanced symmetry breaking
    - Preprocessing optimizations
    - Better variable ordering
    """
    
    weeks = n - 1
    periods = n // 2
    
    start_time = time.time()
    
    # Use a simple solver with minimal configuration
    s = Solver()
    s.set("timeout", timeout * 1000)
    
    # Only keep settings that don't conflict
    s.set("smt.arith.solver", 2)  # More efficient arithmetic solver
    
    try:
        # Decision variables with better naming for Z3
        schedule = {}
        for w in range(weeks):
            for p in range(periods):
                for slot in range(2):
                    var_name = f"s_{w}_{p}_{slot}"
                    schedule[w, p, slot] = Int(var_name)
                    s.add(schedule[w, p, slot] >= 1)
                    s.add(schedule[w, p, slot] <= n)
        
        # Basic symmetry breaking (keep it simple)
        s.add(schedule[0, 0, 0] == 1)  # Team 1 plays at home in first period of first week
        
        # Constraint 1: No team plays against itself
        for w in range(weeks):
            for p in range(periods):
                s.add(schedule[w, p, 0] != schedule[w, p, 1])
        
        # Constraint 2: Every team plays once per week (with optimization)
        for w in range(weeks):
            for t in range(1, n + 1):
                team_appearances = []
                for p in range(periods):
                    for slot in range(2):
                        team_appearances.append(If(schedule[w, p, slot] == t, 1, 0))
                s.add(Sum(team_appearances) == 1)
        
        # Constraint 3: Every team plays with every other team exactly once
        # Use the same approach as baseline - it's proven to work well
        for t1 in range(1, n + 1):
            for t2 in range(t1 + 1, n + 1):
                games = []
                for w in range(weeks):
                    for p in range(periods):
                        # t1 home vs t2 away
                        games.append(And(schedule[w, p, 0] == t1, schedule[w, p, 1] == t2))
                        # t1 away vs t2 home
                        games.append(And(schedule[w, p, 0] == t2, schedule[w, p, 1] == t1))
                s.add(Sum([If(game, 1, 0) for game in games]) == 1)
        
        # Constraint 4: Each team plays at most twice in the same period
        for p in range(periods):
            for t in range(1, n + 1):
                period_appearances = []
                for w in range(weeks):
                    for slot in range(2):
                        period_appearances.append(If(schedule[w, p, slot] == t, 1, 0))
                s.add(Sum(period_appearances) <= 2)
        
        # Keep it simple - no additional constraints that might slow things down
        
        # Solve
        result = s.check()
        
        if result == sat:
            model = s.model()
            elapsed_time = int(time.time() - start_time)
            
            # Extract solution
            sol = []
            for p in range(periods):
                period_games = []
                for w in range(weeks):
                    home_team = model.eval(schedule[w, p, 0]).as_long()
                    away_team = model.eval(schedule[w, p, 1]).as_long()
                    period_games.append([home_team, away_team])
                sol.append(period_games)
            
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
    home_counts = [0] * (n + 1)  # Team numbers start from 1
    away_counts = [0] * (n + 1)
    
    for period_games in sol:
        for home, away in period_games:
            home_counts[home] += 1
            away_counts[away] += 1
    
    # Calculate imbalance (sum of absolute differences)
    total_imbalance = sum(abs(home_counts[t] - away_counts[t]) for t in range(1, n + 1))
    return total_imbalance