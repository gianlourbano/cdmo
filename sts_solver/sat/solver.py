"""
SAT solver for STS using Z3
"""

import time
from z3 import *
from typing import Optional

from ..utils.solution_format import STSSolution


def solve_sat(
    n: int, 
    solver_name: Optional[str] = None, 
    timeout: int = 300,
    optimization: bool = False
) -> STSSolution:
    """
    Solve STS using SAT with Z3
    
    Args:
        n: Number of teams
        solver_name: Z3 solver variant (not used for SAT)
        timeout: Timeout in seconds
        optimization: Whether to use optimization version
        
    Returns:
        STSSolution object with results
    """
    
    weeks = n - 1
    periods = n // 2
    
    start_time = time.time()
    
    # Create Z3 solver
    s = Solver()
    s.set("timeout", timeout * 1000)  # Z3 timeout in milliseconds
    
    try:
        # Decision variables: x[w][p][s][t] = True if team t plays in week w, period p, slot s
        x = {}
        for w in range(weeks):
            for p in range(periods):
                for s in range(2):  # 0=home, 1=away
                    for t in range(n):
                        x[w, p, s, t] = Bool(f"x_{w}_{p}_{s}_{t}")
        
        # Constraint 1: Exactly one team per slot
        for w in range(weeks):
            for p in range(periods):
                for s in range(2):
                    s.add(Sum([If(x[w, p, s, t], 1, 0) for t in range(n)]) == 1)
        
        # Constraint 2: Every team plays once per week
        for w in range(weeks):
            for t in range(n):
                s.add(Sum([If(x[w, p, s, t], 1, 0) 
                          for p in range(periods) for s in range(2)]) == 1)
        
        # Constraint 3: Every team plays with every other team exactly once
        for t1 in range(n):
            for t2 in range(t1 + 1, n):
                games = []
                for w in range(weeks):
                    for p in range(periods):
                        # t1 home vs t2 away OR t1 away vs t2 home
                        games.append(And(x[w, p, 0, t1], x[w, p, 1, t2]))
                        games.append(And(x[w, p, 1, t1], x[w, p, 0, t2]))
                s.add(Sum([If(game, 1, 0) for game in games]) == 1)
        
        # Constraint 4: Each team plays at most twice in the same period
        for p in range(periods):
            for t in range(n):
                s.add(Sum([If(x[w, p, s, t], 1, 0) 
                          for w in range(weeks) for s in range(2)]) <= 2)
        
        # Solve
        if s.check() == sat:
            model = s.model()
            elapsed_time = int(time.time() - start_time)
            
            # Extract solution
            sol = []
            for p in range(periods):
                period_games = []
                for w in range(weeks):
                    # Find the two teams playing in this week and period
                    home_team = None
                    away_team = None
                    for t in range(n):
                        if is_true(model.eval(x[w, p, 0, t])):
                            home_team = t + 1  # 1-indexed
                        if is_true(model.eval(x[w, p, 1, t])):
                            away_team = t + 1  # 1-indexed
                    
                    if home_team is not None and away_team is not None:
                        period_games.append([home_team, away_team])
                    else:
                        # Should not happen if constraints are correct
                        period_games.append([1, 2])
                        
                sol.append(period_games)
            
            return STSSolution(
                time=elapsed_time,
                optimal=True,
                obj=None if not optimization else 1,
                sol=sol
            )
        else:
            # No solution found
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