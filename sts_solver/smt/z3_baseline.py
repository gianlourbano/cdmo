"""
Baseline SMT solver for STS using Z3
"""

import time
from z3 import *
from typing import Optional

from ..utils.solution_format import STSSolution
from .registry import register_smt_solver


@register_smt_solver("baseline")
def solve_smt(
    n: int, 
    solver_name: Optional[str] = None, 
    timeout: int = 300,
    optimization: bool = False
) -> STSSolution:
    """
    Solve STS using SMT with Z3
    
    Args:
        n: Number of teams
        solver_name: Z3 solver variant (not used for SMT)
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
        # Decision variables: schedule[w][p][slot] = team playing in week w, period p, slot
        schedule = {}
        for w in range(weeks):
            for p in range(periods):
                for slot in range(2):  # 0=home, 1=away
                    schedule[w, p, slot] = Int(f"schedule_{w}_{p}_{slot}")
                    s.add(schedule[w, p, slot] >= 1)
                    s.add(schedule[w, p, slot] <= n)
        
        # Constraint 1: No team plays against itself
        for w in range(weeks):
            for p in range(periods):
                s.add(schedule[w, p, 0] != schedule[w, p, 1])
        
        # Constraint 2: Every team plays once per week
        for w in range(weeks):
            for t in range(1, n + 1):
                appearances = []
                for p in range(periods):
                    for slot in range(2):
                        appearances.append(If(schedule[w, p, slot] == t, 1, 0))
                s.add(Sum(appearances) == 1)
        
        # Constraint 3: Every team plays with every other team exactly once
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
                appearances = []
                for w in range(weeks):
                    for slot in range(2):
                        appearances.append(If(schedule[w, p, slot] == t, 1, 0))
                s.add(Sum(appearances) <= 2)
        
        # Optional: Symmetry breaking
        # Fix team 1 to play at home in the first period of the first week
        s.add(schedule[0, 0, 0] == 1)
        
        # Solve
        if s.check() == sat:
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