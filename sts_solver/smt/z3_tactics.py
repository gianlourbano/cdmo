"""
Tactics-focused SMT solver for STS using Z3 with different solving strategies
"""

import time
from z3 import *
from typing import Optional

from ..utils.solution_format import STSSolution


def solve_smt_tactics(
    n: int, 
    solver_name: Optional[str] = None, 
    timeout: int = 300,
    optimization: bool = False
) -> STSSolution:
    """
    Solve STS using SMT with Z3 - Multiple tactics approach
    
    This version tries different Z3 tactics based on instance size:
    - Small instances (n <= 8): SAT-based approach
    - Medium instances (n <= 12): Hybrid SMT approach  
    - Large instances (n > 12): Preprocessing-heavy approach
    """
    
    weeks = n - 1
    periods = n // 2
    
    start_time = time.time()
    
    # Choose tactic based on instance size
    if n <= 8:
        return solve_with_sat_tactics(n, timeout, optimization, start_time)
    elif n <= 12:
        return solve_with_hybrid_tactics(n, timeout, optimization, start_time)
    else:
        return solve_with_preprocessing_tactics(n, timeout, optimization, start_time)


def solve_with_sat_tactics(n, timeout, optimization, start_time):
    """Use simpler tactics for small instances"""
    
    weeks = n - 1
    periods = n // 2
    
    # Use simpler tactics that work better
    s = Solver()
    s.set("timeout", timeout * 1000)
    
    schedule = create_standard_variables(s, n, weeks, periods)
    add_standard_constraints(s, schedule, n, weeks, periods)
    
    return solve_and_extract(s, schedule, n, weeks, periods, timeout, optimization, start_time)


def solve_with_hybrid_tactics(n, timeout, optimization, start_time):
    """Use basic tactics for medium instances"""
    
    weeks = n - 1
    periods = n // 2
    
    # Use basic tactics
    tactics = Then('simplify', 'smt')
    s = tactics.solver()
    s.set("timeout", timeout * 1000)
    
    schedule = create_standard_variables(s, n, weeks, periods)
    add_standard_constraints(s, schedule, n, weeks, periods)
    add_advanced_symmetry_breaking(s, schedule, n, weeks, periods)
    
    return solve_and_extract(s, schedule, n, weeks, periods, timeout, optimization, start_time)


def solve_with_preprocessing_tactics(n, timeout, optimization, start_time):
    """Use preprocessing tactics for large instances"""
    
    weeks = n - 1
    periods = n // 2
    
    # Use moderate preprocessing
    tactics = Then('simplify', 'propagate-values', 'smt')
    s = tactics.solver()
    s.set("timeout", timeout * 1000)
    
    schedule = create_standard_variables(s, n, weeks, periods)
    add_standard_constraints(s, schedule, n, weeks, periods)
    add_advanced_symmetry_breaking(s, schedule, n, weeks, periods)
    add_implied_constraints(s, schedule, n, weeks, periods)
    
    return solve_and_extract(s, schedule, n, weeks, periods, timeout, optimization, start_time)


def create_standard_variables(s, n, weeks, periods):
    """Create standard schedule variables"""
    schedule = {}
    for w in range(weeks):
        for p in range(periods):
            for slot in range(2):
                schedule[w, p, slot] = Int(f"s_{w}_{p}_{slot}")
                s.add(schedule[w, p, slot] >= 1)
                s.add(schedule[w, p, slot] <= n)
    return schedule


def add_standard_constraints(s, schedule, n, weeks, periods):
    """Add basic STS constraints"""
    
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
                    games.append(And(schedule[w, p, 0] == t1, schedule[w, p, 1] == t2))
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


def add_advanced_symmetry_breaking(s, schedule, n, weeks, periods):
    """Add advanced symmetry breaking constraints"""
    
    # Basic symmetry breaking
    s.add(schedule[0, 0, 0] == 1)
    
    # Fix first opponent
    s.add(schedule[0, 0, 1] == 2)
    
    # Order teams in first week periods
    for p in range(min(periods - 1, n // 2 - 1)):
        s.add(schedule[0, p, 0] < schedule[0, p + 1, 0])
    
    # Lexicographic ordering for first few weeks
    if weeks >= 2:
        for p in range(periods):
            s.add(schedule[0, p, 0] <= schedule[1, p, 0])


def add_implied_constraints(s, schedule, n, weeks, periods):
    """Add implied constraints for better pruning"""
    
    # Each team must play exactly n-1 games total
    for t in range(1, n + 1):
        total_games = []
        for w in range(weeks):
            for p in range(periods):
                for slot in range(2):
                    total_games.append(If(schedule[w, p, slot] == t, 1, 0))
        s.add(Sum(total_games) == weeks)
    
    # Balance home/away games as much as possible
    for t in range(1, n + 1):
        home_games = []
        away_games = []
        for w in range(weeks):
            for p in range(periods):
                home_games.append(If(schedule[w, p, 0] == t, 1, 0))
                away_games.append(If(schedule[w, p, 1] == t, 1, 0))
        
        total_home = Sum(home_games)
        total_away = Sum(away_games)
        
        # Difference should be at most 1
        s.add(Abs(total_home - total_away) <= 1)


def solve_and_extract(s, schedule, n, weeks, periods, timeout, optimization, start_time):
    """Solve and extract solution"""
    
    try:
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
    home_counts = [0] * (n + 1)
    away_counts = [0] * (n + 1)
    
    for period_games in sol:
        for home, away in period_games:
            home_counts[home] += 1
            away_counts[away] += 1
    
    total_imbalance = sum(abs(home_counts[t] - away_counts[t]) for t in range(1, n + 1))
    return total_imbalance