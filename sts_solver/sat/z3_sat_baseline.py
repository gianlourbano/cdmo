"""
A baseline pure SAT model for the STS problem.

This implementation uses a 'sequential counter' encoding to translate all
cardinality constraints (e.g., "at most k", "exactly k") into pure SAT clauses.
"""
from z3 import *
from itertools import combinations
import time
from typing import Optional

from ..utils.solution_format import STSSolution


def at_most_k(solver, literals, k, prefix=""):
    """
    Encodes the Sum(literals) <= k constraint into pure SAT clauses.
    Implements the Sequential Counter encoding.
    """
    n = len(literals)
    if k < 0:
        for l in literals: solver.add(Not(l))
        return
    if n <= k:
        return  # Constraint is trivially true

    s = [[Bool(f"{prefix}_s_{i}_{j}") for j in range(k + 1)] for i in range(n)]

    solver.add(s[0][0] == literals[0])
    for j in range(1, k + 1):
        solver.add(Not(s[0][j]))

    for i in range(1, n):
        solver.add(s[i][0] == Or(s[i-1][0], literals[i]))
        for j in range(1, k + 1):
            solver.add(s[i][j] == Or(s[i-1][j], And(s[i-1][j-1], literals[i])))

    solver.add(Not(s[n-1][k]))

def at_least_k(solver, literals, k, prefix=""):
    """
    Encodes the Sum(literals) >= k constraint into pure SAT clauses.
    """
    if k <= 0: return
    n = len(literals)
    if n < k:
        solver.add(BoolVal(False))  # Impossible to satisfy
        return

    s = [[Bool(f"{prefix}_s_{i}_{j}") for j in range(k)] for i in range(n)]
    
    solver.add(s[0][0] == literals[0])
    for j in range(1, k):
        solver.add(Not(s[0][j]))
        
    for i in range(1, n):
        solver.add(s[i][0] == Or(s[i-1][0], literals[i]))
        for j in range(1, k):
            solver.add(s[i][j] == Or(s[i-1][j], And(s[i-1][j-1], literals[i])))

    solver.add(s[n-1][k-1])

def exactly_k(solver, literals, k, prefix=""):
    """
    Encodes the Sum(literals) == k constraint by combining AtMost and AtLeast.
    """
    at_most_k(solver, literals, k, prefix=f"{prefix}_am")
    at_least_k(solver, literals, k, prefix=f"{prefix}_al")

def solve_sat_baseline(
    n: int, 
    solver_name: Optional[str] = None, 
    timeout: int = 300,
    optimization: bool = False
) -> STSSolution:
    """
    Solves the STS problem using a pure SAT encoding with Z3.
    
    Args:
        n: Number of teams.
        solver_name: Solver variant name.
        timeout: Timeout in seconds.
        optimization: Flag for optimization (ignored in this SAT decision model).
        
    Returns:
        STSSolution object with the results.
    """
    num_weeks = n - 1
    num_periods = n // 2
    
    teams = range(1, n + 1)
    weeks = range(1, n) 
    periods = range(1, n // 2 + 1)
    
    start_time = time.time()
    
    s = Solver()
    s.set("timeout", timeout * 1000)  # Z3 timeout is in milliseconds
    
    try:
        # --- Decision Variables ---
        # A boolean variable for each team's potential home/away status in each slot.
        plays_home = {t: {w: {p: Bool(f"home_{t}_{w}_{p}") for p in periods} for w in weeks} for t in teams}
        plays_away = {t: {w: {p: Bool(f"away_{t}_{w}_{p}") for p in periods} for w in weeks} for t in teams}

        # --- Structural Constraints ---
        # For each slot, exactly one team plays home and one plays away.
        for w in weeks:
            for p in periods:
                home_teams = [plays_home[t][w][p] for t in teams]
                away_teams = [plays_away[t][w][p] for t in teams]
                exactly_k(s, home_teams, 1, prefix=f"h_slot_w{w}_p{p}")
                exactly_k(s, away_teams, 1, prefix=f"a_slot_w{w}_p{p}")
                # A team cannot play against itself.
                for t in teams:
                    s.add(Not(And(plays_home[t][w][p], plays_away[t][w][p])))

        # --- Tournament Constraints ---
        
        # Rule: "every team plays once a week"
        for t in teams:
            for w in weeks:
                weekly_app = [plays_home[t][w][p] for p in periods] + [plays_away[t][w][p] for p in periods]
                exactly_k(s, weekly_app, 1, prefix=f"weekly_t{t}_w{w}")

        # Rule: "every team plays with every other team only once"
        for t1, t2 in combinations(teams, 2):
            meetings = []
            for w in weeks:
                for p in periods:
                    meeting = Bool(f'meet_{t1}_{t2}_{w}_{p}')
                    s.add(meeting == Or(And(plays_home[t1][w][p], plays_away[t2][w][p]),
                                           And(plays_home[t2][w][p], plays_away[t1][w][p])))
                    meetings.append(meeting)
            exactly_k(s, meetings, 1, prefix=f"meet_{t1}_{t2}")

        # Rule: "every team plays at most twice in the same period"
        for t in teams:
            for p in periods:
                period_app = [plays_home[t][w][p] for w in weeks] + [plays_away[t][w][p] for w in weeks]
                at_most_k(s, period_app, 2, prefix=f"period_t{t}_p{p}")

        # --- Symmetry Breaking Constraints ---
        
        # Fix the first week
        for p in periods:
            home_team = 2 * p - 1
            away_team = 2 * p
            s.add(plays_home[home_team][1][p])
            s.add(plays_away[away_team][1][p])
        
        # --- Solve ---
        if s.check() == sat:
            model = s.model()
            elapsed_time = int(time.time() - start_time)
            
            # Extract solution
            sol = []
            for p in periods:
                period_games = []
                for w in weeks:
                    home_team, away_team = -1, -1
                    for t in teams:
                        if is_true(model.evaluate(plays_home[t][w][p], model_completion=True)):
                            home_team = t
                        if is_true(model.evaluate(plays_away[t][w][p], model_completion=True)):
                            away_team = t
                    period_games.append([home_team, away_team])
                sol.append(period_games)
            
            return STSSolution(
                time=elapsed_time,
                optimal=True,
                obj=None if not optimization else 1,
                sol=sol
            )
        else:
            # No solution found or timeout
            return STSSolution(
                time=timeout,
                optimal=False,
                obj=None,
                sol=None
            )
            
    except Exception:
        elapsed_time = int(time.time() - start_time)
        if elapsed_time >= timeout:
            elapsed_time = timeout
            
        return STSSolution(
            time=elapsed_time,
            optimal=False,
            obj=None,
            sol=None
        )