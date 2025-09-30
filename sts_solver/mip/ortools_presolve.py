import time
from typing import Optional, List, Dict, Tuple
from ortools.linear_solver import pywraplp

from ..utils.solution_format import STSSolution

def schedule_iterative_divide_and_conquer(n: int) -> Dict[int, List[Tuple[int, int]]]:
    """
    Generates a round-robin tournament schedule for n teams using an iterative,
    bottom-up version of the divide-and-conquer algorithm. This avoids
    recursion depth errors and is more efficient.

    Args:
        n: The number of teams. Must be an even number.

    Returns:
        A dictionary where keys are week numbers (1-indexed) and values are
        lists of (team1, team2) game tuples.
    """
    if n == 0: return {}
    teams = list(range(1, n + 1))
    schedule = {}
    
    for week in range(1, n):
        schedule[week] = []
        # Pair team `n` with a rotating opponent
        opponent_for_n = teams[(week - 1) % (n - 1)]
        schedule[week].append(tuple(sorted((n, opponent_for_n))))
        
        # Pair up the remaining n-2 teams
        for i in range(1, n // 2):
            team1_idx = (week - 1 + i) % (n - 1)
            team2_idx = (week - 1 - i + (n - 1)) % (n - 1) # Ensure positive index
            
            team1 = teams[team1_idx]
            team2 = teams[team2_idx]
            schedule[week].append(tuple(sorted((team1, team2))))
            
    return schedule


def home_away_balance(matches_per_week, n):
    """
    Apply heuristic for home/away balance based on team number differences.
    """
    balanced = {}
    for w, matches in matches_per_week.items():
        row = []
        for (i, j) in matches:
            d = (j - i) % n
            row.append((i, j) if d < n // 2 else (j, i))
        balanced[w] = row
    return balanced

def presolve_sts(n:int,
                 solver_name: Optional[str] = None,
                 timeout: int = 300, optimization: bool = False) -> STSSolution:
    if n % 2 != 0:
        return STSSolution(time=0, optimal=False, obj=None, sol=[])
    
    solver = pywraplp.Solver.CreateSolver('SCIP')
    solver.SetTimeLimit(timeout * 1000)  # milliseconds

    if not solver:
        return STSSolution(time=0, optimal=False, obj=None, sol=[])
    
    #solver.setNumThreads(1)
    #solver.setTimeLimit(timeout * 1000) 

    # apply circle method to generate initial schedule
    start_time = time.time()
    weeks = n - 1
    periods = n // 2

    games = schedule_iterative_divide_and_conquer(n)
    matches_per_week = home_away_balance(games, n)

    weeks = sorted(matches_per_week.keys())
    teams=list(range(1, n + 1))

    # after having precomputed the matches, we only need to assign them to periods, so the only contraint we have
    # is that no team can play more than twice in the same period

    # === VARIABLES ===
    # x[w, p, i, j] = 1 if match (i,j) in week w is assigned to period p.
    # Note: (i,j) are already fixed for a given week w by the presolver.
    x = {}
    for w in weeks:
        for p in range(1, periods + 1):
            for i, j in matches_per_week[w]:
                x[w, p, i, j] = solver.BoolVar(f"x_{w}_{p}_{i}_{j}")

    # === CONSTRAINTS ===

    # 1. Each pre-solved match must be assigned to exactly one period.
    for w in weeks:
        for i, j in matches_per_week[w]:
            solver.Add(solver.Sum(x[w, p, i, j] for p in range(1, periods + 1)) == 1, f"match_once_{w}_{i}_{j}")

    # 2. In each week, each period can only have one match.
    for w in weeks:
        for p in range(1, periods + 1):
            solver.Add(solver.Sum(x[w, p, i, j] for i, j in matches_per_week[w]) == 1, f"slot_once_{w}_{p}")

    # 3. Each team plays at most twice in any given period across all weeks.
    for t in teams:
        for p in range(1, periods + 1):
            team_appearances = []
            for w in weeks:
                for i, j in matches_per_week[w]:
                    if t == i or t == j:
                        team_appearances.append(x[w, p, i, j])
            solver.Add(solver.Sum(team_appearances) <= 2, f"period_limit_{t}_{p}")
            
    # === SYMMETRY BREAKING (Optional but helpful) ===
    w1 = weeks[0]
    for k, (i, j) in enumerate(sorted(matches_per_week[w1])):
        # Assign match k of week 1 to period k+1
        solver.Add(x[w1, k + 1, i, j] == 1)

    # === SOLVE ===
    status = solver.Solve()
    elapsed_time = int(time.time() - start_time)

     # === EXTRACT SOLUTION ===
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        sol_weeks = [[] for _ in weeks]
        for w_idx, w in enumerate(weeks):
            week_schedule = [[] for _ in range(periods)]
            for p in range(1, periods + 1):
                for i, j in matches_per_week[w]:
                    if x[w, p, i, j].solution_value() > 0.5:
                        week_schedule[p-1] = [i, j]
                        break
            sol_weeks[w_idx] = week_schedule
        
        # Transpose to periods x weeks format
        sol_periods = list(zip(*sol_weeks))

        return STSSolution(
            time=elapsed_time,
            optimal=(status == pywraplp.Solver.OPTIMAL),
            obj=None, # Objective can be added here if needed
            sol=sol_periods
        )
    else:
        return STSSolution(
            time=min(elapsed_time, timeout),
            optimal=False,
            obj=None,
            sol=[]
        )
