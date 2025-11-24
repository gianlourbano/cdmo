import time
from typing import Any, List, Dict, Tuple
from ortools.linear_solver import pywraplp

from ..utils.solution_format import STSSolution
from .base import MIPBaseSolver
from ..base_solver import SolverMetadata

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

class MIPPresolveSolver(MIPBaseSolver):
    """Presolve-based formulation using deterministic round-robin schedule.
    Name: 'presolve'. Supports optimization (home/away imbalance minimization).
    """

    def _build_model(self) -> Any:
        return None

    def _solve_model(self, model: Any) -> STSSolution:
        build_start = time.time()
        n = self.n
        backend = (self.backend or "SCIP").upper()
        solver = pywraplp.Solver.CreateSolver(backend)
        if not solver:
            return STSSolution(time=0, optimal=False, obj=None, sol=[])
        solver.SetTimeLimit(self.timeout * 1000)
        weeks = n - 1
        periods = n // 2
        
        # Generate schedule
        matches_per_week = schedule_iterative_divide_and_conquer(n)
        #matches_per_week = home_away_balance(games, n)
        week_keys = sorted(matches_per_week.keys())
        teams = list(range(1, n + 1))
        
        # Pre-compute which matches involve each team (optimization)
        team_matches = {t: [] for t in teams}
        for w in week_keys:
            for i, j in matches_per_week[w]:
                team_matches[i].append((w, i, j))
                team_matches[j].append((w, i, j))
        
        # Create variables
        x = {}
        for w in week_keys:
            for p in range(1, periods + 1):
                for i, j in matches_per_week[w]:
                    x[w, p, i, j] = solver.BoolVar(f"x_{w}_{p}_{i}_{j}")
        
        # Constraints
        # Each match assigned to exactly one period
        for w in week_keys:
            for i, j in matches_per_week[w]:
                solver.Add(solver.Sum(x[w, p, i, j] for p in range(1, periods + 1)) == 1)
        
        # Each period has exactly one match per week
        for w in week_keys:
            for p in range(1, periods + 1):
                solver.Add(solver.Sum(x[w, p, i, j] for i, j in matches_per_week[w]) == 1)
        
        # Each team plays at most twice in same period (optimized)
        for t in teams:
            for p in range(1, periods + 1):
                # Use pre-computed team_matches instead of nested loops
                team_apps = [x[w, p, i, j] for w, i, j in team_matches[t]]
                solver.Add(solver.Sum(team_apps) <= 2)
        
        # Optimization: minimize home/away imbalance
        if self.optimization:
            # Count home and away games for each team
            home_count = {}
            away_count = {}
            for t in teams:
                home_games = []
                away_games = []
                for w in week_keys:
                    for i, j in matches_per_week[w]:
                        for p in range(1, periods + 1):
                            if i == t:  # t plays at home
                                home_games.append(x[w, p, i, j])
                            elif j == t:  # t plays away
                                away_games.append(x[w, p, i, j])
                home_count[t] = solver.Sum(home_games) if home_games else 0
                away_count[t] = solver.Sum(away_games) if away_games else 0
            
            # Create imbalance variables for each team
            imbalance = {}
            for t in teams:
                imbalance[t] = solver.NumVar(0, n - 1, f"imbalance_{t}")
                # imbalance[t] >= |home_count[t] - away_count[t]|
                solver.Add(imbalance[t] >= home_count[t] - away_count[t])
                solver.Add(imbalance[t] >= away_count[t] - home_count[t])
            
            # Minimize maximum imbalance across all teams
            max_imbalance = solver.NumVar(0, n - 1, "max_imbalance")
            for t in teams:
                solver.Add(max_imbalance >= imbalance[t])
            
            solver.Minimize(max_imbalance)
        
        build_time = time.time() - build_start
        # If model building took too long, return early
        if build_time > self.timeout:
            return STSSolution(time=int(build_time), optimal=False, obj=None, sol=[])
        # Symmetry: first week ordering
        # w1 = week_keys[0]
        # for k, (i, j) in enumerate(sorted(matches_per_week[w1])):
        #     solver.Add(x[w1, k + 1, i, j] == 1)
        status = solver.Solve()
        elapsed = self.elapsed_time
        if status not in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE}:
            return STSSolution(time=min(elapsed, self.timeout), optimal=False, obj=None, sol=[])
        
        # Extract solution
        sol_weeks = [[] for _ in week_keys]
        for idx, w in enumerate(week_keys):
            week_sched = [[] for _ in range(periods)]
            for p in range(1, periods + 1):
                for i, j in matches_per_week[w]:
                    if x[w, p, i, j].solution_value() > 0.5:
                        week_sched[p - 1] = [i, j]
                        break
            sol_weeks[idx] = week_sched
        sol_periods = list(zip(*sol_weeks))
        sol_periods = [list(games) for games in sol_periods]
        
        # Extract objective value if optimization enabled
        obj_value = None
        if self.optimization and solver.Objective():
            obj_value = int(solver.Objective().Value())
        
        return STSSolution(
            time=min(elapsed, self.timeout),
            optimal=(status == pywraplp.Solver.OPTIMAL),
            obj=obj_value,
            sol=sol_periods,
        )

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="presolve",
            approach="MIP",
            version="1.0",
            supports_optimization=True,
            description="Presolve formulation assigning precomputed matches to periods; supports imbalance minimization",
        )
