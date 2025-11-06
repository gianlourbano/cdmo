"""
Compact SMT solver for STS using Z3 with reduced variables and direct encoding
"""

import time
from z3 import *
from typing import Optional

from ..utils.solution_format import STSSolution
from .registry import register_smt_solver
import random

from typing import List, Tuple, Dict

from typing import List, Tuple, Dict

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

def bipartite_matchings(n):
    """
    Generate round-robin matchings using a backtracking approach to find a 1-factorization
    of the complete graph K_n. This is guaranteed to find a schedule.
    Returns a dictionary mapping week -> list of (team1, team2) matches.
    """
    if n % 2 != 0:
        raise ValueError("Number of teams must be even.")

    teams = list(range(1, n + 1))
    all_games = set()
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            all_games.add(tuple(sorted((teams[i], teams[j]))))

    schedule = {}
    num_weeks = n - 1

    # This is our main recursive function with backtracking
    def find_schedule(week_index, games_left):
        # Base case: If all weeks are scheduled, we are done
        if week_index == num_weeks:
            return True
        
        # Get the teams that are not yet scheduled in this week
        teams_to_schedule = list(range(1, n + 1))
        
        # Start finding a perfect matching for the current week
        return find_weekly_matching([], teams_to_schedule, games_left, week_index)

    def find_weekly_matching(weekly_matches, teams_to_schedule, games_left, week_index):
        # If no more teams need to be scheduled this week, this week is complete.
        # Move on to the next week.
        if not teams_to_schedule:
            schedule[week_index] = weekly_matches
            return find_schedule(week_index + 1, games_left)

        # Pick the first available team to find a match for
        team1 = teams_to_schedule[0]
        
        # Try to pair team1 with any other available team
        # We iterate on a copy since we might modify the list
        for i in range(1, len(teams_to_schedule)):
            team2 = teams_to_schedule[i]
            match = tuple(sorted((team1, team2)))

            # If this game is available to be scheduled
            if match in games_left:
                # Try this match
                new_weekly_matches = weekly_matches + [match]
                
                # Update the list of teams that still need a match this week
                new_teams_to_schedule = [t for t in teams_to_schedule if t != team1 and t != team2]
                
                # Recurse: try to complete the rest of the week's schedule
                if find_weekly_matching(new_weekly_matches, new_teams_to_schedule, games_left - {match}, week_index):
                    return True # Success!

        # Backtrack: If we get here, no valid matching could be found with the current path
        return False

    # Start the process from week 0
    find_schedule(0, all_games)
    return schedule

def circle_matchings(n):
    """
    Generate round-robin matchings using the circle method.
    Returns a dictionary mapping week -> list of (team1, team2) matches.
    """
    pivot, circle = n, list(range(1, n))
    weeks = n - 1
    m = {}
    
    for w in range(weeks):
        ms = [(pivot, circle[w])]
        for k in range(1, n // 2):
            i = circle[(w + k) % (n - 1)]
            j = circle[(w - k) % (n - 1)]
            ms.append((i, j))
        m[w] = ms
    return m


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


def solve_model(matches_per_week, periods, timeout=300):
    """
    Build and solve the SMT model using the exact efficient approach.
    """
    weeks = sorted(matches_per_week.keys())
    
    # Create period assignment variables p[w,i,j] ∈ [1..periods]
    p = {(w, i, j): Int(f"p_{w}_{i}_{j}")
         for w in weeks
         for (i, j) in matches_per_week[w]}

    # Use solver with cardinality optimization
    try:
        solver = Then('card2bv', 'smt').solver()
    except:
        solver = Solver()
    solver.set(timeout=timeout * 1000)

    # Domain constraints
    try:
        for var in p.values():
            solver.add(var >= 1, var <= periods)

        w1 = weeks[0]
        for k, (i, j) in enumerate(sorted(matches_per_week[w1])):
            solver.add(p[(w1, i, j)] == k + 1)

        # One match per slot per week (from MiniZinc constraint line 41)
        for w in weeks:
            for k in range(1, periods + 1):
                guards = [(p[(w, i, j)] == k, 1) for (i, j) in matches_per_week[w]]
                solver.add(PbEq(guards, 1))

        # At most max games per team per slot (from MiniZinc constraint line 25)
        teams = {t for w in weeks for (i, j) in matches_per_week[w] for t in (i, j)}
        for t in teams:
            for k in range(1, periods + 1):
                guards = [(p[(w, i, j)] == k, 1)
                        for w in weeks
                        for (i, j) in matches_per_week[w]
                        if t in (i, j)]
                solver.add(PbLe(guards, 2))

        # MiniZinc constraint: exactly 2 teams play once in each period (lines 82-84)
        if periods * 2 >= 22:
            for k in range(1, periods + 1):
                # Use pseudo-boolean constraints for efficiency
                deficient_guards = []
                for t in teams:
                    # count how many games team t plays in period k
                    games_count = 0
                    period_guards = []
                    for w in weeks:
                        for (i, j) in matches_per_week[w]:
                            if t in (i, j):
                                period_guards.append((p[(w, i, j)] == k, 1))
                                games_count += 1

                    if period_guards:
                        # team t is deficient if it plays exactly once in period k
                        if games_count == 1:
                            # If team only plays once total, it's always deficient in this period
                            deficient_guards.append((True, 1))
                        else:
                            # Create auxiliary boolean variable for deficiency
                            deficient_var = Bool(f"deficient_{t}_{k}")
                            # Constraint: deficient_var is true iff team plays exactly once
                            # Use reification with pseudo-boolean for efficiency
                            solver.add(deficient_var == (PbEq(period_guards, 1)))
                            deficient_guards.append((deficient_var, 1))

                # exactly 2 teams are deficient (play once) in each period
                if deficient_guards:
                    solver.add(PbEq(deficient_guards, 2))

        # MiniZinc constraint: exactly n_teams-2 teams play twice in each period (lines 87-89)
        if periods * 2 >= 22:
            for k in range(1, periods + 1):
                # Use pseudo-boolean constraints for efficiency
                double_guards = []
                for t in teams:
                    # count how many games team t plays in period k
                    games_count = 0
                    period_guards = []
                    for w in weeks:
                        for (i, j) in matches_per_week[w]:
                            if t in (i, j):
                                period_guards.append((p[(w, i, j)] == k, 1))
                                games_count += 1

                    if period_guards:
                        # team t plays twice if sum of guards == 2
                        if games_count == 2:
                            # If team plays exactly twice total, it always plays twice in this period
                            double_guards.append((True, 1))
                        else:
                            # Create auxiliary boolean variable for playing twice
                            double_var = Bool(f"double_{t}_{k}")
                            # Constraint: double_var is true iff team plays exactly twice
                            # Use reification with pseudo-boolean for efficiency
                            solver.add(double_var == (PbEq(period_guards, 2)))
                            double_guards.append((double_var, 1))

                # exactly n_teams-2 teams play twice in each period
                if double_guards:
                    solver.add(PbEq(double_guards, len(teams) - 2))

        # MiniZinc constraint: Every team is deficient in at most one period (lines 131-133)
        if periods * 2 >= 22:
            for t in teams:
                # Use pseudo-boolean constraints for efficiency
                deficiency_guards = []
                for k in range(1, periods + 1):
                    # count how many games team t plays in period k
                    games_count = 0
                    period_guards = []
                    for w in weeks:
                        for (i, j) in matches_per_week[w]:
                            if t in (i, j):
                                period_guards.append((p[(w, i, j)] == k, 1))
                                games_count += 1

                    if period_guards:
                        # team t is deficient in period k if it plays exactly once
                        if games_count == 1:
                            # If team only plays once total, it's always deficient
                            deficiency_guards.append((True, 1))
                        else:
                            # Create auxiliary boolean variable for deficiency in this period
                            deficient_var = Bool(f"def_period_{t}_{k}")
                            # Constraint: deficient_var is true iff team plays exactly once in period k
                            # Use reification with pseudo-boolean for efficiency
                            solver.add(deficient_var == (PbEq(period_guards, 1)))
                            deficiency_guards.append((deficient_var, 1))

                # each team is deficient in at most one period
                if deficiency_guards:
                    solver.add(PbLe(deficiency_guards, 1))

        solver.check()
        m = solver.model()

        # Build as weeks x periods
        sol = []
        for w in weeks:
            row = []
            for k in range(1, periods + 1):
                for (a, b) in matches_per_week[w]:
                    if m[p[(w, a, b)]].as_long() == k:
                        row.append([a, b])
                        break
            sol.append(row)
        return sol

    except Exception as e:
        return None

@register_smt_solver("presolve_3")
def solve_smt_compact_with_presolve(
    n: int, 
    solver_name: Optional[str] = None, 
    timeout: int = 300,
    optimization: bool = False
) -> STSSolution:
    """
    Solve STS using SMT with Z3 - Version with presolving using circle method
    
    This version exactly mirrors the efficient approach:
    - Circle method presolving to generate complete round-robin schedule
    - Home/away balance heuristic
    - Only period assignment variables (no week variables!)
    - Pseudo-Boolean constraints for efficiency
    """
    
    start_time = time.time()
    
    # Presolve: Generate complete schedule with home/away assignments
    balanced_matches = circle_matchings(n)
    #balanced_matches = home_away_balance(raw_matches, n)
    periods = n // 2

    presolve_time = time.time() - start_time
    if presolve_time >= timeout:
        return STSSolution(
            time=timeout,
            optimal=False,
            obj=None,
            sol=[]
        )

    # Solve using the efficient period labeling approach
    sol_weeks = solve_model(balanced_matches, periods, timeout=timeout-int(presolve_time))

    elapsed_time = int(time.time() - start_time)
    
    if sol_weeks is not None:
        # Transpose to periods x weeks format
        sol_periods = []
        num_weeks = len(sol_weeks)
        for p in range(periods):
            row = []
            for w in range(num_weeks):
                row.append(sol_weeks[w][p])
            sol_periods.append(row)
        
        return STSSolution(
            time=elapsed_time,
            optimal=True,
            obj=None if not optimization else calculate_balance_objective(sol_periods, n),
            sol=sol_periods
        )
    else:
        return STSSolution(
            time=min(elapsed_time, timeout),
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