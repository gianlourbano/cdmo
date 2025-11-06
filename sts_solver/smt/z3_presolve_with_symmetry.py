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
        # Symmetry breaking: increasing order of periods in week 1 (from MiniZinc line 70)
        week1_matches = sorted(matches_per_week[w1])
        for k, (i, j) in enumerate(week1_matches):
            solver.add(p[(w1, i, j)] == k + 1)

        # Additional symmetry breaking: ensure week1 periods are strictly increasing
        # This matches the MiniZinc "increasing(week1_periods)" constraint
        # for k in range(1, len(week1_matches)):
        #     prev_match = week1_matches[k-1]
        #     curr_match = week1_matches[k]
        #     solver.add(p[(w1, prev_match[0], prev_match[1])] < p[(w1, curr_match[0], curr_match[1])])

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

        # Period-based symmetry breaking constraints
        # These are more effective for SMT since we work directly with period assignments

        # 1. Period assignment ordering for team pairs (from MiniZinc symmetry principles)
        # For any two teams (i,j), the sequence of periods they play in should be ordered
        teams_list = sorted(teams)
        for i in range(len(teams_list)):
            for j in range(i + 1, len(teams_list)):
                team1, team2 = teams_list[i], teams_list[j]

                # Get all weeks where these two teams play (should be exactly one in round-robin)
                team_pair_periods = []
                for w in weeks:
                    for (a, b) in matches_per_week[w]:
                        if (team1 == a and team2 == b) or (team1 == b and team2 == a):
                            team_pair_periods.append(p[(w, a, b)])

                # For single round-robin, there should be exactly one match between each pair
                if len(team_pair_periods) == 1:
                    # This constraint is already handled by the week 1 ordering above
                    pass

        # 2. Lexicographic ordering of team period schedules
        # Team 1 should have the "lexicographically smallest" period assignment sequence
        # for team_idx in range(1, len(teams_list)):
        #     prev_team = teams_list[team_idx - 1]
        #     curr_team = teams_list[team_idx]

        #     # Get period assignments for each team across all weeks
        #     prev_team_periods = []
        #     curr_team_periods = []

        #     for w in weeks:
        #         for (i, j) in matches_per_week[w]:
        #             if prev_team in (i, j):
        #                 prev_team_periods.append(p[(w, i, j)])
        #             if curr_team in (i, j):
        #                 curr_team_periods.append(p[(w, i, j)])

        #     # Add lexicographic constraint: previous team <= current team
        #     # Compare period sequences lexicographically
        #     if len(prev_team_periods) == len(curr_team_periods):
        #         for k in range(len(prev_team_periods)):
        #             # Create implication: if previous periods are equal so far, current <= next
        #             if k == 0:
        #                 solver.add(prev_team_periods[k] <= curr_team_periods[k])
        #             else:
        #                 # If all previous periods are equal, then current period <= next
        #                 all_equal_prev = And([prev_team_periods[i] == curr_team_periods[i] for i in range(k)])
        #                 implies_current_leq = Implies(all_equal_prev, prev_team_periods[k] <= curr_team_periods[k])
        #                 solver.add(implies_current_leq)

        # # 3. Period usage minimization for early periods
        # # Encourage using smaller period numbers first to break symmetry
        # for k in range(1, periods + 1):
        #     period_usage = []
        #     for w in weeks:
        #         for (i, j) in matches_per_week[w]:
        #             period_usage.append(p[(w, i, j)] == k)

        #     # Early periods should be used at least as much as later periods
        #     if k < periods:
        #         next_period_usage = []
        #         for w in weeks:
        #             for (i, j) in matches_per_week[w]:
        #                 next_period_usage.append(p[(w, i, j)] == (k + 1))

        #         if period_usage and next_period_usage:
        #             # Use simple pseudo-boolean constraint for period usage comparison
        #             # Early periods should be used at least as much as later periods

        #             period_k_guards = [(guard, 1) for guard in period_usage]
        #             period_k1_guards = [(guard, 1) for guard in next_period_usage]

        #             # The constraint |period_k| >= |period_{k+1}| can be expressed as:
        #             # For every subset of size |period_{k+1}| + 1 from period_k_guards,
        #             # at least one guard must be false.

        #             # Simpler approach: Use the property that if A >= B, then
        #             # there's no subset of A with size < B that can cover all B elements

        #             # We'll use a practical encoding: for each period k+1 usage,
        #             # ensure there are enough period k usages
        #             if len(period_k1_guards) > 0 and len(period_k_guards) > 0:
        #                 # Create auxiliary boolean for the comparison
        #                 enough_usage = Bool(f"enough_usage_{k}")

        #                 # The constraint holds if either:
        #                 # 1. There are no period k+1 usages, or
        #                 # 2. There are at least as many period k usages as period k+1 usages

        #                 # Use pseudo-boolean constraint: at most |period_k| guards of period k+1 can be true
        #                 # This is equivalent to: sum(period_k1_guards) <= len(period_k_guards)
        #                 solver.add(PbLe(period_k1_guards, len(period_k_guards)))

        #                 # Additionally, if we want strict inequality for some k:
        #                 if k <= periods // 2:  # For early periods, encourage more usage
        #                     # At least one period k should be used if any period k+1 is used
        #                     if len(period_k1_guards) > 0:
        #                         # If any period k+1 is used, then at least one period k must be used
        #                         any_k1_used = Or(period_k1_guards)
        #                         any_k_used = Or(period_k_guards)
        #                         solver.add(Implies(any_k1_used, any_k_used))

        # 4. Fix specific team-period assignments to break remaining symmetries
        # Team 1 should play in period 1 in week 1
        # for (i, j) in week1_matches:
        #     if 1 in (i, j):
        #         solver.add(p[(w1, i, j)] == 1)
        #         break

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

@register_smt_solver("presolve_symmetry")
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