import time
import math
from itertools import combinations
from typing import Any, Dict, Tuple
from z3 import Solver, Bool, Not, Or, And, BoolVal, is_true, sat, Implies

from ..utils.solution_format import STSSolution
from .base import SATBaseSolver
from ..base_solver import SolverMetadata

def _circle_matchings(n):
    if n % 2 != 0:
        raise ValueError("Circle method requires an even number of teams.")
    teams = list(range(1, n + 1))
    schedule = {}
    for w in range(1, n):
        week_matches = []
        pivot = teams[-1]
        week_matches.append(tuple(sorted((pivot, teams[w-1]))))
        for i in range(n // 2 - 1):
            team1 = teams[(w + i) % (n - 1)]
            team2 = teams[(w - i - 2 + (n - 1)) % (n - 1)]
            week_matches.append(tuple(sorted((team1, team2))))
        schedule[w] = week_matches
    return schedule

def _at_most_1_pairwise(solver, literals):
    for l1, l2 in combinations(literals, 2):
        solver.add(Or(Not(l1), Not(l2)))

def _at_least_1(s, literals):
    if not literals:
        s.add(BoolVal(False))
        return
    s.add(Or(*literals))

def _exactly_1_pairwise(s, literals):
    _at_most_1_pairwise(s, literals)
    _at_least_1(s, literals)

def _at_most_2_pairwise(solver, literals):
    for l1, l2, l3 in combinations(literals, 3):
        solver.add(Or(Not(l1), Not(l2), Not(l3)))

def _lex_lesseq_sat(solver, list1, list2, prefix=""):
    """Enforces list1 <=lex list2 using boolean logic."""
    n = len(list1)
    eq_prefix = [Bool(f"{prefix}_eq_{i}") for i in range(n)]
    solver.add(eq_prefix[0] == (list1[0] == list2[0]))
    for i in range(1, n):
        solver.add(eq_prefix[i] == And(eq_prefix[i-1], list1[i] == list2[i]))
    solver.add(Implies(list1[0], list2[0]))
    for i in range(1, n):
        solver.add(Implies(eq_prefix[i-1], Implies(list1[i], list2[i])))

def _totalizer_merge(solver, left_sum, right_sum, prefix=""):
    merged_sum = [Bool(f"{prefix}_m_{i}") for i in range(len(left_sum) + len(right_sum))]
    for i in range(len(left_sum) + 1):
        for j in range(len(right_sum) + 1):
            if i > 0: left_literal = left_sum[i-1]
            if j > 0: right_literal = right_sum[j-1]
            if i > 0 and j > 0:
                solver.add(Implies(And(left_literal, right_literal), merged_sum[i + j - 1]))
            if i > 0: solver.add(Implies(left_literal, merged_sum[i - 1]))
            if j > 0: solver.add(Implies(right_literal, merged_sum[j - 1]))
    return merged_sum

def _at_least_k_totalizer(solver, literals, k, prefix=""):
    n = len(literals)
    if k <= 0: return
    if k > n:
        solver.add(BoolVal(False))
        return
    nodes = [[l] for l in literals]  # noqa: E741
    idx = 0
    while len(nodes) > 1:
        next_nodes = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                merged = _totalizer_merge(solver, nodes[i], nodes[i+1], prefix=f"{prefix}_merge{idx}")
                next_nodes.append(merged)
                idx += 1
            else:
                next_nodes.append(nodes[i])
        nodes = next_nodes
    total_sum_vars = nodes[0]
    # Sum >= k means the (k-1)-th variable is True
    solver.add(total_sum_vars[k-1])

def _at_most_k_totalizer(solver, literals, k, prefix=""):
    n = len(literals)
    if k >= n: return
    if k < 0:
        for l in literals: solver.add(Not(l))  # noqa: E741
        return
    nodes = [[l] for l in literals]  # noqa: E741
    idx = 0
    while len(nodes) > 1:
        next_nodes = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                merged = _totalizer_merge(solver, nodes[i], nodes[i+1], prefix=f"{prefix}_merge{idx}")
                next_nodes.append(merged)
                idx += 1
            else:
                next_nodes.append(nodes[i])
        nodes = next_nodes
    total_sum_vars = nodes[0]
    if k < len(total_sum_vars):
        solver.add(Not(total_sum_vars[k]))

def _add_balance_constraints(solver, n, matches, t1_plays_home, teams, max_imbalance):
    num_games = n - 1
    for t in teams:
        home_games_for_t = []
        for m in matches:
            if t == m[0]:
                home_games_for_t.append(t1_plays_home[m])
            elif t == m[1]:
                home_games_for_t.append(Not(t1_plays_home[m]))
        
        lower_bound = math.ceil((num_games - max_imbalance) / 2)
        upper_bound = math.floor((num_games + max_imbalance) / 2)
        
        _at_least_k_totalizer(solver, home_games_for_t, lower_bound, prefix=f"bal_min_t{t}")
        _at_most_k_totalizer(solver, home_games_for_t, upper_bound, prefix=f"bal_max_t{t}")

class SATPairwiseSBSolver(SATBaseSolver):
    def _build_model(self) -> Tuple[Solver, Dict[str, Any]]:
        n = self.n
        
        # pre-solving
        week_matchings = _circle_matchings(n)
        pair_to_week = {match: week for week, matches in week_matchings.items() for match in matches}
        matches = list(pair_to_week.keys())

        teams = range(1, n + 1)
        weeks = range(1, n)
        periods = range(1, n // 2 + 1)

        s = Solver()
        s.set("random_seed", 42)
        s.set("timeout", self.timeout * 1000)

        # decision variables
        match_period = {m: {p: Bool(f"mp_{m[0]}_{m[1]}_{p}") for p in periods} for m in matches}
        t1_plays_home = {m: Bool(f"home_{m[0]}_{m[1]}") for m in matches}

        # structural constraints
        
        # each match in exactly one period
        for m in matches:
            _exactly_1_pairwise(s, list(match_period[m].values()))

        # one game per slot
        for w in weeks:
            matches_in_week = week_matchings[w]
            for p in periods:
                literals = [match_period[m][p] for m in matches_in_week]
                _exactly_1_pairwise(s, literals)

        # max 2 games per period per team
        for t in teams:
            for p in periods:
                matches_for_team_t = [m for m in matches if t in m]
                literals = [match_period[m][p] for m in matches_for_team_t]
                _at_most_2_pairwise(s, literals)

        # symmetry breaking 1: home/away (standard pairwise)
        matches_of_team_1 = sorted([m for m in matches if 1 in m], key=lambda m: pair_to_week[m])
        num_home_games_for_t1 = (n - 1) // 2
        for i in range(num_home_games_for_t1):
            match_key = matches_of_team_1[i]
            if match_key[0] == 1:
                s.add(t1_plays_home[match_key])
            else:
                s.add(Not(t1_plays_home[match_key]))

        # symmetry breaking 2: period lexicographic ordering
        period_list = list(periods)
        for p_idx in range(len(period_list) - 1):
            p1 = period_list[p_idx]
            p2 = period_list[p_idx + 1]
            period1_vector = []
            period2_vector = []
            for m in matches:
                period1_vector.append(match_period[m][p1])
                period2_vector.append(match_period[m][p2])
            _lex_lesseq_sat(s, period1_vector, period2_vector, prefix=f"lex_p{p1}_p{p2}")

        # state dictionary
        state = {
            "match_period": match_period,
            "t1_plays_home": t1_plays_home,
            "pair_to_week": pair_to_week,
            "matches": matches,
            "teams": teams
        }
        return s, state

    def _solve_model(self, model: Tuple[Solver, Dict[str, Any]]) -> STSSolution:
        solver, state = model
        start_time = time.time()
        
        # Check if optimization is requested (assuming passed via config or kwargs in base)
        # If not explicit, default to False or check internal attribute
        optimization = getattr(self, 'optimization', False)
        
        n = self.n
        matches = state["matches"]
        teams = state["teams"]
        t1_plays_home = state["t1_plays_home"]
        best_solution_model = None
        optimal_imbalance = None
        
        if not optimization:
            print("Starting solving process...")

            result = solver.check()
            if result == sat:
                best_solution_model = solver.model()
            else:
                return STSSolution(time=int(time.time() - start_time), optimal=False, obj=None, sol=[])
        else:
            low = 1
            high = n - 1
            while low <= high:
                k_mid = (low + high) // 2
                solver.push()
                _add_balance_constraints(solver, n, matches, t1_plays_home, teams, k_mid)
                
                if solver.check() == sat:
                    optimal_imbalance = k_mid
                    best_solution_model = solver.model()
                    high = k_mid - 1
                else:
                    low = k_mid + 1
                
                solver.pop()
                if time.time() - start_time > self.timeout:
                    break

        elapsed = int(time.time() - start_time)
        print(solver.statistics())

        if best_solution_model:
            weeks = range(1, n)
            periods = range(1, n // 2 + 1)
            match_period = state["match_period"]
            pair_to_week = state["pair_to_week"]
            
            sol = [[None for _ in weeks] for _ in periods]
            for m in matches:
                w_sol = pair_to_week[m]
                p_sol = next(p for p in periods if is_true(best_solution_model.evaluate(match_period[m][p])))
                
                if is_true(best_solution_model.evaluate(t1_plays_home[m])):
                    home, away = m[0], m[1]
                else:
                    home, away = m[1], m[0]
                sol[p_sol - 1][w_sol - 1] = [home, away]
            
            return STSSolution(time=elapsed, optimal=(low > high if optimization else True), obj=optimal_imbalance, sol=sol)
            
        return STSSolution(time=elapsed, optimal=False, obj=None, sol=[])

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="pairwise_sb",
            approach="SAT",
            version="1.0",
            supports_optimization=True,
            description="SAT strategy using Pairwise encoding with symmetry breaking."
        )