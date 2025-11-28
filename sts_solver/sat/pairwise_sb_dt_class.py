import time
import math
from itertools import combinations
from typing import Any, Dict, Tuple
from z3 import Solver, Bool, Not, Or, And, Implies, is_true, sat, BoolVal

from ..utils.solution_format import STSSolution
from .base import SATBaseSolver
from ..base_solver import SolverMetadata

def _sequential_at_most_k(solver, literals, k, prefix=""):
    n = len(literals)
    if k >= n: return
    if k < 0:
        for l in literals: solver.add(Not(l))
        return

    s = [[Bool(f"{prefix}_s_{i}_{j}") for j in range(k + 1)] for i in range(n)]
    
    # Base case: i=0
    solver.add(s[0][1] == literals[0])
    for j in range(2, k + 1):
        solver.add(Not(s[0][j]))
        
    # Induction
    for i in range(1, n):
        solver.add(s[i][1] == Or(s[i-1][1], literals[i]))
        
        for j in range(2, k + 1):
            solver.add(s[i][j] == Or(s[i-1][j], And(s[i-1][j-1], literals[i])))
        solver.add(Not(And(s[i-1][k], literals[i])))

def _sequential_at_least_k(solver, literals, k, prefix=""):
    n = len(literals)
    if k <= 0: return
    if n < k:
        solver.add(BoolVal(False))
        return
        
    s = [[Bool(f"{prefix}_al_{i}_{j}") for j in range(k + 1)] for i in range(n)]
    
    # Base case: i=0
    solver.add(s[0][1] == literals[0])
    for j in range(2, k + 1):
        solver.add(Not(s[0][j]))
        
    # Induction
    for i in range(1, n):
        solver.add(s[i][1] == Or(s[i-1][1], literals[i]))
        for j in range(2, k + 1):
            solver.add(s[i][j] == Or(s[i-1][j], And(s[i-1][j-1], literals[i])))

    solver.add(s[n-1][k])

def _circle_matchings(n):
    if n % 2 != 0:
        raise ValueError("An even number of teams is required.")
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
    n = len(list1)

    eq_prefix = [Bool(f"{prefix}_eq_{i}") for i in range(n)]

    # Base case
    solver.add(eq_prefix[0] == (list1[0] == list2[0]))
    
    # Recursive step
    for i in range(1, n):
        solver.add(eq_prefix[i] == And(eq_prefix[i-1], list1[i] == list2[i]))

    solver.add(Implies(list1[0], list2[0])) # i=0
    for i in range(1, n):
        solver.add(Implies(eq_prefix[i-1], Implies(list1[i], list2[i])))

def _totalizer_merge(solver, left_sum, right_sum, prefix=""):
    merged_sum = [Bool(f"{prefix}_m_{i}") for i in range(len(left_sum) + len(right_sum))]
    for i in range(len(left_sum) + 1):
        for j in range(len(right_sum) + 1):
            if i > 0:
                left_lit = left_sum[i-1]
            if j > 0:
                right_lit = right_sum[j-1]
            if i > 0 and j > 0:
                solver.add(Implies(And(left_lit, right_lit), merged_sum[i + j - 1]))
            if i > 0: 
                solver.add(Implies(left_lit, merged_sum[i - 1]))
            if j > 0: 
                solver.add(Implies(right_lit, merged_sum[j - 1]))
    return merged_sum

def _build_totalizer_sum(solver, literals, prefix=""):
    n = len(literals)
    if n == 0: 
        return []
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
    return nodes[0]

def _exactly_k_totalizer(solver, literals, k, prefix=""):
    n = len(literals)
    if k < 0 or k > n:
        solver.add(BoolVal(False))
        return
    if n == 0 and k == 0: 
        return
    
    total_sum_vars = _build_totalizer_sum(solver, literals, prefix)
    if k > 0: 
        solver.add(total_sum_vars[k-1])
    if k < n: 
        solver.add(Not(total_sum_vars[k]))

def _at_least_k_totalizer(solver, literals, k, prefix=""):
    total_sum_vars = _build_totalizer_sum(solver, literals, prefix)
    if k > 0 and (k-1) < len(total_sum_vars):
        solver.add(total_sum_vars[k-1])

def _at_most_k_totalizer(solver, literals, k, prefix=""):
    total_sum_vars = _build_totalizer_sum(solver, literals, prefix)
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


class SATPairwiseSBDTSolver(SATBaseSolver):
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
        
        # s.set("restart.max", 15000)
        # s.set("dack.gc_inv_decay", 0.99)

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

        # deficient teams implied constraints
        
        # auxiliary variables: is_deficient[t][p]
        is_deficient = {t: {p: Bool(f"def_{t}_{p}") for p in periods} for t in teams}

        for t in teams:
            for p in periods:
                matches_for_team_t = [m for m in matches if t in m]
                lits = [match_period[m][p] for m in matches_for_team_t]
                
                # logic: deficient <==> sum(lits) == 1
                # since we already have sum <= 2 global constraint:
                # sum == 1 is (at_least_1 AND NOT at_least_2)
                
                at_least_1 = Or(lits)
                
                at_least_2_clauses = [And(l1, l2) for l1, l2 in combinations(lits, 2)]
                at_least_2 = Or(at_least_2_clauses)
                
                s.add(Implies(is_deficient[t][p], at_least_1))

                for l1, l2 in combinations(lits, 2):
                    s.add(Implies(is_deficient[t][p], Or(Not(l1), Not(l2))))

                count_is_0 = Not(at_least_1)
                count_is_2 = at_least_2
                
                s.add(is_deficient[t][p] == And(Not(count_is_0), Not(count_is_2)))

        # implied constraint 1: in every period, exactly 2 teams are deficient
        for p in periods:
            deficient_in_p = [is_deficient[t][p] for t in teams]
            _sequential_at_least_k(s, deficient_in_p, 2, prefix=f"def_chk_p{p}")
            _sequential_at_most_k(s, deficient_in_p, 2, prefix=f"def_chk_p{p}")
            # _exactly_k_totalizer(s, deficient_in_p, 2, prefix=f"def_chk_p{p}")

        # implied constraint 2: every team is deficient in at most 1 period
        for t in teams:
            deficient_periods = [is_deficient[t][p] for p in periods]
            _at_most_1_pairwise(s, deficient_periods)

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
            
            # build vectors representing the schedule for p1 and p2 using a deterministic order of matches
            for m in matches:
                period1_vector.append(match_period[m][p1])
                period2_vector.append(match_period[m][p2])
            
            # enforce: schedule(p1) <=lex schedule(p2)
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
        
        optimization = getattr(self, 'optimization', False)
        
        n = self.n
        matches = state["matches"]
        teams = state["teams"]
        t1_plays_home = state["t1_plays_home"]
        match_period = state["match_period"]
        pair_to_week = state["pair_to_week"]
        
        best_solution_model = None
        optimal_imbalance = None
        
        if not optimization:
            if solver.check() == sat:
                best_solution_model = solver.model()
            else:
                return STSSolution(time=int(time.time() - start_time), optimal=False, obj=None, sol=[])
        else:
            low = 1
            high = n - 1
            while low <= high:
                k = (low + high) // 2
                solver.push()
                _add_balance_constraints(solver, n, matches, t1_plays_home, teams, k)
                
                if solver.check() == sat:
                    best_solution_model = solver.model()
                    print(solver.statistics())
                    optimal_imbalance = k
                    high = k - 1
                else:
                    low = k + 1
                
                solver.pop()
                if time.time() - start_time > self.timeout:
                    break
        
        elapsed = int(time.time() - start_time)
        print(solver.statistics())
        
        if best_solution_model:
            weeks = range(1, n)
            periods = range(1, n // 2 + 1)
            
            sol = [[None for _ in weeks] for _ in periods]
            for m in matches:
                w_sol = pair_to_week[m]
                p_sol = next(p for p in periods if is_true(best_solution_model.evaluate(match_period[m][p])))
                
                if is_true(best_solution_model.evaluate(t1_plays_home[m])):
                    h, a = m[0], m[1]
                else:
                    h, a = m[1], m[0]
                sol[p_sol - 1][w_sol - 1] = [h, a]
            
            return STSSolution(time=elapsed, optimal=(low > high if optimization else True), obj=optimal_imbalance, sol=sol)
            
        return STSSolution(time=elapsed, optimal=False, obj=None, sol=[])

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="pairwise_sb_dt",
            approach="SAT",
            version="1.0",
            supports_optimization=True,
            description="SAT strategy using Pairwise encoding with symmetry breaking and deficient teams."
        )