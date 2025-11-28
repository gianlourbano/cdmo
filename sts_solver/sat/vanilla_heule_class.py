import time
import math
from typing import Any, Dict, Tuple, List
from z3 import Solver, Bool, Not, Or, And, Implies, sat, BoolVal, is_true

from ..utils.solution_format import STSSolution
from .base import SATBaseSolver
from ..base_solver import SolverMetadata
from .pairwise_class import _circle_matchings, _at_most_2_pairwise

def _totalizer_merge(solver, left_sum, right_sum, prefix=""):
    merged_len = len(left_sum) + len(right_sum)
    merged_sum = [Bool(f"{prefix}_m_{i}") for i in range(merged_len)]
    
    for i in range(len(left_sum) + 1):
        for j in range(len(right_sum) + 1):
            if i > 0 and j > 0:
                target_val = i + j
                if target_val <= merged_len:
                    solver.add(Implies(And(left_sum[i-1], right_sum[j-1]), merged_sum[target_val - 1]))
            elif i > 0 and j == 0:
                if i <= merged_len:
                    solver.add(Implies(left_sum[i-1], merged_sum[i - 1]))
            elif j > 0 and i == 0:
                if j <= merged_len:
                    solver.add(Implies(right_sum[j-1], merged_sum[j - 1]))
    return merged_sum

def _build_totalizer(solver, literals, prefix=""):
    n = len(literals)
    if n == 0: return []
    nodes = [[l] for l in literals]
    idx = 0
    while len(nodes) > 1:
        next_nodes = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                merged = _totalizer_merge(solver, nodes[i], nodes[i+1], prefix=f"{prefix}_{idx}")
                next_nodes.append(merged)
                idx += 1
            else:
                next_nodes.append(nodes[i])
        nodes = next_nodes
    return nodes[0]

def _at_least_k_tot(solver, literals, k, prefix=""):
    n = len(literals)
    if k <= 0: return
    if k > n:
        solver.add(BoolVal(False))
        return
    sum_vars = _build_totalizer(solver, literals, prefix)
    if k > 0 and k-1 < len(sum_vars):
        solver.add(sum_vars[k-1])

def _at_most_k_tot(solver, literals, k, prefix=""):
    n = len(literals)
    if k >= n: return
    if k < 0:
        for l in literals: solver.add(Not(l))
        return
    sum_vars = _build_totalizer(solver, literals, prefix)
    if k < len(sum_vars):
        solver.add(Not(sum_vars[k]))

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
        
        _at_least_k_tot(solver, home_games_for_t, lower_bound, prefix=f"bal_min_t{t}")
        _at_most_k_tot(solver, home_games_for_t, upper_bound, prefix=f"bal_max_t{t}")


class SATVanillaHeuleSolver(SATBaseSolver):
    def __init__(self, n: int, timeout: int = 300, optimization: bool = False):
        super().__init__(n, timeout, optimization)
        self._aux_counter = 0

    def _get_aux_var(self, name: str) -> Bool:
        var = Bool(f"h_{name}_{self._aux_counter}")
        self._aux_counter += 1
        return var

    def _heule_at_most_one(self, solver: Solver, literals: List[Any], name: str = ''):
        n = len(literals)
        if n <= 1: return

        if n <= 4:
            from itertools import combinations
            for l1, l2 in combinations(literals, 2):
                solver.add(Or(Not(l1), Not(l2)))
            return
        
        aux_var = self._get_aux_var(name)
        group1 = literals[:3] + [aux_var]
        self._heule_at_most_one(solver, group1, name=f"{name}_a")
        group2 = [Not(aux_var)] + literals[3:]
        self._heule_at_most_one(solver, group2, name=f"{name}_b")

    def _heule_exactly_one(self, solver: Solver, literals: List[Any], name: str = ''):
        if not literals:
            solver.add(BoolVal(False))
            return
        solver.add(Or(*literals))
        self._heule_at_most_one(solver, literals, name)

    def _build_model(self) -> Tuple[Solver, Dict[str, Any]]:
        self._aux_counter = 0
        n = self.n
        week_matchings = _circle_matchings(n)
        pair_to_week = {match: week for week, matches in week_matchings.items() for match in matches}
        matches = list(pair_to_week.keys())
        
        teams = range(1, n + 1)
        weeks = range(1, n)
        periods = range(1, n // 2 + 1)
        
        s = Solver()
        s.set("timeout", self.timeout * 1000)
        s.set("random_seed", 42)

        # decision variables
        match_period = {m: {p: Bool(f"mp_{m[0]}_{m[1]}_{p}") for p in periods} for m in matches}
        t1_plays_home = {m: Bool(f"home_{m[0]}_{m[1]}") for m in matches}
        
        # structural constraints
        
        # each match in exactly one period
        for m in matches:
            self._heule_exactly_one(s, list(match_period[m].values()), name=f"match_{m}")
            
        # one game per slot
        for w in weeks:
            matches_in_week = week_matchings[w]
            for p in periods:
                lits = [match_period[m][p] for m in matches_in_week]
                self._heule_exactly_one(s, lits, name=f"slot_{w}_{p}")
                
        # max 2 games per period per team
        for t in teams:
            for p in periods:
                matches_for_team = [m for m in matches if t in m]
                lits = [match_period[m][p] for m in matches_for_team]
                _at_most_2_pairwise(s, lits)

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
        
        best_solution_model = None
        optimal_imbalance = None
        
        if not optimization:
            if solver.check() == sat:
                best_solution_model = solver.model()
        else:
            matches = state["matches"]
            teams = state["teams"]
            t1_plays_home = state["t1_plays_home"]
            
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
            sol = [[None for _ in weeks] for _ in periods]
            
            try:
                for m in state["matches"]:
                    w = state["pair_to_week"][m]
                    for p in periods:
                        if is_true(best_solution_model.evaluate(state["match_period"][m][p], model_completion=True)):
                            if is_true(best_solution_model.evaluate(state["t1_plays_home"][m], model_completion=True)):
                                h, a = m[0], m[1]
                            else:
                                h, a = m[1], m[0]
                            if 0 <= p-1 < len(sol) and 0 <= w-1 < len(sol[0]):
                                sol[p-1][w-1] = [h, a]
                            break
            except Exception as e:
                print(f"Error extracting solution: {e}")

            for p_idx in range(len(sol)):
                for w_idx in range(len(sol[0])):
                    if sol[p_idx][w_idx] is None:
                        sol[p_idx][w_idx] = [-1, -1]
            return STSSolution(time=elapsed, optimal=(low > high if optimization else True), obj=optimal_imbalance, sol=sol)
            
        return STSSolution(time=elapsed, optimal=False, obj=None, sol=[])

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="vanilla_heule",
            approach="SAT",
            version="1.0",
            supports_optimization=True,
            description="SAT strategy using Heule encoding.",
        )