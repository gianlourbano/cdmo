import time
import math
from typing import Any, Dict, Tuple
from z3 import Solver, Bool, Not, Or, And, is_true, sat, BoolVal

from ..utils.solution_format import STSSolution
from .base import SATBaseSolver
from ..base_solver import SolverMetadata
from .pairwise_class import _circle_matchings

def _sequential_exactly_2(solver, literals, prefix=""):
    """
    Optimized Sequential Counter specifically for Sum(literals) == 2.
    Uses fewer auxiliary variables than the generic k version.
    """
    n = len(literals)
    if n < 2:
        solver.add(BoolVal(False))
        return

    s1 = [Bool(f"{prefix}_sq2_ge1_{i}") for i in range(n)]
    s2 = [Bool(f"{prefix}_sq2_ge2_{i}") for i in range(n)]

    # base case i=0
    solver.add(s1[0] == literals[0])
    solver.add(Not(s2[0])) # cannot be >= 2 with 1 item

    for i in range(1, n):
        # true if previously >=1 OR current is true
        solver.add(s1[i] == Or(s1[i-1], literals[i]))
        
        # true if previously >=2 OR (prev >=1 AND current is true)
        solver.add(s2[i] == Or(s2[i-1], And(s1[i-1], literals[i])))
        
        # prevents the sum from reaching 3.
        solver.add(Not(And(s2[i-1], literals[i])))

    solver.add(s2[n-1])

def _sequential_at_most_k(solver, literals, k, prefix=""):
    """Encodes Sum(literals) <= k using Sequential Counter."""
    n = len(literals)
    if k >= n: return
    if k < 0:
        for l in literals: solver.add(Not(l))
        return

    s = [[Bool(f"{prefix}_s_{i}_{j}") for j in range(k + 1)] for i in range(n)]
    
    # base case: i=0
    solver.add(s[0][1] == literals[0])
    for j in range(2, k + 1):
        solver.add(Not(s[0][j]))
        
    # induction
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
    
    # base case: i=0
    solver.add(s[0][1] == literals[0])
    for j in range(2, k + 1):
        solver.add(Not(s[0][j]))
        
    # induction
    for i in range(1, n):
        solver.add(s[i][1] == Or(s[i-1][1], literals[i]))
        for j in range(2, k + 1):
            solver.add(s[i][j] == Or(s[i-1][j], And(s[i-1][j-1], literals[i])))

    solver.add(s[n-1][k])

def _add_balance_constraints(solver, n, matches, t1_plays_home, teams, max_imbalance):
    """Adds home-away balance constraints using Sequential Counter."""
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
        
        # Use LOCAL Sequential functions
        _sequential_at_least_k(solver, home_games_for_t, lower_bound, prefix=f"bal_min_t{t}")
        _sequential_at_most_k(solver, home_games_for_t, upper_bound, prefix=f"bal_max_t{t}")


class SATVanillaSequentialSolver(SATBaseSolver):
    def _build_model(self) -> Tuple[Solver, Dict[str, Any]]:
        n = self.n
        week_matchings = _circle_matchings(n)
        pair_to_week = {match: week for week, matches in week_matchings.items() for match in matches}
        matches = list(pair_to_week.keys())
        
        s = Solver()
        s.set("timeout", self.timeout * 1000)
        s.set("random_seed", 42)

        # decision variables
        match_period = {m: {p: Bool(f"mp_{m[0]}_{m[1]}_{p}") for p in range(1, n // 2 + 1)} for m in matches}
        t1_plays_home = {m: Bool(f"home_{m[0]}_{m[1]}") for m in matches}
        
        periods = range(1, n // 2 + 1)
        weeks = range(1, n)
        teams = range(1, n + 1)

        # structural constraints
        
        # each match in exactly one period
        for m in matches:
            lits = list(match_period[m].values())
            _sequential_at_most_k(s, lits, 1, prefix=f"match_{m}")
            s.add(Or(*lits)) # At least 1

        # one game per slot
        for w in weeks:
            for p in periods:
                lits = [match_period[m][p] for m in week_matchings[w]]
                _sequential_at_most_k(s, lits, 1, prefix=f"slot_{w}_{p}")
                s.add(Or(*lits))

        # max 2 games per period per team
        for t in teams:
            for p in periods:
                matches_for_team = [m for m in matches if t in m]
                lits = [match_period[m][p] for m in matches_for_team]
                _sequential_at_most_k(s, lits, 2, prefix=f"team_{t}_{p}")

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
        
        best_solution_model = None
        optimal_imbalance = None
        
        if not optimization:
            if solver.check() == sat:
                best_solution_model = solver.model()
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
        
        if best_solution_model:
            weeks = range(1, n)
            periods = range(1, n // 2 + 1)
            match_period = state["match_period"]
            
            sol = [[None for _ in weeks] for _ in periods]
            
            try:
                for m in matches:
                    w = state["pair_to_week"][m]
                    for p in periods:
                        if is_true(best_solution_model.evaluate(match_period[m][p], model_completion=True)):
                            if is_true(best_solution_model.evaluate(t1_plays_home[m], model_completion=True)):
                                h, a = m[0], m[1]
                            else:
                                h, a = m[1], m[0]
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
            name="vanilla_sequential",
            approach="SAT",
            version="1.0",
            supports_optimization=True,
            description="SAT strategy using Sequential encoding.",
        )