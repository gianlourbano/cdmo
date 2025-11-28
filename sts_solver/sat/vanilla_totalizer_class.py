import time
from typing import Any, Dict, Tuple
from z3 import Solver, Bool, is_true, sat

from ..utils.solution_format import STSSolution
from .base import SATBaseSolver
from ..base_solver import SolverMetadata
from .commons import (
    circle_matchings, 
    totalizer_exactly_k, 
    totalizer_at_most_k, 
    add_balance_constraints_totalizer
)

class SATVanillaTotalizerSolver(SATBaseSolver):
    def _build_model(self) -> Tuple[Solver, Dict[str, Any]]:
        n = self.n
        week_matchings = circle_matchings(n)
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
            totalizer_exactly_k(s, list(match_period[m].values()), 1, prefix=f"match_{m}")

        # one game per slot
        for w in weeks:
            matches_in_week = week_matchings[w]
            for p in periods:
                slot_lits = [match_period[m][p] for m in matches_in_week]
                totalizer_exactly_k(s, slot_lits, 1, prefix=f"slot_{w}_{p}")
        
        # max 2 games per period per team
        for t in teams:
            for p in periods:
                matches_for_team = [m for m in matches if t in m]
                lits = [match_period[m][p] for m in matches_for_team]
                totalizer_at_most_k(s, lits, 2, prefix=f"team_{t}_{p}")

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
        
        best_model = None
        optimal_imbalance = None
        
        if not optimization:
            if solver.check() == sat:
                best_model = solver.model()
            else:
                return STSSolution(time=int(time.time() - start_time), optimal=False, obj=None, sol=[])
        else:
            print("Solving SAT with optimization (Totalizer)...")
            low = 1
            high = n - 1
            while low <= high:
                k_mid = (low + high) // 2
                solver.push()
                add_balance_constraints_totalizer(solver, n, matches, t1_plays_home, teams, k_mid)
                
                if solver.check() == sat:
                    optimal_imbalance = k_mid
                    best_model = solver.model()
                    high = k_mid - 1
                else:
                    low = k_mid + 1
                
                solver.pop()
                if time.time() - start_time > self.timeout:
                    print("Timeout reached.")
                    break

        elapsed = int(time.time() - start_time)
        
        if best_model:
            weeks = range(1, n)
            periods = range(1, n // 2 + 1)
            sol = [[None for _ in weeks] for _ in periods]
            
            try:
                for m in state["matches"]:
                    w = state["pair_to_week"][m]
                    for p in periods:
                        if is_true(best_model.evaluate(state["match_period"][m][p], model_completion=True)):
                            if is_true(best_model.evaluate(state["t1_plays_home"][m], model_completion=True)):
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
            name="vanilla_totalizer",
            approach="SAT",
            version="1.0",
            supports_optimization=True,
            description="SAT strategy using Totalizer encoding.",
        )