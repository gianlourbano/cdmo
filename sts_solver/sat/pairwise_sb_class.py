import time
from typing import Any, Dict, Tuple
from z3 import Solver, Bool, Not, is_true, sat

from ..utils.solution_format import STSSolution
from .base import SATBaseSolver
from ..base_solver import SolverMetadata
from .commons import (
    circle_matchings, 
    exactly_1_pairwise, 
    at_most_2_pairwise, 
    lex_lesseq, 
    add_balance_constraints_totalizer
)

class SATPairwiseSBSolver(SATBaseSolver):
    def _build_model(self) -> Tuple[Solver, Dict[str, Any]]:
        n = self.n
        
        # pre-solving
        week_matchings = circle_matchings(n)
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
            exactly_1_pairwise(s, list(match_period[m].values()))

        # one game per slot
        for w in weeks:
            matches_in_week = week_matchings[w]
            for p in periods:
                literals = [match_period[m][p] for m in matches_in_week]
                exactly_1_pairwise(s, literals)

        # max 2 games per period per team
        for t in teams:
            for p in periods:
                matches_for_team_t = [m for m in matches if t in m]
                literals = [match_period[m][p] for m in matches_for_team_t]
                at_most_2_pairwise(s, literals)

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
            lex_lesseq(s, period1_vector, period2_vector, prefix=f"lex_p{p1}_p{p2}")

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
                # Uses Totalizer for balance constraints as in original file
                add_balance_constraints_totalizer(solver, n, matches, t1_plays_home, teams, k_mid)
                
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