import time
from itertools import combinations
from typing import Any, Dict, Tuple
from z3 import Solver, Bool, Not, Or, And, Implies, is_true, sat

from ..utils.solution_format import STSSolution
from .base import SATBaseSolver
from ..base_solver import SolverMetadata
from .commons import (
    circle_matchings, 
    exactly_1_pairwise, 
    at_most_2_pairwise, 
    at_most_1_pairwise,
    sequential_at_least_k, 
    sequential_at_most_k,
    lex_lesseq,
    add_balance_constraints_totalizer
)

class SATPairwiseSBDTSolver(SATBaseSolver):
    def _build_model(self) -> Tuple[Solver, Dict[str, Any]]:
        n = self.n
        
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
        for m in matches:
            exactly_1_pairwise(s, list(match_period[m].values()))

        for w in weeks:
            matches_in_week = week_matchings[w]
            for p in periods:
                literals = [match_period[m][p] for m in matches_in_week]
                exactly_1_pairwise(s, literals)
        
        for t in teams:
            for p in periods:
                matches_for_team_t = [m for m in matches if t in m]
                literals = [match_period[m][p] for m in matches_for_team_t]
                at_most_2_pairwise(s, literals)

        # deficient teams implied constraints
        is_deficient = {t: {p: Bool(f"def_{t}_{p}") for p in periods} for t in teams}

        for t in teams:
            for p in periods:
                matches_for_team_t = [m for m in matches if t in m]
                lits = [match_period[m][p] for m in matches_for_team_t]
                
                at_least_1 = Or(lits)
                at_least_2_clauses = [And(l1, l2) for l1, l2 in combinations(lits, 2)]
                at_least_2 = Or(at_least_2_clauses)
                
                s.add(Implies(is_deficient[t][p], at_least_1))
                for l1, l2 in combinations(lits, 2):
                    s.add(Implies(is_deficient[t][p], Or(Not(l1), Not(l2))))

                count_is_0 = Not(at_least_1)
                count_is_2 = at_least_2
                
                s.add(is_deficient[t][p] == And(Not(count_is_0), Not(count_is_2)))

        for p in periods:
            deficient_in_p = [is_deficient[t][p] for t in teams]
            sequential_at_least_k(s, deficient_in_p, 2, prefix=f"def_chk_p{p}")
            sequential_at_most_k(s, deficient_in_p, 2, prefix=f"def_chk_p{p}")

        for t in teams:
            deficient_periods = [is_deficient[t][p] for p in periods]
            at_most_1_pairwise(s, deficient_periods)

        # symmetry breaking 1: home/away
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
                add_balance_constraints_totalizer(solver, n, matches, t1_plays_home, teams, k)
                
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