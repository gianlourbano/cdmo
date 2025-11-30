import time
from typing import Any, Dict, Tuple, List
from z3 import Solver, Bool, is_true, sat, Or, Not, BoolVal

# Assumo che questi import rimangano validi nel tuo progetto
from ..utils.solution_format import STSSolution
from .base import SATBaseSolver
from ..base_solver import SolverMetadata
from .commons import (
    circle_matchings, 
    totalizer_exactly_k, 
    totalizer_at_most_k, 
    add_balance_constraints_totalizer,
    sequential_at_most_k, 
    add_balance_constraints_sequential,
    exactly_1_pairwise, 
    at_most_2_pairwise
)

class SATVanillaBase(SATBaseSolver):
    def __init__(self, n: int, timeout: int = 300, optimization: bool = False):
        super().__init__(n, timeout, optimization)
        self.start_time = None

    def _add_optimization_constraints(self, solver, n, matches, t1_plays_home, teams, k_mid):
        add_balance_constraints_totalizer(solver, n, matches, t1_plays_home, teams, k_mid)

    def _solve_model(self, model: Tuple[Solver, Dict[str, Any]]) -> STSSolution:
        solver, state = model
        
        start_time_ref = self.start_time if self.start_time else time.time()
        
        optimization = getattr(self, 'optimization', False)
        
        n = self.n
        matches = state["matches"]
        teams = state["teams"]
        t1_plays_home = state["t1_plays_home"]
        
        best_model = None
        optimal_imbalance = None

        def update_solver_timeout():
            elapsed = time.time() - start_time_ref
            remaining = self.timeout - elapsed
            if remaining <= 0:
                solver.set("timeout", 1)
                return False
            solver.set("timeout", int(remaining * 1000))
            return True

        if not optimization:
            if not update_solver_timeout():
                return STSSolution(time=int(time.time() - start_time_ref), optimal=False, obj=None, sol=[])
            if solver.check() == sat:
                best_model = solver.model()
            else:
                elapsed = int(time.time() - start_time_ref)
                return STSSolution(time=elapsed, optimal=False, obj=None, sol=[])
        else:
            print(f"Solving SAT with optimization ({self.get_metadata().name})...")
            low = 1
            high = n - 1
            while low <= high:
                if not update_solver_timeout():
                    print("Timeout reached before iteration.")
                    break
                k_mid = (low + high) // 2
                solver.push()
                
                self._add_optimization_constraints(solver, n, matches, t1_plays_home, teams, k_mid)
                
                if solver.check() == sat:
                    optimal_imbalance = k_mid
                    best_model = solver.model()
                    high = k_mid - 1
                else:
                    low = k_mid + 1
                
                solver.pop()
                if time.time() - start_time_ref > self.timeout:
                    print("Timeout reached.")
                    break

        elapsed = int(time.time() - start_time_ref)
        print(solver.statistics())
        
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
            
        return STSSolution(time=min(elapsed, self.timeout), optimal=False, obj=None, sol=[])


class SATVanillaTotalizerSolver(SATVanillaBase):
    def _build_model(self) -> Tuple[Solver, Dict[str, Any]]:
        self.start_time = time.time()
        n = self.n
        week_matchings = circle_matchings(n)
        pair_to_week = {match: week for week, matches in week_matchings.items() for match in matches}
        matches = list(pair_to_week.keys())
        teams = range(1, n + 1)
        weeks = range(1, n)
        periods = range(1, n // 2 + 1)
        
        s = Solver()
        s.set("random_seed", 42)

        # decision variables
        match_period = {m: {p: Bool(f"mp_{m[0]}_{m[1]}_{p}") for p in periods} for m in matches}
        t1_plays_home = {m: Bool(f"home_{m[0]}_{m[1]}") for m in matches}

        # structural constraints
        for m in matches:
            totalizer_exactly_k(s, list(match_period[m].values()), 1, prefix=f"match_{m}")

        for w in weeks:
            matches_in_week = week_matchings[w]
            for p in periods:
                slot_lits = [match_period[m][p] for m in matches_in_week]
                totalizer_exactly_k(s, slot_lits, 1, prefix=f"slot_{w}_{p}")
        
        for t in teams:
            for p in periods:
                matches_for_team = [m for m in matches if t in m]
                lits = [match_period[m][p] for m in matches_for_team]
                totalizer_at_most_k(s, lits, 2, prefix=f"team_{t}_{p}")

        
        state = {
            "match_period": match_period, 
            "t1_plays_home": t1_plays_home, 
            "pair_to_week": pair_to_week, 
            "matches": matches, 
            "teams": teams
        }
        return s, state

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="vanilla_totalizer",
            approach="SAT",
            version="1.0",
            supports_optimization=True,
            description="SAT strategy using Totalizer encoding.",
        )


class SATVanillaSequentialSolver(SATVanillaBase):
    
    # TODO capire se necessario
    def _add_optimization_constraints(self, solver, n, matches, t1_plays_home, teams, k_mid):
        add_balance_constraints_sequential(solver, n, matches, t1_plays_home, teams, k_mid)

    def _build_model(self) -> Tuple[Solver, Dict[str, Any]]:
        self.start_time = time.time()
        n = self.n
        week_matchings = circle_matchings(n)
        pair_to_week = {match: week for week, matches in week_matchings.items() for match in matches}
        matches = list(pair_to_week.keys())
        
        s = Solver()
        s.set("random_seed", 42)

        # decision variables
        match_period = {m: {p: Bool(f"mp_{m[0]}_{m[1]}_{p}") for p in range(1, n // 2 + 1)} for m in matches}
        t1_plays_home = {m: Bool(f"home_{m[0]}_{m[1]}") for m in matches}
        
        periods = range(1, n // 2 + 1)
        weeks = range(1, n)
        teams = range(1, n + 1)

        # structural constraints
        for m in matches:
            lits = list(match_period[m].values())
            sequential_at_most_k(s, lits, 1, prefix=f"match_{m}")
            s.add(Or(*lits)) # At least 1

        for w in weeks:
            for p in periods:
                lits = [match_period[m][p] for m in week_matchings[w]]
                sequential_at_most_k(s, lits, 1, prefix=f"slot_{w}_{p}")
                s.add(Or(*lits))

        for t in teams:
            for p in periods:
                matches_for_team = [m for m in matches if t in m]
                lits = [match_period[m][p] for m in matches_for_team]
                sequential_at_most_k(s, lits, 2, prefix=f"team_{t}_{p}")


        state = {
            "match_period": match_period, 
            "t1_plays_home": t1_plays_home, 
            "pair_to_week": pair_to_week, 
            "matches": matches,
            "teams": teams
        }
        return s, state

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="vanilla_sequential",
            approach="SAT",
            version="1.0",
            supports_optimization=True,
            description="SAT strategy using Sequential encoding.",
        )


class SATVanillaPairwiseSolver(SATVanillaBase):
    def _build_model(self) -> Tuple[Solver, Dict[str, Any]]:
        self.start_time = time.time()
        n = self.n
        week_matchings = circle_matchings(n)
        pair_to_week = {match: week for week, matches in week_matchings.items() for match in matches}
        matches = list(pair_to_week.keys())

        teams = range(1, n + 1)
        weeks = range(1, n)
        periods = range(1, n // 2 + 1)

        s = Solver()
        s.set("random_seed", 42)

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


        state = {
            "match_period": match_period,
            "t1_plays_home": t1_plays_home,
            "pair_to_week": pair_to_week,
            "matches": matches,
            "teams": teams
        }
        return s, state

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="vanilla_pairwise",
            approach="SAT",
            version="1.0",
            supports_optimization=True,
            description="SAT strategy using Pairwise encoding.",
        )


class SATVanillaHeuleSolver(SATVanillaBase):
    def __init__(self, n: int, timeout: int = 300, optimization: bool = False):
        super().__init__(n, timeout, optimization)
        self._aux_counter = 0

    def _get_aux_var(self, name: str) -> Bool:
        var = Bool(f"h_{name}_{self._aux_counter}")
        self._aux_counter += 1
        return var

    def _heule_at_most_one(self, solver: Solver, literals: List[Any], name: str = ''):
        n_lits = len(literals)
        if n_lits <= 1: return

        if n_lits <= 4:
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
        self.start_time = time.time()
        self._aux_counter = 0
        n = self.n
        week_matchings = circle_matchings(n)
        pair_to_week = {match: week for week, matches in week_matchings.items() for match in matches}
        matches = list(pair_to_week.keys())
        
        teams = range(1, n + 1)
        weeks = range(1, n)
        periods = range(1, n // 2 + 1)
        
        s = Solver()
        s.set("random_seed", 42)

        # decision variables
        match_period = {m: {p: Bool(f"mp_{m[0]}_{m[1]}_{p}") for p in periods} for m in matches}
        t1_plays_home = {m: Bool(f"home_{m[0]}_{m[1]}") for m in matches}
        
        # structural constraints
        for m in matches:
            self._heule_exactly_one(s, list(match_period[m].values()), name=f"match_{m}")
            
        for w in weeks:
            matches_in_week = week_matchings[w]
            for p in periods:
                lits = [match_period[m][p] for m in matches_in_week]
                self._heule_exactly_one(s, lits, name=f"slot_{w}_{p}")
                
        for t in teams:
            for p in periods:
                matches_for_team = [m for m in matches if t in m]
                lits = [match_period[m][p] for m in matches_for_team]
                at_most_2_pairwise(s, lits)


        state = {
            "match_period": match_period,
            "t1_plays_home": t1_plays_home,
            "pair_to_week": pair_to_week,
            "matches": matches,
            "teams": teams
        }
        return s, state

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="vanilla_heule",
            approach="SAT",
            version="1.0",
            supports_optimization=True,
            description="SAT strategy using Heule encoding.",
        )