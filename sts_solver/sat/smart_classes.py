import time
from itertools import combinations
from typing import Any, Dict, Tuple, List
from z3 import Solver, Bool, is_true, sat, Or, Not, And, Implies

from ..utils.solution_format import STSSolution
from ..base_solver import SolverMetadata

# Importiamo la classe base direttamente da vanilla_classes per riutilizzare
# la logica di _solve_model e del ciclo di ottimizzazione
from .vanilla_classes import SATVanillaBase

from .commons import (
    circle_matchings, 
    smart_at_most_k,
    smart_exactly_k,
    sequential_at_most_k,
    sequential_at_least_k,
    lex_lesseq,
    at_most_1_pairwise
)

class SATSmartSolver(SATVanillaBase):
    def _build_model(self) -> Tuple[Solver, Dict[str, Any]]:
        # Impostiamo il tempo di inizio come richiesto da SATVanillaBase
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
        
        # each match in exactly one period
        for m in matches:
            smart_exactly_k(s, list(match_period[m].values()), 1, prefix=f"match_{m}")

        # one game per slot
        for w in weeks:
            matches_in_week = week_matchings[w]
            for p in periods:
                literals = [match_period[m][p] for m in matches_in_week]
                smart_exactly_k(s, literals, 1, prefix=f"week_{w}_period_{p}")

        # max 2 games per period per team
        for t in teams:
            for p in periods:
                matches_for_team_t = [m for m in matches if t in m]
                literals = [match_period[m][p] for m in matches_for_team_t]
                smart_at_most_k(s, literals, 2, prefix=f"team_{t}_{p}")


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
            name="smart",
            approach="SAT",
            version="1.0",
            supports_optimization=True,
            description="SAT strategy using Pairwise encoding.",
        )


class SATSmartSBSolver(SATVanillaBase):
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
        
        # each match in exactly one period
        for m in matches:
            smart_exactly_k(s, list(match_period[m].values()), 1, prefix=f"match_{m}")

        # one game per slot
        for w in weeks:
            matches_in_week = week_matchings[w]
            for p in periods:
                literals = [match_period[m][p] for m in matches_in_week]
                smart_exactly_k(s, literals, 1, prefix=f"week_{w}_period_{p}")

        # max 2 games per period per team
        for t in teams:
            for p in periods:
                matches_for_team_t = [m for m in matches if t in m]
                literals = [match_period[m][p] for m in matches_for_team_t]
                smart_at_most_k(s, literals, 2, prefix=f"team_{t}_{p}")

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
            name="smart_sb",
            approach="SAT",
            version="1.0",
            supports_optimization=True,
            description="SAT strategy using Pairwise encoding with symmetry breaking."
        )


class SATSmartDTSolver(SATVanillaBase):
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
        
        # each match in exactly one period
        for m in matches:
            smart_exactly_k(s, list(match_period[m].values()), 1, prefix=f"match_{m}")
        # one game per slot
        for w in weeks:
            matches_in_week = week_matchings[w]
            for p in periods:
                literals = [match_period[m][p] for m in matches_in_week]
                smart_exactly_k(s, literals, 1, prefix=f"week_{w}_period_{p}")
        # max 2 games per period per team
        for t in teams:
            for p in periods:
                matches_for_team_t = [m for m in matches if t in m]
                literals = [match_period[m][p] for m in matches_for_team_t]
                smart_at_most_k(s, literals, 2, prefix=f"team_{t}_{p}")
                
        # deficient teams implied constraints
        
        # auxiliary variables: is_deficient[t][p]
        is_deficient = {t: {p: Bool(f"def_{t}_{p}") for p in periods} for t in teams}

        for t in teams:
            for p in periods:
                matches_for_team_t = [m for m in matches if t in m]
                lits = [match_period[m][p] for m in matches_for_team_t]
                
                # logic: deficient <==> sum(lits) == 1
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
            sequential_at_least_k(s, deficient_in_p, 2, prefix=f"def_chk_p{p}")
            sequential_at_most_k(s, deficient_in_p, 2, prefix=f"def_chk_p{p}")
        
        # implied constraint 2: every team is deficient in at most 1 period
        for t in teams:
            deficient_periods = [is_deficient[t][p] for p in periods]
            at_most_1_pairwise(s, deficient_periods)


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
            name="smart_dt",
            approach="SAT",
            version="2.0",
            supports_optimization=True,
            description="SAT strategy using Pairwise encoding with deficient teams."
        )


class SATSmartSBDTSolver(SATVanillaBase):
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
            smart_exactly_k(s, list(match_period[m].values()), 1, prefix=f"match_{m}")

        for w in weeks:
            matches_in_week = week_matchings[w]
            for p in periods:
                literals = [match_period[m][p] for m in matches_in_week]
                smart_exactly_k(s, literals, 1, prefix=f"week_{w}_period_{p}")
        
        for t in teams:
            for p in periods:
                matches_for_team_t = [m for m in matches if t in m]
                literals = [match_period[m][p] for m in matches_for_team_t]
                smart_at_most_k(s, literals, 2, prefix=f"team_{t}_{p}")

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

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="smart_sb_dt",
            approach="SAT",
            version="1.0",
            supports_optimization=True,
            description="SAT strategy using Pairwise encoding with symmetry breaking and deficient teams."
        )