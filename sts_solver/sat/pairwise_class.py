"""Class-based SAT solver using pairwise encodings for exactly-one constraints.

Hybrid strategy: pairwise for exactly-1, sequential for at-most-2 in periods.
"""

import time
from itertools import combinations
from typing import Any, Dict, List, Tuple
from z3 import Solver, Bool, Not, Or, And, BoolVal, is_true, sat

from ..utils.solution_format import STSSolution
from .base import SATBaseSolver
from ..base_solver import SolverMetadata


def _at_most_1_pairwise(s, literals):
    for l1, l2 in combinations(literals, 2):
        s.add(Or(Not(l1), Not(l2)))


def _at_least_1(s, literals):
    if not literals:
        s.add(BoolVal(False))
        return
    s.add(Or(*literals))


def _exactly_1_pairwise(s, literals):
    _at_most_1_pairwise(s, literals)
    _at_least_1(s, literals)


def _at_most_k_sequential(s, literals, k, prefix=""):
    n = len(literals)
    if k < 0:
        for l in literals:
            s.add(Not(l))
        return
    if n <= k:
        return
    seq = [[Bool(f"{prefix}_s_{i}_{j}") for j in range(k + 1)] for i in range(n)]
    s.add(seq[0][0] == literals[0])
    for j in range(1, k + 1):
        s.add(Not(seq[0][j]))
    for i in range(1, n):
        s.add(seq[i][0] == Or(seq[i - 1][0], literals[i]))
        for j in range(1, k + 1):
            s.add(seq[i][j] == Or(seq[i - 1][j], And(seq[i - 1][j - 1], literals[i])))
    s.add(Not(seq[n - 1][k]))


class SATPairwiseNativeSolver(SATBaseSolver):
    def _build_model(self) -> Tuple[Solver, Dict[str, Any]]:
        n = self.n
        weeks = range(1, n)
        periods = range(1, n // 2 + 1)
        teams = range(1, n + 1)

        s = Solver()
        s.set("timeout", self.timeout * 1000)

        plays_home: Dict[int, Dict[int, Dict[int, Any]]] = {
            t: {w: {p: Bool(f"home_{t}_{w}_{p}") for p in periods} for w in weeks} for t in teams
        }
        plays_away: Dict[int, Dict[int, Dict[int, Any]]] = {
            t: {w: {p: Bool(f"away_{t}_{w}_{p}") for p in periods} for w in weeks} for t in teams
        }

        # Slot constraints with pairwise exactly-one
        for w in weeks:
            for p in periods:
                home_teams = [plays_home[t][w][p] for t in teams]
                away_teams = [plays_away[t][w][p] for t in teams]
                _exactly_1_pairwise(s, home_teams)
                _exactly_1_pairwise(s, away_teams)
                for t in teams:
                    s.add(Not(And(plays_home[t][w][p], plays_away[t][w][p])))

        # Weekly appearances
        for t in teams:
            for w in weeks:
                weekly_app = [plays_home[t][w][p] for p in periods] + [plays_away[t][w][p] for p in periods]
                _exactly_1_pairwise(s, weekly_app)

        # Unique meetings
        for t1, t2 in combinations(teams, 2):
            meetings = []
            for w in weeks:
                for p in periods:
                    meet = Bool(f"meet_{t1}_{t2}_{w}_{p}")
                    s.add(meet == Or(And(plays_home[t1][w][p], plays_away[t2][w][p]),
                                     And(plays_home[t2][w][p], plays_away[t1][w][p])))
                    meetings.append(meet)
            _exactly_1_pairwise(s, meetings)

        # At most twice in the same period: sequential counter
        for t in teams:
            for p in periods:
                period_app = [plays_home[t][w][p] for w in weeks] + [plays_away[t][w][p] for w in weeks]
                _at_most_k_sequential(s, period_app, 2, prefix=f"period_t{t}_p{p}")

        # Symmetry breaking: fix first week
        for p in periods:
            home_team = 2 * p - 1
            away_team = 2 * p
            s.add(plays_home[home_team][1][p])
            s.add(plays_away[away_team][1][p])

        return s, {"plays_home": plays_home, "plays_away": plays_away}

    def _solve_model(self, model: Tuple[Solver, Dict[str, Any]]) -> STSSolution:
        solver, state = model
        start = time.time()
        result = solver.check()
        elapsed = int(time.time() - start)
        if result == sat:
            m = solver.model()
            n = self.n
            weeks = range(1, n)
            periods = range(1, n // 2 + 1)
            teams = range(1, n + 1)
            plays_home = state["plays_home"]
            plays_away = state["plays_away"]

            sol: List[List[List[int]]] = []
            for p in periods:
                period_games: List[List[int]] = []
                for w in weeks:
                    home_team, away_team = -1, -1
                    for t in teams:
                        if is_true(m.evaluate(plays_home[t][w][p], model_completion=True)):
                            home_team = t
                        if is_true(m.evaluate(plays_away[t][w][p], model_completion=True)):
                            away_team = t
                    period_games.append([home_team, away_team])
                sol.append(period_games)

            return STSSolution(time=elapsed, optimal=True, obj=None, sol=sol)
        return STSSolution(time=min(elapsed, self.timeout), optimal=False, obj=None, sol=[])

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="pairwise",
            approach="SAT",
            version="1.0",
            supports_optimization=False,
            description="Pairwise SAT encoding for exactly-one; sequential at-most-two",
        )
