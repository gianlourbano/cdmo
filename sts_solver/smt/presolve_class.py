"""Class-based SMT presolve solver (Z3) using period labeling after prescheduling."""

import time
from typing import Any, Dict, List, Tuple
from z3 import Solver, Then, Int, Bool, PbEq, PbLe

from ..utils.solution_format import STSSolution
from .base import SMTBaseSolver
from ..base_solver import SolverMetadata


def schedule_iterative_divide_and_conquer(n: int) -> Dict[int, List[Tuple[int, int]]]:
    if n == 0:
        return {}
    teams = list(range(1, n + 1))
    schedule: Dict[int, List[Tuple[int, int]]] = {}
    for week in range(1, n):
        schedule[week] = []
        opponent_for_n = teams[(week - 1) % (n - 1)]
        pair = (min(n, opponent_for_n), max(n, opponent_for_n))
        schedule[week].append(pair)
        for i in range(1, n // 2):
            team1_idx = (week - 1 + i) % (n - 1)
            team2_idx = (week - 1 - i + (n - 1)) % (n - 1)
            team1 = teams[team1_idx]
            team2 = teams[team2_idx]
            pair2 = (min(team1, team2), max(team1, team2))
            schedule[week].append(pair2)
    return schedule


def home_away_balance(matches_per_week: Dict[int, List[Tuple[int, int]]], n: int):
    balanced: Dict[int, List[Tuple[int, int]]] = {}
    for w, matches in matches_per_week.items():
        row: List[Tuple[int, int]] = []
        for (i, j) in matches:
            d = (j - i) % n
            row.append((i, j) if d < n // 2 else (j, i))
        balanced[w] = row
    return balanced


class SMTPresolveNativeSolver(SMTBaseSolver):
    """Presolve variant using heuristic schedule + period labeling (Z3)."""

    def _build_model(self) -> Tuple[Solver, Dict[str, Any]]:
        n = self.n
        periods = n // 2
        raw = schedule_iterative_divide_and_conquer(n)
        matches_per_week = home_away_balance(raw, n)
        weeks = sorted(matches_per_week.keys())

        try:
            solver = Then('card2bv', 'smt').solver()
        except Exception:
            solver = Solver()
        solver.set(timeout=self.timeout * 1000)

        # Period assignment variables p[w,i,j] in [1..periods]
        p: Dict[Tuple[int, int, int], Any] = {}
        for w in weeks:
            for (i, j) in matches_per_week[w]:
                p[(w, i, j)] = Int(f"p_{w}_{i}_{j}")

        for var in p.values():
            solver.add(var >= 1, var <= periods)

        # Week 1 ordering (symmetry)
        w1 = weeks[0]
        for k, (i, j) in enumerate(sorted(matches_per_week[w1])):
            solver.add(p[(w1, i, j)] == k + 1)

        # One match per slot per week
        for w in weeks:
            for k in range(1, periods + 1):
                guards = [(p[(w, i, j)] == k, 1) for (i, j) in matches_per_week[w]]
                solver.add(PbEq(guards, 1))

        # At most two appearances per team per period
        teams = {t for w in weeks for (i, j) in matches_per_week[w] for t in (i, j)}
        for t in teams:
            for k in range(1, periods + 1):
                guards = [(p[(w, i, j)] == k, 1)
                          for w in weeks
                          for (i, j) in matches_per_week[w]
                          if t in (i, j)]
                solver.add(PbLe(guards, 2))

        return solver, {
            "p": p,
            "weeks": weeks,
            "periods": periods,
            "matches_per_week": matches_per_week,
        }

    def _solve_model(self, model: Tuple[Solver, Dict[str, Any]]) -> STSSolution:
        solver, state = model
        start = time.time()
        if solver.check().r == 1:
            m = solver.model()
            weeks: List[int] = state["weeks"]
            periods: int = state["periods"]
            p = state["p"]
            mpw = state["matches_per_week"]

            # Build weeks x periods matrix
            sol_weeks: List[List[List[int]]] = []
            for w in weeks:
                row: List[List[int]] = []
                for k in range(1, periods + 1):
                    for (a, b) in mpw[w]:
                        val = m.eval(p[(w, a, b)])
                        try:
                            assigned = val.as_long()  # type: ignore[attr-defined]
                        except Exception:
                            try:
                                assigned = int(str(val))
                            except Exception:
                                assigned = -1
                        if assigned == k:
                            row.append([a, b])
                            break
                sol_weeks.append(row)

            # Transpose to periods x weeks
            sol_periods: List[List[List[int]]] = []
            for k in range(periods):
                col: List[List[int]] = []
                for w in range(len(sol_weeks)):
                    col.append(sol_weeks[w][k])
                sol_periods.append(col)

            obj = None
            if self.optimization:
                home_counts = [0] * (self.n + 1)
                away_counts = [0] * (self.n + 1)
                for period_games in sol_periods:
                    for home, away in period_games:
                        if home and away:
                            home_counts[home] += 1
                            away_counts[away] += 1
                obj = sum(abs(home_counts[t] - away_counts[t]) for t in range(1, self.n + 1))

            return STSSolution(time=int(time.time() - start), optimal=True, obj=obj, sol=sol_periods)

        return STSSolution(time=self.timeout, optimal=False, obj=None, sol=[])

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="presolve",
            approach="SMT",
            version="1.0",
            supports_optimization=True,
            description="SMT presolve with period labeling (Z3)",
        )
