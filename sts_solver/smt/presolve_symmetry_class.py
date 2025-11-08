"""Class-based SMT presolve_symmetry solver (Z3) with stronger symmetry breaking."""

import time
from typing import Any, Dict, List, Tuple
from z3 import Solver, Then, Int, Bool, PbEq, PbLe

from ..utils.solution_format import STSSolution
from .base import SMTBaseSolver
from ..base_solver import SolverMetadata


def circle_matchings(n: int) -> Dict[int, List[Tuple[int, int]]]:
    pivot, circle = n, list(range(1, n))
    weeks = n - 1
    m: Dict[int, List[Tuple[int, int]]] = {}
    for w in range(weeks):
        ms: List[Tuple[int, int]] = [(pivot, circle[w])]
        for k in range(1, n // 2):
            i = circle[(w + k) % (n - 1)]
            j = circle[(w - k) % (n - 1)]
            a, b = (i, j) if i < j else (j, i)
            ms.append((a, b))
        m[w] = ms
    return m


class SMTPresolveSymmetryNativeSolver(SMTBaseSolver):
    def _build_model(self) -> Tuple[Solver, Dict[str, Any]]:
        n = self.n
        periods = n // 2
        matches_per_week = circle_matchings(n)
        weeks = sorted(matches_per_week.keys())

        try:
            solver = Then('card2bv', 'smt').solver()
        except Exception:
            solver = Solver()
        solver.set(timeout=self.timeout * 1000)

        p: Dict[Tuple[int, int, int], Any] = {}
        for w in weeks:
            for (i, j) in matches_per_week[w]:
                p[(w, i, j)] = Int(f"p_{w}_{i}_{j}")

        for var in p.values():
            solver.add(var >= 1, var <= periods)

        # Stronger symmetry: increasing order of periods in week 1
        w1 = weeks[0]
        week1_matches = sorted(matches_per_week[w1])
        for k, (i, j) in enumerate(week1_matches):
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

        # Optional MiniZinc-inspired constraints for larger instances
        if periods * 2 >= 22:
            for k in range(1, periods + 1):
                deficient_guards = []
                for t in teams:
                    period_guards = [(p[(w, i, j)] == k, 1)
                                     for w in weeks
                                     for (i, j) in matches_per_week[w]
                                     if t in (i, j)]
                    if period_guards:
                        deficient_var = Bool(f"deficient_{t}_{k}")
                        solver.add(deficient_var == PbEq(period_guards, 1))
                        deficient_guards.append((deficient_var, 1))
                if deficient_guards:
                    solver.add(PbEq(deficient_guards, 2))

            for k in range(1, periods + 1):
                double_guards = []
                for t in teams:
                    period_guards = [(p[(w, i, j)] == k, 1)
                                     for w in weeks
                                     for (i, j) in matches_per_week[w]
                                     if t in (i, j)]
                    if period_guards:
                        double_var = Bool(f"double_{t}_{k}")
                        solver.add(double_var == PbEq(period_guards, 2))
                        double_guards.append((double_var, 1))
                if double_guards:
                    solver.add(PbEq(double_guards, len(teams) - 2))

            for t in teams:
                deficiency_guards = []
                for k in range(1, periods + 1):
                    period_guards = [(p[(w, i, j)] == k, 1)
                                     for w in weeks
                                     for (i, j) in matches_per_week[w]
                                     if t in (i, j)]
                    if period_guards:
                        def_var = Bool(f"def_period_{t}_{k}")
                        solver.add(def_var == PbEq(period_guards, 1))
                        deficiency_guards.append((def_var, 1))
                if deficiency_guards:
                    solver.add(PbLe(deficiency_guards, 1))

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
            name="presolve_symmetry",
            approach="SMT",
            version="1.0",
            supports_optimization=True,
            description="SMT presolve with stronger symmetry breaking (Z3)",
        )
