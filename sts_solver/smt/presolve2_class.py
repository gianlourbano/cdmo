"""Class-based SMT presolve_2 solver (Z3) with period constraints.

Adds optional optimization (home/away imbalance minimization) when the
`optimization` flag is set. This mirrors the CP optimization objective:
minimize the maximum absolute difference between home and away counts
per team (total_imbalance).
"""

import time
from typing import Any, Dict, List, Tuple
from z3 import Solver, Then, Int, Bool, If, Sum, PbEq, PbLe, Optimize, is_true

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


class SMTPresolve2NativeSolver(SMTBaseSolver):
    def _build_model(self) -> Tuple[Any, Dict[str, Any]]:  # Solver or Optimize
        n = self.n
        periods = n // 2
        matches_per_week = circle_matchings(n)
        weeks = sorted(matches_per_week.keys())

        # Choose solver kind (Optimize if optimization enabled)
        if self.optimization:
            solver = Optimize()
        else:
            try:
                solver = Then('card2bv', 'smt').solver()
            except Exception:
                solver = Solver()
        solver.set("timeout", self.timeout * 1000)
        if not self.optimization:
            solver.set("random_seed", 69)

        p: Dict[Tuple[int, int, int], Any] = {}
        home: Dict[Tuple[int, int, int], Bool] = {}
        for w in weeks:
            for (i, j) in matches_per_week[w]:
                p[(w, i, j)] = Int(f"p_{w}_{i}_{j}")
                if self.optimization:
                    home[(w, i, j)] = Bool(f"home_{w}_{i}_{j}")  # True => i is home, j away

        for var in p.values():  # domain bounds for period assignment
            solver.add(var >= 1, var <= periods)

        # Symmetry breaking: fix first week ordering (period indices)
        w1 = weeks[0]
        for k, (i, j) in enumerate(sorted(matches_per_week[w1])):
            solver.add(p[(w1, i, j)] == k + 1)

        # Exactly one match in each period per week
        for w in weeks:
            for k in range(1, periods + 1):
                guards = [(p[(w, i, j)] == k, 1) for (i, j) in matches_per_week[w]]
                solver.add(PbEq(guards, 1))

        teams = {t for w in weeks for (i, j) in matches_per_week[w] for t in (i, j)}
        # At most two appearances for a team in same period across the season
        for t in teams:
            for k in range(1, periods + 1):
                guards = [(p[(w, i, j)] == k, 1)
                          for w in weeks
                          for (i, j) in matches_per_week[w]
                          if t in (i, j)]
                solver.add(PbLe(guards, 2))

        # Optional deficient / double / per-team deficiency constraints (large instances)
        if periods * 2 >= 20:
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

        # Optimization objective (home/away imbalance)
        total_imbalance_var = None
        if self.optimization:
            # Define home/away counts implicitly via Boolean orientation.
            # home_count[t] = sum over matches where t is designated home
            # away_count[t] = sum over matches where t is designated away
            imbalance_vars: Dict[int, Int] = {}
            total_imbalance_var = Int("total_imbalance")

            for t in teams:
                home_terms = [If(home[(w, i, j)] if (w, i, j) in home else False,
                                 1 if i == t else 0,
                                 1 if j == t else 0)
                              for w in weeks for (i, j) in matches_per_week[w] if t in (i, j)]
                away_terms = [If(home[(w, i, j)] if (w, i, j) in home else False,
                                 1 if j == t else 0,
                                 1 if i == t else 0)
                              for w in weeks for (i, j) in matches_per_week[w] if t in (i, j)]
                home_sum = Sum(home_terms) if home_terms else Int(f"zero_home_{t}")
                away_sum = Sum(away_terms) if away_terms else Int(f"zero_away_{t}")
                imb = Int(f"imbalance_{t}")
                imbalance_vars[t] = imb
                # imb >= home_sum - away_sum and imb >= away_sum - home_sum
                solver.add(imb >= home_sum - away_sum)
                solver.add(imb >= away_sum - home_sum)
                # Non-negative bounds (implicit above) but keep explicit:
                solver.add(imb >= 0)
                solver.add(imb <= periods)  # loose upper bound

            # total_imbalance_var bounds + relation
            solver.add(total_imbalance_var >= 0)
            solver.add(total_imbalance_var <= periods)
            for imb in imbalance_vars.values():
                solver.add(total_imbalance_var >= imb)

            # Minimize the maximum imbalance
            if isinstance(solver, Optimize):
                solver.minimize(total_imbalance_var)

        return solver, {
            "p": p,
            "weeks": weeks,
            "periods": periods,
            "matches_per_week": matches_per_week,
            "home": home if self.optimization else None,
            "total_imbalance": total_imbalance_var,
        }

    def _solve_model(self, model: Tuple[Any, Dict[str, Any]]) -> STSSolution:  # Solver or Optimize
        solver, state = model
        start = time.time()
        result = solver.check()
        if result.r == 1:
            m = solver.model()
            weeks: List[int] = state["weeks"]
            periods: int = state["periods"]
            p = state["p"]
            mpw = state["matches_per_week"]
            home = state.get("home")

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
                            if home:
                                hv = m.eval(home[(w, a, b)])
                                if is_true(hv):
                                    row.append([a, b])  # a home
                                else:
                                    row.append([b, a])  # b home
                            else:
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
                total_imbalance_var = state.get("total_imbalance")
                if total_imbalance_var is not None:
                    try:
                        obj_val = m.eval(total_imbalance_var).as_long()
                    except Exception:
                        try:
                            obj_val = int(str(m.eval(total_imbalance_var)))
                        except Exception:
                            obj_val = None
                    obj = obj_val

            return STSSolution(time=int(time.time() - start), optimal=True, obj=obj, sol=sol_periods)

        return STSSolution(time=self.timeout, optimal=False, obj=None, sol=[])

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="presolve_2",
            approach="SMT",
            version="1.0",
            supports_optimization=True,
            description="SMT presolve_2 with additional period constraints (Z3); supports imbalance minimization",
        )
