"""Class-based compact match formulation (previously function-based)."""

import time
from typing import Any
from ortools.linear_solver import pywraplp

from ..utils.solution_format import STSSolution
from .base import MIPBaseSolver
from ..base_solver import SolverMetadata


class MIPMatchCompactSolver(MIPBaseSolver):
    """True compact match-based formulation.

    Variables per pair: week, period, home flag.
    Name: 'match_compact'. Supports optimization (home/away balance).
    """

    def _build_model(self) -> Any:
        return None

    def _solve_model(self, model: Any) -> STSSolution:  # noqa: D401
        n = self.n
        weeks = self.weeks
        periods = self.periods
        backend = (self.backend or "CBC").upper()
        if backend not in {"SCIP", "GUROBI", "CBC"}:
            backend = "CBC"
        solver = pywraplp.Solver.CreateSolver(backend)
        if not solver:
            return STSSolution(time=0, optimal=False, obj=None, sol=[])
        solver.set_time_limit(self.timeout * 1000)

        match_week = {}
        match_period = {}
        match_t1_home = {}
        for t1 in range(1, n + 1):
            for t2 in range(t1 + 1, n + 1):
                key = (t1, t2)
                match_week[key] = solver.IntVar(0, weeks - 1, f"week_{t1}_{t2}")
                match_period[key] = solver.IntVar(0, periods - 1, f"period_{t1}_{t2}")
                match_t1_home[key] = solver.BoolVar(f"home_{t1}_{t2}")

        # Team once per week
        for t in range(1, n + 1):
            for w in range(weeks):
                weekly = []
                for t2 in range(1, n + 1):
                    if t2 == t:
                        continue
                    key = (t, t2) if t < t2 else (t2, t)
                    weekly.append(match_week[key] == w)
                solver.Add(solver.Sum(weekly) == 1, f"team_week_{t}_{w}")

        # Slot uniqueness via AND linearization
        week_period_match = {}
        for w in range(weeks):
            for p in range(periods):
                slot_vars = []
                for t1 in range(1, n + 1):
                    for t2 in range(t1 + 1, n + 1):
                        key = (t1, t2)
                        wp = solver.BoolVar(f"slot_{w}_{p}_{t1}_{t2}")
                        week_period_match[w, p, key] = wp
                        week_ind = solver.BoolVar(f"w_ind_{w}_{t1}_{t2}")
                        per_ind = solver.BoolVar(f"p_ind_{p}_{t1}_{t2}")
                        # match_week == w
                        solver.Add(week_ind * weeks >= match_week[key] - w + 1)
                        solver.Add(week_ind * weeks <= match_week[key] - w + weeks)
                        solver.Add((1 - week_ind) * weeks >= w - match_week[key] + 1)
                        # match_period == p
                        solver.Add(per_ind * periods >= match_period[key] - p + 1)
                        solver.Add(per_ind * periods <= match_period[key] - p + periods)
                        solver.Add((1 - per_ind) * periods >= p - match_period[key] + 1)
                        # AND
                        solver.Add(wp <= week_ind)
                        solver.Add(wp <= per_ind)
                        solver.Add(wp >= week_ind + per_ind - 1)
                        slot_vars.append(wp)
                solver.Add(solver.Sum(slot_vars) == 1, f"unique_slot_{w}_{p}")

        # At most twice per period per team
        for t in range(1, n + 1):
            for p in range(periods):
                appear = []
                for t2 in range(1, n + 1):
                    if t2 == t:
                        continue
                    key = (t, t2) if t < t2 else (t2, t)
                    appear.append(match_period[key] == p)
                solver.Add(solver.Sum(appear) <= 2, f"period_cap_{t}_{p}")

        # Symmetry
        if n >= 2:
            key = (1, 2)
            solver.Add(match_week[key] == 0)
            solver.Add(match_period[key] == 0)
            solver.Add(match_t1_home[key] == 1)

        status = solver.Solve()
        elapsed = self.elapsed_time
        if status not in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE}:
            return STSSolution(time=min(elapsed, self.timeout), optimal=False, obj=None, sol=[])

        sol = [[] for _ in range(periods)]
        for t1 in range(1, n + 1):
            for t2 in range(t1 + 1, n + 1):
                key = (t1, t2)
                week = int(match_week[key].solution_value())
                period = int(match_period[key].solution_value())
                home_first = match_t1_home[key].solution_value() > 0.5
                home_team, away_team = (t1, t2) if home_first else (t2, t1)
                while len(sol[period]) <= week:
                    sol[period].append([0, 0])
                sol[period][week] = [home_team, away_team]
        for p in range(periods):
            while len(sol[p]) < weeks:
                sol[p].append([0, 0])

        obj = None
        if self.optimization:
            home_counts = [0] * (n + 1)
            away_counts = [0] * (n + 1)
            for period_games in sol:
                for home, away in period_games:
                    if home > 0 and away > 0:
                        home_counts[home] += 1
                        away_counts[away] += 1
            obj = sum(abs(home_counts[t] - away_counts[t]) for t in range(1, n + 1))

        return STSSolution(
            time=min(elapsed, self.timeout),
            optimal=(status == pywraplp.Solver.OPTIMAL),
            obj=obj,
            sol=sol,
        )

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="match_compact",
            approach="MIP",
            version="1.0",
            supports_optimization=True,
            description="True compact match-based formulation (week/period/home per pair)",
        )