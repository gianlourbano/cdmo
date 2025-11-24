"""Class-based simplified match formulation (previously function-based)."""

import time
from typing import Any
from ortools.linear_solver import pywraplp

from ..utils.solution_format import STSSolution
from .base import MIPBaseSolver
from ..base_solver import SolverMetadata


class MIPMatchSolver(MIPBaseSolver):
    """Simplified match-based MIP formulation using x/h variables.

    Metadata name: 'match'
    Supports optimization via home/away imbalance objective.
    """

    def _build_model(self) -> Any:  # Not separating model build; all in _solve_model
        return None

    def _solve_model(self, model: Any) -> STSSolution:  # noqa: D401
        n = self.n
        weeks = self.weeks
        periods = self.periods

        # Solver selection by backend (SCIP, GUROBI, CBC)
        backend = (self.backend or "CBC").upper()
        if backend not in {"SCIP", "GUROBI", "CBC"}:
            backend = "CBC"
        solver = pywraplp.Solver.CreateSolver(backend)
        if not solver:
            return STSSolution(time=0, optimal=False, obj=None, sol=[])
        solver.set_time_limit(self.timeout * 1000)

        # Variables
        x = {}
        h = {}
        for t1 in range(1, n + 1):
            for t2 in range(t1 + 1, n + 1):
                for w in range(weeks):
                    for p in range(periods):
                        x[t1, t2, w, p] = solver.BoolVar(f"x_{t1}_{t2}_{w}_{p}")
                        h[t1, t2, w, p] = solver.BoolVar(f"h_{t1}_{t2}_{w}_{p}")
                        solver.Add(h[t1, t2, w, p] <= x[t1, t2, w, p])

        # Constraints
        # 1. Each pair once
        for t1 in range(1, n + 1):
            for t2 in range(t1 + 1, n + 1):
                solver.Add(
                    solver.Sum(x[t1, t2, w, p] for w in range(weeks) for p in range(periods)) == 1,
                    f"pair_once_{t1}_{t2}"
                )
        # 2. Team once per week
        for t in range(1, n + 1):
            for w in range(weeks):
                weekly = []
                for t2 in range(1, n + 1):
                    if t2 == t:
                        continue
                    for p in range(periods):
                        if t < t2:
                            weekly.append(x[t, t2, w, p])
                        else:
                            weekly.append(x[t2, t, w, p])
                solver.Add(solver.Sum(weekly) == 1, f"team_week_{t}_{w}")
        # 3. One match per (week,period)
        for w in range(weeks):
            for p in range(periods):
                solver.Add(
                    solver.Sum(x[t1, t2, w, p] for t1 in range(1, n + 1) for t2 in range(t1 + 1, n + 1)) == 1,
                    f"slot_unique_{w}_{p}"
                )
        # 4. At most twice per period per team
        for t in range(1, n + 1):
            for p in range(periods):
                appear = []
                for t2 in range(1, n + 1):
                    if t2 == t:
                        continue
                    for w in range(weeks):
                        if t < t2:
                            appear.append(x[t, t2, w, p])
                        else:
                            appear.append(x[t2, t, w, p])
                solver.Add(solver.Sum(appear) <= 2, f"period_cap_{t}_{p}")
        # Symmetry
        if n >= 2:
            solver.Add(x[1, 2, 0, 0] == 1)
            solver.Add(h[1, 2, 0, 0] == 1)

        status = solver.Solve()
        elapsed = self.elapsed_time
        if status not in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE}:
            return STSSolution(time=min(elapsed, self.timeout), optimal=False, obj=None, sol=[])

        sol = [[] for _ in range(periods)]
        for w in range(weeks):
            for p in range(periods):
                for t1 in range(1, n + 1):
                    placed = False
                    for t2 in range(t1 + 1, n + 1):
                        if x[t1, t2, w, p].solution_value() > 0.5:
                            home_team, away_team = (t1, t2) if h[t1, t2, w, p].solution_value() > 0.5 else (t2, t1)
                            while len(sol[p]) <= w:
                                sol[p].append([0, 0])
                            sol[p][w] = [home_team, away_team]
                            placed = True
                            break
                    if placed:
                        continue
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
            name="match",
            approach="MIP",
            version="1.0",
            supports_optimization=True,
            description="Simplified match-based formulation with x/h variables",
            max_recommended_size=None,
        )