"""Class-based formulations for compact and flow MIP models.

This file now provides classes instead of legacy solver functions.
"""

from typing import Any
from ortools.linear_solver import pywraplp

from ..utils.solution_format import STSSolution
from .base import MIPBaseSolver
from ..base_solver import SolverMetadata


class MIPFlowSolver(MIPBaseSolver):
    """Flow-based multi-commodity formulation.
    Name: 'flow'. Does not implement optimization objective.
    """

    def _build_model(self) -> Any:
        return None

    def _solve_model(self, model: Any) -> STSSolution:
        n = self.n
        weeks = list(range(1, n))
        periods = list(range(1, n // 2 + 1))
        backend = (self.backend or "CBC").upper()
        if backend not in {"SCIP", "GUROBI", "CBC"}:
            backend = "CBC"
        solver = pywraplp.Solver.CreateSolver(backend)
        if not solver:
            return STSSolution(time=0, optimal=False, obj=None, sol=[])
        solver.SetNumThreads(1)
        solver.SetTimeLimit(self.timeout * 1000)
        teams = list(range(1, n + 1))
        slots = [(w, p) for w in weeks for p in periods]
        pairs = [(i, j) for i in teams for j in teams if i < j]
        flow = {}
        for (i, j) in pairs:
            for (w, p) in slots:
                flow[i, j, w, p] = solver.BoolVar(f"flow_{i}_{j}_{w}_{p}")
        team_slot = {}
        for t in teams:
            for (w, p) in slots:
                for s in [0, 1]:
                    team_slot[t, w, p, s] = solver.BoolVar(f"team_{t}_{w}_{p}_{s}")
        # Constraints
        for (i, j) in pairs:
            solver.Add(solver.Sum(flow[i, j, w, p] for (w, p) in slots) == 1)
        for (w, p) in slots:
            solver.Add(solver.Sum(flow[i, j, w, p] for (i, j) in pairs) == 1)
        for (w, p) in slots:
            for (i, j) in pairs:
                solver.Add(flow[i, j, w, p] <= team_slot[i, w, p, 0] + team_slot[j, w, p, 0])
                solver.Add(flow[i, j, w, p] <= team_slot[i, w, p, 1] + team_slot[j, w, p, 1])
        for (w, p) in slots:
            solver.Add(solver.Sum(team_slot[t, w, p, 0] for t in teams) == 1)
            solver.Add(solver.Sum(team_slot[t, w, p, 1] for t in teams) == 1)
        for t in teams:
            for w in weeks:
                solver.Add(solver.Sum(team_slot[t, w, p, s] for p in periods for s in [0, 1]) == 1)
        for t in teams:
            for p in periods:
                solver.Add(solver.Sum(team_slot[t, w, p, s] for w in weeks for s in [0, 1]) <= 2)
        if pairs:
            i, j = pairs[0]
            solver.Add(flow[i, j, 1, 1] == 1)
        status = solver.Solve()
        elapsed = self.elapsed_time
        if status not in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE}:
            return STSSolution(time=min(elapsed, self.timeout), optimal=False, obj=None, sol=[])
        sol = []
        for p in periods:
            period_games = []
            for w in weeks:
                home_team, away_team = None, None
                for t in teams:
                    if team_slot[t, w, p, 0].solution_value() > 0.5:
                        home_team = t
                    if team_slot[t, w, p, 1].solution_value() > 0.5:
                        away_team = t
                if home_team and away_team:
                    period_games.append([home_team, away_team])
                else:
                    period_games.append([1, 2])
            sol.append(period_games)
        return STSSolution(
            time=min(elapsed, self.timeout),
            optimal=(status == pywraplp.Solver.OPTIMAL),
            obj=None,
            sol=sol,
        )

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="flow",
            approach="MIP",
            version="1.0",
            supports_optimization=False,
            description="Multi-commodity flow formulation (slots as nodes)",
        )