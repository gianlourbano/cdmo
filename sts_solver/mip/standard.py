"""Class-based standard OR-Tools MIP formulation for STS."""

import time
from typing import Any, Dict, Tuple
from ortools.linear_solver import pywraplp

from ..utils.solution_format import STSSolution
from .base import MIPBaseSolver
from ..base_solver import SolverMetadata


class MIPStandardNativeSolver(MIPBaseSolver):
    """Standard MIP formulation using match variables (class-based)."""

    def _create_solver(self) -> pywraplp.Solver:
        name = (self.backend or "CBC").upper()
        if name == "SCIP":
            solver = pywraplp.Solver.CreateSolver("SCIP")
        elif name == "GUROBI":
            solver = pywraplp.Solver.CreateSolver("GUROBI")
        elif name == "CPLEX":
            solver = pywraplp.Solver.CreateSolver("CPLEX")
        elif name == "GLOP":
            # LP-only; not supported for discrete model
            raise RuntimeError("GLOP is not supported for MIP STS")
        else:
            solver = pywraplp.Solver.CreateSolver("CBC")
        if not solver:
            raise RuntimeError("Failed to create OR-Tools solver")
        solver.SetNumThreads(1)
        solver.SetTimeLimit(self.timeout * 1000)
        return solver

    def _build_model(self) -> Tuple[pywraplp.Solver, Dict[str, Any]]:
        solver = self._create_solver()
        teams = list(range(1, self.n + 1))
        weeks = list(range(1, self.weeks + 1))
        periods = list(range(1, self.periods + 1))

        match_vars = {}
        for i in teams:
            for j in teams:
                if i == j:
                    continue
                for w in weeks:
                    for p in periods:
                        match_vars[i, j, w, p] = solver.BoolVar(f"match_{i}_{j}_{w}_{p}")

        # Pair plays exactly once
        for i in teams:
            for j in teams:
                if i < j:
                    solver.Add(
                        solver.Sum(match_vars[i, j, w, p] + match_vars[j, i, w, p]
                                   for w in weeks for p in periods) == 1
                    )

        # One game per team per week
        for k in teams:
            for w in weeks:
                solver.Add(
                    solver.Sum(match_vars[k, j, w, p] for j in teams if k != j for p in periods)
                    + solver.Sum(match_vars[i, k, w, p] for i in teams if i != k for p in periods)
                    == 1
                )

        # One game per slot
        for w in weeks:
            for p in periods:
                solver.Add(
                    solver.Sum(match_vars[i, j, w, p] for i in teams for j in teams if i != j) == 1
                )

        # At most two per period per team
        for k in teams:
            for p in periods:
                solver.Add(
                    solver.Sum(match_vars[k, j, w, p] for j in teams if k != j for w in weeks)
                    + solver.Sum(match_vars[i, k, w, p] for i in teams if i != k for w in weeks)
                    <= 2
                )

        # Optional objective
        if self.optimization:
            home_games = {k: solver.Sum(match_vars[k, j, w, p] for j in teams if k != j for w in weeks for p in periods) for k in teams}
            away_games = {k: solver.Sum(match_vars[i, k, w, p] for i in teams if i != k for w in weeks for p in periods) for k in teams}
            deviation = {k: solver.NumVar(0, solver.infinity(), f"dev_{k}") for k in teams}
            for k in teams:
                solver.Add(deviation[k] >= home_games[k] - away_games[k])
                solver.Add(deviation[k] >= away_games[k] - home_games[k])
            solver.Minimize(solver.Sum(deviation[k] for k in teams))

        return solver, {"match_vars": match_vars, "weeks": weeks, "periods": periods}

    def _solve_model(self, model: Tuple[pywraplp.Solver, Dict[str, Any]]) -> STSSolution:
        solver, state = model
        start = time.time()
        status = solver.Solve()
        elapsed = int(time.time() - start)

        match_vars = state["match_vars"]
        weeks = state["weeks"]
        periods = state["periods"]

        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            schedule = [[None for _ in weeks] for _ in periods]
            for (i, j, w, p), var in match_vars.items():
                if var.solution_value() > 0.5:
                    schedule[p - 1][w - 1] = [i, j]

            sol = []
            for p_idx in range(len(periods)):
                period_games = []
                for w_idx in range(len(weeks)):
                    game = schedule[p_idx][w_idx]
                    period_games.append(game if game else [1, 2])
                sol.append(period_games)

            obj_value = None
            if self.optimization and solver.Objective():
                obj_value = int(solver.Objective().Value())

            return STSSolution(time=elapsed, optimal=(status == pywraplp.Solver.OPTIMAL), obj=obj_value, sol=sol)

        return STSSolution(time=min(elapsed, self.timeout), optimal=False, obj=None, sol=[])

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="standard",
            approach="MIP",
            version="1.0",
            supports_optimization=True,
            description="Standard MIP formulation (class-based)",
        )
