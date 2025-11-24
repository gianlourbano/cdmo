"""Class-based compact OR-Tools MIP formulation for STS."""

import time
from typing import Any, Optional, Dict, Tuple, List
from ortools.linear_solver import pywraplp

from ..utils.solution_format import STSSolution
from .base import MIPBaseSolver
from ..base_solver import SolverMetadata


class MIPCompactNativeSolver(MIPBaseSolver):
    """Compact MIP formulation using schedule variables (class-based)."""

    def __init__(self, n: int, timeout: int = 300, optimization: bool = False, backend: Optional[str] = None):
        super().__init__(n, timeout, optimization, backend)
        self._solver: Optional[pywraplp.Solver] = None
        self._vars: Dict[str, Any] = {}

    def _create_solver(self) -> pywraplp.Solver:
        name = (self.backend or "CBC").upper()
        if name == "SCIP":
            solver = pywraplp.Solver.CreateSolver("SCIP")
        elif name == "GUROBI":
            solver = pywraplp.Solver.CreateSolver("GUROBI")
        elif name == "CPLEX":
            solver = pywraplp.Solver.CreateSolver("CPLEX")
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

        home: Dict[Tuple[int, int], Any] = {}
        away: Dict[Tuple[int, int], Any] = {}
        for w in weeks:
            for p in periods:
                home[w, p] = solver.IntVar(1, self.n, f"home_{w}_{p}")
                away[w, p] = solver.IntVar(1, self.n, f"away_{w}_{p}")

        played: Dict[Tuple[int, int], Any] = {}
        for i in teams:
            for j in teams:
                if i < j:
                    played[i, j] = solver.BoolVar(f"played_{i}_{j}")

        # No team plays itself
        for w in weeks:
            for p in periods:
                solver.Add(home[w, p] != away[w, p])

        # Each team plays exactly once per week
        for k in teams:
            for w in weeks:
                appearances = []
                for p in periods:
                    appearances.append(solver.Sum([home[w, p] == k, away[w, p] == k]))
                solver.Add(solver.Sum(appearances) == 1)

        # Link schedule to matches + one pair per slot
        for w in weeks:
            for p in periods:
                slot_pairs = []
                for i in teams:
                    for j in teams:
                        if i < j:
                            match_ij = solver.BoolVar(f"match_{i}_{j}_{w}_{p}_home")
                            match_ji = solver.BoolVar(f"match_{j}_{i}_{w}_{p}_home")

                            solver.Add(match_ij * self.n <= home[w, p] - i + self.n - 1)
                            solver.Add(match_ij * self.n <= i - home[w, p] + self.n - 1)
                            solver.Add(match_ij * self.n <= away[w, p] - j + self.n - 1)
                            solver.Add(match_ij * self.n <= j - away[w, p] + self.n - 1)

                            solver.Add(match_ji * self.n <= home[w, p] - j + self.n - 1)
                            solver.Add(match_ji * self.n <= j - home[w, p] + self.n - 1)
                            solver.Add(match_ji * self.n <= away[w, p] - i + self.n - 1)
                            solver.Add(match_ji * self.n <= i - away[w, p] + self.n - 1)

                            solver.Add(played[i, j] >= match_ij)
                            solver.Add(played[i, j] >= match_ji)

                            slot_pairs.append(match_ij)
                            slot_pairs.append(match_ji)
                solver.Add(solver.Sum(slot_pairs) == 1)

        # Each pair plays exactly once
        for i in teams:
            for j in teams:
                if i < j:
                    solver.Add(played[i, j] == 1)

        # Period limit: at most twice per period per team
        for k in teams:
            for p in periods:
                period_apps = []
                for w in weeks:
                    period_apps.append(solver.Sum([home[w, p] == k, away[w, p] == k]))
                solver.Add(solver.Sum(period_apps) <= 2)

        # Symmetry breaking
        solver.Add(home[1, 1] == 1)
        if self.n >= 2:
            solver.Add(away[1, 1] == 2)
        if self.periods > 1:
            solver.Add(home[1, 1] < home[1, 2])
        if self.weeks > 1:
            solver.Add(home[1, 1] < home[2, 1])

        # Objective (optional)
        if self.optimization:
            teams_set = list(range(1, self.n + 1))
            home_count = {}
            away_count = {}
            for k in teams_set:
                home_apps = []
                away_apps = []
                for w in weeks:
                    for p in periods:
                        home_apps.append(home[w, p] == k)
                        away_apps.append(away[w, p] == k)
                home_count[k] = solver.Sum(home_apps)
                away_count[k] = solver.Sum(away_apps)
            deviation = {}
            for k in teams_set:
                deviation[k] = solver.NumVar(0, self.n - 1, f"dev_{k}")
                solver.Add(deviation[k] >= home_count[k] - away_count[k])
                solver.Add(deviation[k] >= away_count[k] - home_count[k])
            solver.Minimize(solver.Sum(deviation[k] for k in teams_set))

        self._solver = solver
        self._vars = {"home": home, "away": away}
        return solver, self._vars

    def _solve_model(self, model: Tuple[pywraplp.Solver, Dict[str, Any]]) -> STSSolution:
        solver, vars_dict = model
        start = time.time()
        status = solver.Solve()
        elapsed_time = int(time.time() - start)

        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            sol: List[List[List[int]]] = []
            home = vars_dict["home"]
            away = vars_dict["away"]
            for p in range(1, self.periods + 1):
                period_games = []
                for w in range(1, self.weeks + 1):
                    period_games.append([
                        int(home[w, p].solution_value()),
                        int(away[w, p].solution_value()),
                    ])
                sol.append(period_games)

            obj_value = None
            if self.optimization and solver.Objective():
                obj_value = int(solver.Objective().Value())

            return STSSolution(
                time=elapsed_time,
                optimal=(status == pywraplp.Solver.OPTIMAL),
                obj=obj_value,
                sol=sol,
            )

        return STSSolution(time=min(elapsed_time, self.timeout), optimal=False, obj=None, sol=[])

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="compact",
            approach="MIP",
            version="1.0",
            supports_optimization=False,
            description="Compact MIP formulation (class-based)",
        )
