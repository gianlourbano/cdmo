"""Class-based optimized SMT solver using Z3 for STS."""

import time
from typing import Any, Dict, Tuple
from z3 import Solver, Int, If, Sum, And, sat

from ..utils.solution_format import STSSolution
from .base import SMTBaseSolver
from ..base_solver import SolverMetadata


class SMTOptimizedNativeSolver(SMTBaseSolver):
    """Optimized SMT formulation (class-based) mirroring z3_optimized."""

    def _build_model(self) -> Tuple[Solver, Dict[str, Any]]:
        weeks = self.weeks
        periods = self.periods
        n = self.n
        s = Solver()
        s.set("timeout", self.timeout * 1000)
        s.set("smt.arith.solver", 2)

        schedule: Dict[Tuple[int, int, int], Any] = {}
        for w in range(weeks):
            for p in range(periods):
                for slot in range(2):
                    var_name = f"s_{w}_{p}_{slot}"
                    var = Int(var_name)
                    schedule[w, p, slot] = var
                    s.add(var >= 1, var <= n)

        # Basic symmetry breaking
        s.add(schedule[0, 0, 0] == 1)

        # No self play
        for w in range(weeks):
            for p in range(periods):
                s.add(schedule[w, p, 0] != schedule[w, p, 1])

        # Each team exactly once per week
        for w in range(weeks):
            for t in range(1, n + 1):
                team_apps = []
                for p in range(periods):
                    for slot in range(2):
                        team_apps.append(If(schedule[w, p, slot] == t, 1, 0))
                s.add(Sum(team_apps) == 1)

        # Each pair exactly once
        for t1 in range(1, n + 1):
            for t2 in range(t1 + 1, n + 1):
                games = []
                for w in range(weeks):
                    for p in range(periods):
                        games.append(And(schedule[w, p, 0] == t1, schedule[w, p, 1] == t2))
                        games.append(And(schedule[w, p, 0] == t2, schedule[w, p, 1] == t1))
                s.add(Sum([If(g, 1, 0) for g in games]) == 1)

        # Period limit
        for p in range(periods):
            for t in range(1, n + 1):
                per_apps = []
                for w in range(weeks):
                    for slot in range(2):
                        per_apps.append(If(schedule[w, p, slot] == t, 1, 0))
                s.add(Sum(per_apps) <= 2)

        return s, {"schedule": schedule}

    def _solve_model(self, model: Tuple[Solver, Dict[str, Any]]) -> STSSolution:
        solver, state = model
        start = time.time()
        result = solver.check()
        elapsed = int(time.time() - start)
        if result == sat:
            m = solver.model()
            weeks = self.weeks
            periods = self.periods
            schedule = state["schedule"]
            sol = []
            for p in range(periods):
                period_games = []
                for w in range(weeks):
                    hv = m.eval(schedule[w, p, 0])
                    av = m.eval(schedule[w, p, 1])
                    try:
                        home = hv.as_long()  # type: ignore[attr-defined]
                    except Exception:
                        home = int(str(hv))
                    try:
                        away = av.as_long()  # type: ignore[attr-defined]
                    except Exception:
                        away = int(str(av))
                    period_games.append([home, away])
                sol.append(period_games)

            obj_value = None
            if self.optimization:
                # Use imbalance as simple objective like original function
                home_counts = [0] * (self.n + 1)
                away_counts = [0] * (self.n + 1)
                for period_games in sol:
                    for home, away in period_games:
                        home_counts[home] += 1
                        away_counts[away] += 1
                total_imbalance = sum(abs(home_counts[t] - away_counts[t]) for t in range(1, self.n + 1))
                obj_value = total_imbalance

            return STSSolution(time=elapsed, optimal=True, obj=obj_value, sol=sol)
        return STSSolution(time=min(elapsed, self.timeout), optimal=False, obj=None, sol=[])

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="optimized",
            approach="SMT",
            version="1.0",
            supports_optimization=True,
            description="Optimized SMT formulation (class-based)",
        )
