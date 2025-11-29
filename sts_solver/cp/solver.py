"""Constraint Programming solver logic for STS using MiniZinc.

Supports two modes:
 - Satisfy only (legacy): uses `sts.mzn` model producing a flat `schedule` array
 - Optimization (home/away imbalance minimization): uses
   `circle_method_SB_modified_new_sb.mzn` and reconstructs schedule from
   `period` and `home` decision variables plus a generated circle-method
   week matrix.

Objective reported is `total_imbalance` from optimization model when present.
"""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import tempfile
import re

from datetime import timedelta
from minizinc import Model, Solver, Instance
from minizinc.result import Status

from ..utils.solution_format import STSSolution

# ==== Old main style helpers (input & output handling) ====
def circle_matchings(n: int) -> Dict[int, List[Tuple[int, int]]]:
    pivot = n
    circle = list(range(1, n))
    weeks = n - 1
    matchings: Dict[int, List[Tuple[int, int]]] = {}
    for w in range(1, weeks + 1):
        ms: List[Tuple[int, int]] = [(pivot, circle[w - 1])]
        for k in range(1, n // 2):
            i = circle[(w - 1 + k) % (n - 1)]
            j = circle[(w - 1 - k) % (n - 1)]
            ms.append((i, j))
        matchings[w] = ms
    return matchings

import numpy as np

def generate_dzn(n, matchings, filename):
    with open(filename, 'w') as f:
        f.write(f"num_teams = {n};\n")
        f.write(f"num_weeks = {n - 1};\n")
        f.write(f"num_periods = {n // 2};\n\n")

        weeks = np.zeros((n, n), dtype=int)
        for week_num, matches in matchings.items():
            for match in matches:
                weeks[match[0] - 1, match[1] - 1] = week_num
                weeks[match[1] - 1, match[0] - 1] = week_num

        f.write("week = [|\n")
        for i in range(n):
            row = ", ".join(str(weeks[i, j]) for j in range(n))
            if i == n - 1:
                f.write(f"{row} |];\n")  
            else:
                f.write(f"{row} |\n")


def _solution_transform(n: int, matchings: Dict[int, List[Tuple[int, int]]], period_matrix, home_matrix) -> List[List[List[int]]]:
    periods = n // 2
    weeks = n - 1
    matrix: List[List[List[int]]] = [[[] for _ in range(weeks)] for _ in range(periods)]
    for week in range(1, weeks + 1):
        matches = matchings.get(week, [])
        for (i, j) in matches:
            p = period_matrix[i - 1][j - 1]
            home_team, away_team = (i, j) if home_matrix[i - 1][j - 1] else (j, i)
            matrix[p - 1][week - 1] = [home_team, away_team]
    # Fill any missing cells with [0,0] for robustness
    for p in range(periods):
        for w in range(weeks):
            if not matrix[p][w]:
                matrix[p][w] = [0, 0]
    return matrix

def solve_cp_mzn(
    model_path: Path,
    n: int,
    backend: str = "gecode",
    timeout: int = 300,
    search_strategy: Optional[str] = None, # DEPRECATED
    optimization: bool = False,
) -> STSSolution:
    """Execute MiniZinc model via native API; extract period/home and objective.

    We no longer rely on printed output; variables are accessed directly from
    the result object. Schedule reconstruction mimics legacy approach.
    """
    start = time.time()
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"MiniZinc model not found: {model_path}")

        import os
        # Build matchings and week matrix; write stable dzn next to model (no temp model copy)
        matchings = circle_matchings(n)
        dzn_file = os.path.join("source", "CP", "circle_method.dzn")
        generate_dzn(n, matchings, dzn_file)
        m = Model(str(model_path))
        s = Solver.lookup(backend)
        inst = Instance(s, m)
        inst.add_file(str(dzn_file))
        remaining = timeout - (time.time() - start)
        if remaining <= 0:
            return STSSolution(time=timeout, optimal=False, obj=None, sol=[])

        results = inst.solve(timeout=timedelta(seconds=remaining), random_seed=42)
        elapsed = int(time.time() - start)

        status = results.status
        if status in (Status.UNSATISFIABLE, Status.UNKNOWN):
            optimal_flag = status == Status.UNSATISFIABLE
            return STSSolution(time=min(elapsed, timeout), optimal=optimal_flag, obj=None, sol=[])

        # Output handling identical to old_main: extract matrices, build schedule matrix
        try:
            period_raw = results["period"]
            home_raw = results["home"]
        except Exception:
            return STSSolution(time=min(elapsed, timeout), optimal=False, obj=None, sol=[])

        # Transform to (n/2) x (n-1) matrix of [home, away]
        schedule_matrix = _solution_transform(n, matchings, period_raw, home_raw)

        # Objective (may be absent for satisfy-only models)
        try:
            obj_val = results.objective
        except Exception:
            obj_val = None
        if isinstance(obj_val, float):
            try:
                obj_val = int(obj_val)
            except Exception:
                obj_val = None
        elif not isinstance(obj_val, int):
            obj_val = None

        # Optimal flag semantics: decisional (satisfy) treats SATISFIED as optimal; optimization requires OPTIMAL_SOLUTION
        if optimization:
            optimal_flag = status == Status.OPTIMAL_SOLUTION
        else:
            optimal_flag = status == Status.SATISFIED

        return STSSolution(time=min(elapsed, timeout), optimal=optimal_flag, obj=obj_val, sol=schedule_matrix)
    except Exception:
        elapsed = int(time.time() - start)
        if elapsed > timeout:
            elapsed = timeout
        return STSSolution(time=elapsed, optimal=False, obj=None, sol=[])
    # No temp files to clean up (stable DZN intentionally retained)
    
