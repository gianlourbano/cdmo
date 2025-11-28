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

_LAST_RAW_CP_OUTPUT: Dict[str, str] = {}


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


def _weeks_matrix_from_matchings(n: int, matchings: Dict[int, List[Tuple[int, int]]]) -> List[List[int]]:
    weeks_mat = [[0 for _ in range(n)] for _ in range(n)]
    for week_num, matches in matchings.items():
        for (i, j) in matches:
            weeks_mat[i - 1][j - 1] = week_num
            weeks_mat[j - 1][i - 1] = week_num
    return weeks_mat


def _format_week_matrix_mzn_from_array(week: List[List[int]]) -> str:
    rows = []
    for idx, row in enumerate(week):
        line = ", ".join(str(v) for v in row)
        if idx == len(week) - 1:
            rows.append(f"{line} |];")
        else:
            rows.append(f"{line} |")
    return "week = [|\n" + "\n".join(rows)


def _build_dzn_content(n: int, week_matrix: List[List[int]]) -> str:
    return (
        f"num_teams = {n};\n"
        f"num_weeks = {n - 1};\n"
        f"num_periods = {n // 2};\n\n"
        + _format_week_matrix_mzn_from_array(week_matrix)
        + "\n"
    )


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


def _circle_weeks(n: int):
    pivot, circle = n, list(range(1, n))
    #pivot, circle = n, list(range(2, n)) + [1]
    weeks = n - 1
    m = {}
    for w in range(1, weeks + 1):
        ms = [(pivot, circle[w-1])]
        for k in range(1, n//2):
            i = circle[(w-1 + k) % (n-1)]
            j = circle[(w-1 - k) % (n-1)]
            ms.append((i, j))
        m[w] = ms
    return m


def _reconstruct_schedule_from_period_home(n: int, week: List[List[int]], period: List[List[int]], home: List[List[int]]) -> List[List[List[int]]]:
    """Build schedule periods x weeks from period/home matrices.

    Returns: periods list each containing weeks list of [home, away].
    """
    weeks = n - 1
    periods = n // 2
    # Initialize structure periods x weeks
    schedule: List[List[List[int]]] = [[[] for _ in range(weeks)] for _ in range(periods)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            w = week[i - 1][j - 1]
            p = period[i - 1][j - 1]
            if w < 1 or p < 1:
                continue
            # Determine orientation: home[i-1][j-1] == 1 means i home
            if home[i - 1][j - 1] == 1:
                game = [i, j]
            else:
                game = [j, i]
            schedule[p - 1][w - 1] = game
    # Safety fill (MiniZinc should have assigned all games)
    for p in range(periods):
        for w in range(weeks):
            if not schedule[p][w]:  # fallback empty placeholder
                schedule[p][w] = [0, 0]
    return schedule


# (Legacy solve_cp function removed; unified bridge uses solve_cp_mzn)


def _rewrite_solve_line(model_text: str, strategy: str, optimization: bool) -> str:
    """Inject a chosen search strategy by replacing the 'solve' line.

    Strategies supported (case-insensitive):
    - ff_luby: first_fail, indomain_min + restart_luby(50)
    - ff_geometric: first_fail, indomain_min + restart_geometric(50)
    - plain: remove custom search annotations (just 'solve satisfy' or 'solve minimize total_imbalance')
    - raw:<annotation>: inject raw annotation after 'solve ::'

    If optimization is True or model declares minimize total_imbalance, we keep 'minimize total_imbalance'; otherwise 'satisfy'.
    """
    lines = model_text.splitlines()
    # Detect optimization intent from existing solve
    existing_opt = any("minimize" in l for l in lines if l.strip().startswith("solve"))
    do_opt = optimization or existing_opt

    def build(annotation: Optional[str]) -> str:
        if do_opt:
            tail = "minimize total_imbalance;"
        else:
            tail = "satisfy;"
        if annotation:
            return f"solve :: {annotation} {tail}"
        return f"solve {tail}"

    strategy_lower = strategy.lower()
    annotation = None
    if strategy_lower == "ff_luby":
        annotation = ("int_search([period[i,j] | i,j in Teams where i<j]++"
                      "[home[i,j] | i,j in Teams where i<j], first_fail, indomain_min) :: restart_luby(50)")
    elif strategy_lower == "ff_geometric":
        annotation = ("int_search([period[i,j] | i,j in Teams where i<j]++"
                      "[home[i,j] | i,j in Teams where i<j], first_fail, indomain_min) :: restart_geometric(50)")
    elif strategy_lower.startswith("raw:"):
        annotation = strategy[4:]
    elif strategy_lower == "plain":
        annotation = None
    else:
        # Unrecognized strategy -> leave original
        return model_text

    solve_pattern = re.compile(r"^\s*solve.*;\s*$")
    replaced = False
    for idx, line in enumerate(lines):
        if solve_pattern.match(line):
            lines[idx] = build(annotation)
            replaced = True
            break
    if not replaced:
        # Append solve line if missing
        lines.append(build(annotation))
    return "\n".join(lines) + "\n"


# (Old helper names replaced by explicit old_main-style variants above)


def solve_cp_mzn(
    model_path: Path,
    n: int,
    backend: str = "gecode",
    timeout: int = 300,
    search_strategy: Optional[str] = None,
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

        # EXACT old_main-style input handling
        matchings = circle_matchings(n)
        week_matrix = _weeks_matrix_from_matchings(n, matchings)
        dzn_content = _build_dzn_content(n, week_matrix)

        # Read model (no solve line rewrite since user requested exact behavior)
        model_text = model_path.read_text()
        tmp_mzn = tempfile.NamedTemporaryFile(delete=False, suffix=".mzn")
        tmp_mzn.write(model_text.encode("utf-8"))
        tmp_mzn.flush()
        m = Model(tmp_mzn.name)
        s = Solver.lookup(backend)
        inst = Instance(s, m)
        inst.add_string(dzn_content)

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
        contains_minimize = any("minimize" in line for line in model_text.splitlines() if line.strip().startswith("solve"))
        if contains_minimize:
            optimal_flag = status == Status.OPTIMAL_SOLUTION
        else:
            optimal_flag = status == Status.SATISFIED

        return STSSolution(time=min(elapsed, timeout), optimal=optimal_flag, obj=obj_val, sol=schedule_matrix)
    except Exception:
        elapsed = int(time.time() - start)
        if elapsed > timeout:
            elapsed = timeout
        return STSSolution(time=elapsed, optimal=False, obj=None, sol=[])
    
