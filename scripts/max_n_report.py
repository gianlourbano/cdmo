import os
import json
from typing import Dict, Tuple, List


def is_solution(entry: Dict) -> bool:
    # Consider solved if 'optimal' is True or 'sol' exists and is non-empty
    if isinstance(entry, dict):
        if entry.get("optimal") is True:
            return True
        sol = entry.get("sol")
        if isinstance(sol, list) and len(sol) > 0:
            return True
        status = entry.get("status")
        if isinstance(status, str) and status.upper() in {"FEAS", "OPTIMAL", "SAT"}:
            return True
    return False


def collect_max_n(res_root: str) -> Dict[Tuple[str, str], int]:
    result: Dict[Tuple[str, str], int] = {}

    # Iterate approaches (e.g., CP, MIP, SMT, SAT)
    for approach in os.listdir(res_root):
        approach_dir = os.path.join(res_root, approach)
        if not os.path.isdir(approach_dir):
            continue

        # Iterate instance size files (e.g., 6.json, 8.json, ...)
        for fname in os.listdir(approach_dir):
            if not fname.endswith('.json'):
                continue
            try:
                n = int(os.path.splitext(fname)[0])
            except ValueError:
                continue

            path = os.path.join(approach_dir, fname)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
            except Exception:
                continue

            # Data can be a dict of model variants or a list; normalize to list of (model_name, entry)
            items: List[Tuple[str, Dict]] = []
            if isinstance(data, dict):
                for key, entry in data.items():
                    # key example: "compact" or "opt-smart_dt" etc.
                    model = str(key)
                    if isinstance(entry, dict):
                        items.append((model, entry))
                    elif isinstance(entry, list):
                        # pick entries with solution
                        for e in entry:
                            if isinstance(e, dict):
                                items.append((model, e))
            elif isinstance(data, list):
                for entry in data:
                    # unknown model; tag as "unknown"
                    if isinstance(entry, dict):
                        model = str(entry.get("model") or entry.get("variant") or "unknown")
                        items.append((model, entry))

            # Update max n per (approach, model)
            for model, entry in items:
                if is_solution(entry):
                    key = (approach, model)
                    result[key] = max(result.get(key, 0), n)

    return result


def main():
    workspace_root = os.path.dirname(os.path.dirname(__file__))
    res_root = os.path.join(workspace_root, 'res')
    if not os.path.isdir(res_root):
        print("res directory not found")
        return

    result = collect_max_n(res_root)
    # Group by approach for prettier output
    grouped: Dict[str, List[Tuple[str, int]]] = {}
    for (approach, model), max_n in result.items():
        grouped.setdefault(approach, []).append((model, max_n))

    # Write findings to file in format: (approach, model_name, max_n, optimization)
    # optimization inferred by model name prefix 'opt-'; strip 'opt-' from model name
    out_lines = []
    for approach in sorted(grouped.keys()):
        for model, max_n in sorted(grouped[approach], key=lambda x: (x[0], x[1])):
            opt = False
            mname = model
            if mname.startswith('opt-'):
                opt = True
                mname = mname[len('opt-'):]
            solver_name = None
            if approach.upper() == 'CP':
                # Expect model names like 'base-chuffed', 'opt-base-gecode', etc.
                if '-' in mname:
                    parts = mname.split('-')
                    if len(parts) >= 2:
                        solver_name = parts[-1]
                        mname = '-'.join(parts[:-1])
            out_lines.append(f"({approach}, {mname}, {max_n}, {opt}, {solver_name})")

    out_path = os.path.join(workspace_root, 'res', 'max_n_summary.txt')
    try:
        with open(out_path, 'w') as f:
            for line in out_lines:
                f.write(line + "\n")
    except Exception as e:
        print(f"Failed to write summary: {e}")

    # Also print to stdout for convenience
    for line in out_lines:
        print(line)


if __name__ == '__main__':
    main()
