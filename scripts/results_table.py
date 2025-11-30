import os
import json
from typing import Dict, List, Tuple


def infer_optimization(model_name: str) -> bool:
    return model_name.startswith('opt-')


def normalize_model(model_name: str) -> str:
    return model_name[len('opt-'):] if model_name.startswith('opt-') else model_name


def load_entries(path: str) -> List[Tuple[str, Dict]]:
    entries: List[Tuple[str, Dict]] = []
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception:
        return entries

    if isinstance(data, dict):
        for model, entry in data.items():
            if isinstance(entry, dict):
                entries.append((model, entry))
            elif isinstance(entry, list):
                for e in entry:
                    if isinstance(e, dict):
                        entries.append((model, e))
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                model = str(entry.get('model') or entry.get('variant') or 'unknown')
                entries.append((model, entry))
    return entries


def collect(res_root: str):
    # structure: results[approach][opt_flag][model] = list of rows (n, time, obj)
    results: Dict[str, Dict[bool, Dict[str, List[Tuple[int, float, int]]]]] = {}
    for approach in os.listdir(res_root):
        app_dir = os.path.join(res_root, approach)
        if not os.path.isdir(app_dir):
            continue
        for fname in os.listdir(app_dir):
            if not fname.endswith('.json'):
                continue
            try:
                n = int(os.path.splitext(fname)[0])
            except ValueError:
                continue
            path = os.path.join(app_dir, fname)
            for model_name, entry in load_entries(path):
                opt = infer_optimization(model_name)
                mname = normalize_model(model_name)
                time = entry.get('time')
                obj = entry.get('obj')
                # Some files use seconds as int, ensure type
                try:
                    time = float(time) if time is not None else None
                except Exception:
                    time = None
                try:
                    obj = int(obj) if obj is not None else None
                except Exception:
                    obj = None
                results.setdefault(approach, {}).setdefault(opt, {}).setdefault(mname, []).append((n, time or -1.0, obj if obj is not None else -1))
    return results


def to_markdown(results) -> str:
    lines: List[str] = []
    for approach in sorted(results.keys()):
        lines.append(f"## {approach}")
        for opt in (False, True):
            if opt not in results[approach]:
                continue
            title = "Optimization" if opt else "Decision"
            lines.append(f"### {title}")
            lines.append("| Model | n | Time (s) | Objective |")
            lines.append("|---|---:|---:|---:|")
            for model in sorted(results[approach][opt].keys()):
                rows = sorted(results[approach][opt][model], key=lambda r: r[0])
                for n, t, obj in rows:
                    obj_str = "-" if not opt else (str(obj) if obj >= 0 else "-")
                    t_str = f"{t:.0f}" if t >= 0 else "-"
                    lines.append(f"| {model} | {n} | {t_str} | {obj_str} |")
            lines.append("")
    return "\n".join(lines)


def to_latex(results) -> str:
    lines: List[str] = []
    for approach in sorted(results.keys()):
        lines.append(f"% {approach}")
        for opt in (False, True):
            if opt not in results[approach]:
                continue
            title = "Optimization" if opt else "Decision"
            lines.append("\\paragraph{" + title + "}")
            lines.append("\\begin{center}")
            lines.append("\\begin{tabular}{|l|c|c|c|}")
            lines.append("\\hline")
            lines.append("Model & n & Time (s) & Objective \\\\ ")
            lines.append("\\hline")
            for model in sorted(results[approach][opt].keys()):
                rows = sorted(results[approach][opt][model], key=lambda r: r[0])
                for n, t, obj in rows:
                    obj_str = "-" if not opt else (str(obj) if obj >= 0 else "-")
                    t_str = f"{int(t)}" if t >= 0 else "-"
                    lines.append("{} & {} & {} & {} \\".format(model, n, t_str, obj_str))
            lines.append("\\hline")
            lines.append("\\end{tabular}")
            lines.append("\\end{center}")
    return "\n".join(lines)


def main():
    workspace_root = os.path.dirname(os.path.dirname(__file__))
    res_root = os.path.join(workspace_root, 'res')
    results = collect(res_root)

    md = to_markdown(results)
    with open(os.path.join(res_root, 'results_table.md'), 'w') as f:
        f.write(md)

    tex = to_latex(results)
    with open(os.path.join(res_root, 'results_table.tex'), 'w') as f:
        f.write(tex)

    print("Generated:")
    print(" - res/results_table.md")
    print(" - res/results_table.tex")


if __name__ == '__main__':
    main()
