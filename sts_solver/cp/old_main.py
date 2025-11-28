from minizinc import Instance, Model, Solver
from minizinc.result import Status
import time
from datetime import timedelta
import numpy as np
import argparse
import re
import json
import os
import math

def circle_matchings(n):
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

def parse_n_teams(n_input):
    """
    Parses the input for -n argument, allowing range input like 2-18.
    Ensures only even numbers are returned.
    """
    result = set()
    for item in n_input:
        if re.match(r"^\d+-\d+$", item):  # range type: 2-18
            start, end = map(int, item.split("-"))
            for n in range(start, end + 1):
                if n % 2 == 0:
                    result.add(n)
        else:
            try:
                n = int(item)
                if n % 2 == 0:
                    result.add(n)
                else:
                    print(f"[WARNING] Skipping odd number: {n}")
            except ValueError:
                print(f"[WARNING] Invalid value for -n: {item}")
    return sorted(result)

def human_readable_schedule(schedule_tuple):
    """
    schedule_tuple: tuple di 3 elementi
        1. dict: {week_number: [(team1, team2), ...]}
        2. 2D array: period[i][j] = period number per la coppia (i,j)
        3. 2D array: home[i][j] = True se i gioca in casa, False se j gioca in casa
    """
    match_dict, period_matrix, home_matrix = schedule_tuple

    for week in sorted(match_dict.keys()):
        print(f"\n=== Week {week} ===")
        matches = match_dict[week]
        
        matches_sorted = sorted(matches, key=lambda match: period_matrix[match[0]-1][match[1]-1])
        
        for match in matches_sorted:
            i, j = match
            period = period_matrix[i-1][j-1]
            home_team, away_team = (i, j) if home_matrix[i-1][j-1] else (j, i)
            print(f"Period {period}: Team {home_team} (home) vs Team {away_team} (away)")

def solution_transform(num_teams, schedule_tuple):
    """
    Transforms schedule tuple into a (n/2) x (n-1) matrix, 
    where each entry is [home_team, away_team].
    """
    match_dict, period_matrix, home_matrix = schedule_tuple
    n = num_teams
    periods = n // 2
    weeks = n - 1

    matrix = [[None for _ in range(weeks)] for _ in range(periods)]

    for week in range(1, weeks + 1):
        matches = match_dict[week]
        for match in matches:
            i, j = match
            period = period_matrix[i-1][j-1]
            home_team, away_team = (i, j) if home_matrix[i-1][j-1] else (j, i)
            matrix[period-1][week-1] = [home_team, away_team]

    return matrix

def save_results_as_json(n, results, model_name, output_dir="/res/CP"):
    """
    Saves the results dictionary to a JSON file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    json_path = os.path.join(output_dir, f"{n}.json")
    
    json_obj = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            try:
                json_obj = json.load(f)
            except json.JSONDecodeError:
                json_obj = {}
                
    result = {}
    result[model_name] = results
    
    for method, res in result.items():
        runtime = res.get("time", 300.0)
        
        time_field = 300 if not res.get("optimal") else math.floor(runtime)
        sol = res.get("sol")
        matrix = solution_transform(n, sol) if sol else []
        
        json_obj[method] = {
            "time": time_field,
            "optimal": res.get("optimal"),
            "obj": res.get("obj"),
            "sol": matrix,
        }
    
    with open(json_path, "w") as f:
        json.dump(json_obj, f, indent=1)
    

def run_minizinc(n, dzn_file, model_file, solver, timeout, init_time):
    matchings = circle_matchings(n)
    dzn_file = os.path.join("source", "CP", "circle_method.dzn")
    generate_dzn(n, matchings, dzn_file)
    model = Model(os.path.join("source", "CP", model_file))
    solver = Solver.lookup(solver)
    instance = Instance(solver, model)
    instance.add_file(dzn_file)
    remaining_time = timeout - (time.perf_counter() - init_time)
    results = instance.solve(timeout=timedelta(seconds=remaining_time), random_seed=42)
    return results, matchings

def solve_cp_decisional(n, timeout, solver, search_strategy="base", symmetry_breaking='N'):
    
    if n % 2 != 0:
        raise ValueError("Number of teams must be even.")
    init_time = time.perf_counter()


    model_file = f"d_circle_method_{symmetry_breaking}_{search_strategy}.mzn"
    dzn_file = "circle_method.dzn"

    results, matchings = run_minizinc(n, dzn_file, model_file, solver, timeout, init_time)
    
    end_time = time.perf_counter()
    solve_time = end_time - init_time
    status = results.status
    
    if status == Status.SATISFIED:
        result = {
            'obj': results.objective,
            'sol': (matchings, results['period'], results['home']),
            'optimal': True,
            'time': solve_time
        }
    elif status == Status.UNSATISFIABLE:
        result = {
            'obj': None,
            'sol': [],
            'optimal': True,
            'time': 0
        }
    elif status == Status.UNKNOWN:
        result = {
            'obj': None,
            'sol': None,
            'optimal': False,
            'time': solve_time
        }
    return result

def solve_cp_optimization(n, timeout, solver, search_strategy="base", symmetry_breaking=True):
    
    if n % 2 != 0:
        raise ValueError("Number of teams must be even.")
    init_time = time.perf_counter()

    model_file = f"circle_method_{symmetry_breaking}_{search_strategy}.mzn"
    dzn_file = "circle_method.dzn"

    results, matchings = run_minizinc(n, dzn_file, model_file, solver, timeout, init_time)
    
    end_time = time.perf_counter()
    solve_time = end_time - init_time
    status = results.status

    if status == Status.OPTIMAL_SOLUTION:
        result = {
            'obj': results.objective,
            'sol': (matchings, results["period"], results["home"]),
            'optimal': True,
            'time': solve_time
        }
    elif status == Status.SATISFIED:
        result = {
            'obj': results.objective,
            'sol': (matchings, results['period'], results['home']),
            'optimal': False,
            'time': solve_time
        }
    elif status == Status.UNSATISFIABLE:
        result = {
            'obj': None,
            'sol': [],
            'optimal': True,
            'time': 0
        }
    elif status == Status.UNKNOWN:
        result = {
            'obj': None,
            'sol': None,
            'optimal': False,
            'time': solve_time
        }
    return result
 
def main():
    parser = argparse.ArgumentParser(description="Sport Tournament Scheduler using CP solvers.")
    parser.add_argument(
        "-n", "--n_teams",
        type=str,
        nargs='+',
        default=["2-18"],
        help="List of even numbers or ranges like 2-18 for number of teams to test."
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each solver instance."
    )
    parser.add_argument(
        "-s", "--solver",
        type=str,
        choices=["gecode", "chuffed"],
        default="gecode",
        nargs="+",
        help="The solver to use (gecode, chuffed)."
    )
    parser.add_argument(
        "-ss", "--search_strategy",
        type=str,
        choices=["base", "base_modified", "modified_new_sb","wd_random", "dwd_r_Luby", "dwd_r_rr", "ff_split", "ro_luby"],
        default="base",
        nargs="+",
        help="The search strategy to use (base, dwd_random, dwd_r_Luby, dwd_r_rr, ff_split, ro_luby)."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all the possible combinations to solve the problem."
    )
    parser.add_argument(
        "--run_decisional",
        action="store_true",
        help="Run the decisional solver."
    )
    parser.add_argument(
        "--run_optimization",
        action="store_true",
        help="Run the optimization solver."
    )
    parser.add_argument(
        "--sb",
        type=str,
        choices=["Y", "N"],
        default="N",
        nargs="+",
        help="Control symmetry breaking."
    )
    parser.set_defaults(sb=False)
    parser.add_argument(
        "--save_json",
        action='store_true',
        help="Save solver results to JSON files."
    )

    args = parser.parse_args()

    # Parse and validate number of teams
    args.n_teams = parse_n_teams(args.n_teams)

    if not args.run_decisional and not args.run_optimization:
        print("Error: You must choose to run either --run_decisional or --run_optimization (or both).")
        parser.print_help()
        return
    
    if args.solver == "chuffed":
        if args.search_strategy != "base" and args.search_strategy != "ff_split" and args.search_strategy != "ro_Luby":
            print("Error: Chuffed solver only supports 'base', 'ff_split', and 'ro_Luby' search strategies.")
            parser.print_help()
            return 

    allowed_combinations = [("gecode", "modified_new_sb", "Y"),
                            ("gecode", "base_modified", "Y"),
                            ("gecode", "base", "Y"),
                            ("gecode", "base", "N"),
                            ("gecode", "dwd_random", "Y"),
                            ("gecode", "dwd_r_Luby", "Y"),
                            ("gecode", "dwd_r_rr", "Y"),
                            ("chuffed", "base_modified", "Y"),
                            ("chuffed", "base", "Y"),
                            ("chuffed", "base", "N"),
                            ("chuffed", "ff_split", "Y"),
                            ("chuffed", "ro_Luby", "Y"),
                            ("chuffed", "modified_new_sb", "Y")]
    if args.all:
        solving_combinations = allowed_combinations
    else:
        solving_combinations = [(solver, ss, sb) for ss in args.search_strategy 
                                for solver in args.solver
                                for sb in args.sb
                                if (solver, ss, sb) in allowed_combinations]

    timeout = args.timeout - 1

    if args.run_decisional:
        for n in args.n_teams:
            for solver, ss, sb in solving_combinations:
                    sb_name = "SB" if sb == "Y" else "no_SB"
                    # print(f"\n=== Decisional Solver | Solver: {solver} | Symmetry: {sb_name} | Search strategy: {ss} ===\n")
                    model_name = f"d_{solver}_{sb_name}_{ss}"
                    try: 
                        results = solve_cp_decisional(
                            n,
                            timeout=timeout,
                            symmetry_breaking=sb_name,
                            solver=solver,
                            search_strategy=ss,
                        )
                    except ValueError as e:
                        print(f"Skipping n={n}: {e}")
                        continue
                    if args.save_json:
                        save_results_as_json(n, model_name=model_name, results=results)
                    if results['sol']:
                        print(f"\n[Decisional Result] n={n} | time={results['time']}")
                        # human_readable_schedule(results['sol'])
                    else:
                        print(f"[!] No solution found for n={n}")

    if args.run_optimization:
        for n in args.n_teams:
            for solver, ss, sb in solving_combinations:
                    sb_name = "SB" if sb == "Y" else "no_SB"
                    # print(f"\n=== Optimization Solver | Solver: {solver} | Symmetry: {sb_name} | Search strategy: {ss} ===\n")
                    model_name = f"o_{solver}_{sb_name}_{ss}"
                    try:
                        results = solve_cp_optimization(
                            n,
                            timeout=timeout,
                            symmetry_breaking=sb_name,
                            solver=solver,
                            search_strategy=ss,
                        )
                    except ValueError as e:
                        print(f"Skipping n={n}: {e}")
                        continue
                    if args.save_json:
                        save_results_as_json(n, model_name=model_name, results=results)
                    if results['sol']:
                        print(f"\n[Optimization Result] n={n} | obj={results['obj']} | time={results['time']}")
                        # human_readable_schedule(results['sol'])
                    else:
                        print(f"[!] No solution found for n={n}")

if __name__ == "__main__":
    main()