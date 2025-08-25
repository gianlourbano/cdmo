"""
Standalone OR-Tools MIP solver for STS problem
Based on the user's improved implementation

This file provides a standalone version that matches the user's original
implementation for testing and comparison purposes.
"""

from ortools.linear_solver import pywraplp
import argparse
import json
import time


def solve_sts(n, optimize=False, solver_name='CBC', timeout=300):
    """
    Builds and solves the Sports Tournament Scheduling MIP model using Google OR-Tools.

    Args:
        n (int): The number of teams. Must be an even number.
        optimize (bool): If True, enables the optimization objective to balance
                         home/away games. If False, solves the decision problem.
        solver_name (str): The MIP solver backend to use ('SCIP', 'CBC', etc.).
                           CBC is generally very strong and is bundled with OR-Tools.
        timeout (int): Timeout in seconds.
        
    Returns:
        Dictionary with solution information in project format
    """
    if n % 2 != 0:
        print("Error: The number of teams (n) must be even.")
        return None

    start_time = time.time()

    # 1. --- Solver Initialization ---
    # Create the MIP solver with the specified backend.
    solver = pywraplp.Solver.CreateSolver(solver_name)
    if not solver:
        print(f"Solver {solver_name} not available.")
        return None
    
    solver.SetNumThreads(1)  # Limit to one thread for consistency
    solver.SetTimeLimit(timeout * 1000)  # Time limit in milliseconds

    # 2. --- Parameters ---
    teams = list(range(1, n + 1))
    weeks = list(range(1, n))
    periods = list(range(1, n // 2 + 1))

    # 3. --- Decision Variables ---
    # In OR-Tools, variables are typically created individually and stored in a dictionary.
    # match_vars[i, j, w, p] is a binary variable.
    match_vars = {}
    for i in teams:
        for j in teams:
            if i == j: continue
            for w in weeks:
                for p in periods:
                    match_vars[i, j, w, p] = solver.BoolVar(f'match_{i}_{j}_{w}_{p}')

    # 4. --- Constraints ---
    
    # Constraint 1: Every team plays every other team exactly once.
    for i in teams:
        for j in teams:
            if i < j:
                solver.Add(
                    solver.Sum(match_vars[i, j, w, p] + match_vars[j, i, w, p] 
                               for w in weeks for p in periods) == 1,
                    f"meet_once_{i}_{j}"
                )

    # Constraint 2: Every team plays exactly once per week.
    for k in teams:
        for w in weeks:
            solver.Add(
                solver.Sum(match_vars[k, j, w, p] for j in teams if k != j for p in periods) +
                solver.Sum(match_vars[i, k, w, p] for i in teams if i != k for p in periods) == 1,
                f"one_game_per_week_{k}_{w}"
            )

    # Constraint 3: Each slot (week, period) has exactly one game.
    for w in weeks:
        for p in periods:
            solver.Add(
                solver.Sum(match_vars[i, j, w, p] for i in teams for j in teams if i != j) == 1,
                f"one_game_per_slot_{w}_{p}"
            )

    # Constraint 4: Every team plays at most twice in the same period over the tournament.
    for k in teams:
        for p in periods:
            solver.Add(
                solver.Sum(match_vars[k, j, w, p] for j in teams if k != j for w in weeks) +
                solver.Sum(match_vars[i, k, w, p] for i in teams if i != k for w in weeks) <= 2,
                f"period_limit_{k}_{p}"
            )

    # 5. --- Conditional Objective Function ---
    if optimize:
        print(f"\nRunning in OPTIMIZATION mode with {solver_name} solver...")
        home_games = {k: solver.Sum(match_vars[k, j, w, p] for j in teams if k!=j for w in weeks for p in periods) for k in teams}
        away_games = {k: solver.Sum(match_vars[i, k, w, p] for i in teams if i!=k for w in weeks for p in periods) for k in teams}
        
        # In OR-Tools, continuous variables are created with NumVar.
        deviation = {k: solver.NumVar(0, solver.infinity(), f'dev_{k}') for k in teams}

        for k in teams:
            solver.Add(deviation[k] >= home_games[k] - away_games[k])
            solver.Add(deviation[k] >= away_games[k] - home_games[k])

        # Set the objective using solver.Minimize()
        solver.Minimize(solver.Sum(deviation[k] for k in teams))
    else:
        print(f"\nRunning in DECISION mode with {solver_name} solver...")
        # No objective is set.

    # 6. --- Solve ---
    status = solver.Solve()
    elapsed_time = int(time.time() - start_time)

    # 7. --- Process Results ---
    print("-" * 50)
    # For OR-Tools, a time-limited run that finds a solution returns FEASIBLE, not OPTIMAL
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print("Solution found!")
        if status == pywraplp.Solver.FEASIBLE and optimize:
             print("Warning: Solver finished, but the solution may not be optimal (e.g., due to time limit).")

        if optimize:
            # Objective value is retrieved with Objective().Value()
            print(f"Total Imbalance (Objective Value): {solver.Objective().Value():.2f}")
        print("-" * 50)

        schedule = [[None for _ in weeks] for _ in periods]
        
        for (i, j, w, p), var in match_vars.items():
            # Solution values are retrieved with solution_value()
            if var.solution_value() > 0.5:
                schedule[p-1][w-1] = f"{i} vs {j}"

        # Display schedule
        header = f"{'Period':<8}" + "".join([f"{'Week '+str(w):<12}" for w in weeks])
        print(header)
        print("-" * len(header))

        for p_idx, period_row in enumerate(schedule):
            row_str = f"{p_idx + 1:<8}"
            for game in period_row:
                row_str += f"{game if game else '---':<12}"
            print(row_str)

        print("-" * 50)
        print("Home/Away Game Counts:")
        # Re-calculate home/away games using their solution values for printing
        for k in teams:
            h_count = sum(match_vars[k, j, w, p].solution_value() for j in teams if k!=j for w in weeks for p in periods)
            a_count = sum(match_vars[i, k, w, p].solution_value() for i in teams if i!=k for w in weeks for p in periods)
            print(f"  Team {k}: {h_count:.0f} Home, {a_count:.0f} Away")

        # Convert to project JSON format
        sol = []
        for p_idx in range(len(periods)):
            period_games = []
            for w_idx in range(len(weeks)):
                # Find the game in this period and week
                game_found = False
                for (i, j, w, p), var in match_vars.items():
                    if w == weeks[w_idx] and p == periods[p_idx] and var.solution_value() > 0.5:
                        period_games.append([i, j])
                        game_found = True
                        break
                if not game_found:
                    period_games.append([1, 2])  # Fallback
            sol.append(period_games)

        return {
            "time": elapsed_time,
            "optimal": (status == pywraplp.Solver.OPTIMAL),
            "obj": int(solver.Objective().Value()) if optimize and solver.Objective() else None,
            "sol": sol
        }

    elif status == pywraplp.Solver.INFEASIBLE:
        print("Model is infeasible. No solution exists.")
    elif status == pywraplp.Solver.MODEL_INVALID:
        print("Model is invalid.")
    else:
        print(f"Solver stopped with status: {status}. No solution found.")
    
    print("-" * 50)
    
    return {
        "time": elapsed_time if elapsed_time < timeout else timeout,
        "optimal": False,
        "obj": None,
        "sol": []
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the Sports Tournament Scheduling problem with Google OR-Tools.")
    parser.add_argument("n", type=int, help="The number of teams (must be an even number).")
    parser.add_argument("--optimize", action="store_true", help="Enable the optimization objective to balance home/away games.")
    parser.add_argument("--solver", type=str, default="CBC", help="Specify the solver backend (e.g., 'SCIP', 'CBC').")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds.")
    parser.add_argument("--output", type=str, help="Output JSON file path.")

    args = parser.parse_args()
    result = solve_sts(args.n, args.optimize, args.solver, args.timeout)
    
    if result and args.output:
        output_data = {args.solver.lower(): result}
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")