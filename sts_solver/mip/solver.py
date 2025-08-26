"""
Mixed-Integer Programming solver for STS using PuLP and OR-Tools
"""

import time
import pulp as pl
from typing import Optional

from ..utils.solution_format import STSSolution
from .ortools_solver import solve_sts_ortools, get_available_ortools_solvers
from .ortools_optimized import solve_sts_ortools_optimized
from .ortools_compact import solve_sts_compact, solve_sts_flow_based
from .ortools_fixed import solve_sts_fixed_compact
from .ortools_match_based import solve_sts_match_based
from .ortools_simple_match import solve_sts_simple_match


def solve_mip(
    n: int, 
    solver_name: Optional[str] = None, 
    timeout: int = 300,
    optimization: bool = False
) -> STSSolution:
    """Solve STS using MIP with multiple formulation options"""
    
    # Parse solver name to extract formulation and solver
    formulation = "standard"
    actual_solver = solver_name
    
    if solver_name and "-" in solver_name:
        parts = solver_name.split("-", 1)
        formulation = parts[0].lower()
        actual_solver = parts[1].upper()
    
    # Choose formulation
    if formulation == "optimized":
        return solve_sts_ortools_optimized(n, actual_solver, timeout, optimization)
    elif formulation == "compact":
        return solve_sts_fixed_compact(n, actual_solver, timeout, optimization)  # Use fixed version
    elif formulation == "match":
        return solve_sts_simple_match(n, actual_solver, timeout, optimization)  # Simplified match-based
    elif formulation == "flow":
        return solve_sts_flow_based(n, actual_solver, timeout, optimization)
    elif formulation == "pulp":
        return solve_mip_pulp(n, actual_solver.lower(), timeout, optimization)
    else:
        # Standard formulation
        ortools_solvers = ["CBC", "SCIP", "GUROBI", "CPLEX"]
        if actual_solver and actual_solver.upper() in ortools_solvers:
            return solve_sts_ortools(n, actual_solver, timeout, optimization)
        
        # Use OR-Tools by default
        if solver_name is None:
            available_ortools = get_available_ortools_solvers()
            if available_ortools:
                # Use optimized formulation for larger instances
                if n >= 12:
                    preferred_solver = "SCIP" if "SCIP" in available_ortools else available_ortools[0]
                    return solve_sts_ortools_optimized(n, preferred_solver, timeout, optimization)
                else:
                    # Standard formulation for smaller instances
                    preferred_solver = "SCIP" if "SCIP" in available_ortools else available_ortools[0]
                    return solve_sts_ortools(n, preferred_solver, timeout, optimization)
        
        # Fallback to PuLP
        return solve_mip_pulp(n, solver_name, timeout, optimization)


def solve_mip_pulp(
    n: int, 
    solver_name: Optional[str] = None, 
    timeout: int = 300,
    optimization: bool = False
) -> STSSolution:
    """Solve STS using PuLP"""
    
    weeks = n - 1
    periods = n // 2
    
    start_time = time.time()
    
    try:
        # Create the problem
        if optimization:
            prob = pl.LpProblem("STS_Optimization", pl.LpMinimize)
        else:
            prob = pl.LpProblem("STS_Decision", pl.LpMaximize)
        
        # Variables
        x = {}
        for w in range(weeks):
            for p in range(periods):
                for s in range(2):  # 0=home, 1=away
                    for t in range(n):
                        x[w, p, s, t] = pl.LpVariable(f"x_{w}_{p}_{s}_{t}", cat='Binary')
        
        # One team per slot
        for w in range(weeks):
            for p in range(periods):
                for s in range(2):
                    prob += pl.lpSum([x[w, p, s, t] for t in range(n)]) == 1
        
        # One game per team per week
        for w in range(weeks):
            for t in range(n):
                prob += pl.lpSum([x[w, p, s, t] 
                                for p in range(periods) for s in range(2)]) == 1
        
        # Every pair plays once
        for t1 in range(n):
            for t2 in range(t1 + 1, n):
                games = []
                for w in range(weeks):
                    for p in range(periods):
                        # Home/away combinations
                        games.append(x[w, p, 0, t1] * x[w, p, 1, t2])
                        games.append(x[w, p, 1, t1] * x[w, p, 0, t2])
                
                # Linearization with auxiliary variables
                for w in range(weeks):
                    for p in range(periods):
                        # Create auxiliary variables for games
                        game1 = pl.LpVariable(f"game_{t1}_{t2}_{w}_{p}_0", cat='Binary')
                        game2 = pl.LpVariable(f"game_{t1}_{t2}_{w}_{p}_1", cat='Binary')
                        
                        # Link variables
                        prob += game1 <= x[w, p, 0, t1]
                        prob += game1 <= x[w, p, 1, t2]
                        prob += game1 >= x[w, p, 0, t1] + x[w, p, 1, t2] - 1
                        
                        prob += game2 <= x[w, p, 1, t1]
                        prob += game2 <= x[w, p, 0, t2]
                        prob += game2 >= x[w, p, 1, t1] + x[w, p, 0, t2] - 1
                
                # One game per pair
                games_aux = []
                for w in range(weeks):
                    for p in range(periods):
                        games_aux.extend([
                            pl.LpVariable(f"game_{t1}_{t2}_{w}_{p}_0", cat='Binary'),
                            pl.LpVariable(f"game_{t1}_{t2}_{w}_{p}_1", cat='Binary')
                        ])
                prob += pl.lpSum(games_aux) == 1
        
        # At most twice per period
        for p in range(periods):
            for t in range(n):
                prob += pl.lpSum([x[w, p, s, t] 
                                for w in range(weeks) for s in range(2)]) <= 2
        
        # Objective
        if optimization:
            # Minimize home/away imbalance
            home_games = {}
            away_games = {}
            for t in range(n):
                home_games[t] = pl.lpSum([x[w, p, 0, t] 
                                        for w in range(weeks) for p in range(periods)])
                away_games[t] = pl.lpSum([x[w, p, 1, t] 
                                        for w in range(weeks) for p in range(periods)])
            
            # Minimize maximum imbalance
            imbalance = pl.LpVariable("imbalance", lowBound=0)
            for t in range(n):
                prob += imbalance >= home_games[t] - away_games[t]
                prob += imbalance >= away_games[t] - home_games[t]
            
            prob += imbalance
        else:
            # Dummy objective
            prob += 0
        
        # Select solver
        if solver_name:
            if solver_name.lower() == "gurobi":
                solver = pl.GUROBI(timeLimit=timeout)
            elif solver_name.lower() == "cplex":
                solver = pl.CPLEX(timeLimit=timeout)
            else:
                solver = pl.PULP_CBC_CMD(timeLimit=timeout)
        else:
            solver = pl.PULP_CBC_CMD(timeLimit=timeout)
        
        # Solve
        prob.solve(solver)
        
        elapsed_time = int(time.time() - start_time)
        
        if prob.status == pl.LpStatusOptimal or prob.status == pl.LpStatusFeasible:
            # Extract solution
            sol = []
            for p in range(periods):
                period_games = []
                for w in range(weeks):
                    # Find teams playing
                    home_team = None
                    away_team = None
                    for t in range(n):
                        if x[w, p, 0, t].varValue and x[w, p, 0, t].varValue > 0.5:
                            home_team = t + 1
                        if x[w, p, 1, t].varValue and x[w, p, 1, t].varValue > 0.5:
                            away_team = t + 1
                    
                    if home_team is not None and away_team is not None:
                        period_games.append([home_team, away_team])
                    else:
                        # Fallback
                        period_games.append([1, 2])
                        
                sol.append(period_games)
            
            optimal = (prob.status == pl.LpStatusOptimal)
            obj_value = None
            if optimization and prob.objective:
                obj_value = int(prob.objective.value()) if prob.objective.value() else None
            
            return STSSolution(
                time=elapsed_time,
                optimal=optimal,
                obj=obj_value,
                sol=sol
            )
        else:
            # No solution
            return STSSolution(
                time=elapsed_time if elapsed_time < timeout else timeout,
                optimal=False,
                obj=None,
                sol=[]
            )
            
    except Exception as e:
        elapsed_time = int(time.time() - start_time)
        if elapsed_time >= timeout:
            elapsed_time = timeout
            
        return STSSolution(
            time=elapsed_time,
            optimal=False,
            obj=None,
            sol=[]
        )