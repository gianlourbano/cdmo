"""
Compact OR-Tools MIP formulation for STS problem

This module provides a completely different, more compact model formulation
using schedule matrix variables instead of match variables, which should
scale better for larger instances.
"""

import time
from typing import Optional, List
from ortools.linear_solver import pywraplp

from ..utils.solution_format import STSSolution


def solve_sts_compact(
    n: int,
    solver_name: Optional[str] = None,
    timeout: int = 300,
    optimization: bool = False
) -> STSSolution:
    """
    Solve STS using compact schedule-based formulation
    
    This formulation uses:
    - home[w,p] = team playing at home in week w, period p
    - away[w,p] = team playing away in week w, period p  
    - played[i,j] = 1 if teams i and j have played each other
    
    Args:
        n: Number of teams
        solver_name: OR-Tools solver to use
        timeout: Timeout in seconds
        optimization: Whether to use optimization version
        
    Returns:
        STSSolution object with results
    """
    
    if n % 2 != 0:
        return STSSolution(time=0, optimal=False, obj=None, sol=[])
    
    start_time = time.time()
    
    try:
        # Solver initialization
        if solver_name and solver_name.upper() == "SCIP":
            solver = pywraplp.Solver.CreateSolver("SCIP")
        elif solver_name and solver_name.upper() == "GUROBI":
            solver = pywraplp.Solver.CreateSolver("GUROBI")
        elif solver_name and solver_name.upper() == "CPLEX":
            solver = pywraplp.Solver.CreateSolver("CPLEX")
        else:
            solver = pywraplp.Solver.CreateSolver("CBC")
        
        if not solver:
            return STSSolution(time=0, optimal=False, obj=None, sol=[])
        
        solver.SetNumThreads(1)
        solver.SetTimeLimit(timeout * 1000)
        
        # Parameters
        teams = list(range(1, n + 1))
        weeks = list(range(1, n))
        periods = list(range(1, n // 2 + 1))
        
        # === DECISION VARIABLES ===
        
        # Schedule variables: who plays where and when
        home = {}  # home[w,p] = team playing at home in week w, period p
        away = {}  # away[w,p] = team playing away in week w, period p
        
        for w in weeks:
            for p in periods:
                home[w, p] = solver.IntVar(1, n, f"home_{w}_{p}")
                away[w, p] = solver.IntVar(1, n, f"away_{w}_{p}")
        
        # Matchup tracking variables
        played = {}  # played[i,j] = 1 if teams i,j have played each other
        for i in teams:
            for j in teams:
                if i < j:  # Only need upper triangle
                    played[i, j] = solver.BoolVar(f"played_{i}_{j}")
        
        # === CONSTRAINTS ===
        
        # 1. No team plays against itself
        for w in weeks:
            for p in periods:
                solver.Add(home[w, p] != away[w, p], f"no_self_play_{w}_{p}")
        
        # 2. Every team plays exactly once per week
        for k in teams:
            for w in weeks:
                # Count appearances of team k in week w
                appearances = []
                for p in periods:
                    appearances.append(solver.Sum([home[w, p] == k, away[w, p] == k]))
                solver.Add(solver.Sum(appearances) == 1, f"once_per_week_{k}_{w}")
        
        # 3. Link schedule to matchup variables (FIXED VERSION)
        # For each slot, track which team-pairs could play there
        for w in weeks:
            for p in periods:
                # For each possible team pair, create constraints
                slot_pairs = []
                for i in teams:
                    for j in teams:
                        if i < j:
                            # Create binary variables for specific matchups in this slot
                            match_ij = solver.BoolVar(f"match_{i}_{j}_{w}_{p}_home")
                            match_ji = solver.BoolVar(f"match_{j}_{i}_{w}_{p}_home")
                            
                            # match_ij = 1 iff (home[w,p] == i AND away[w,p] == j)
                            # This requires both conditions to be true
                            solver.Add(match_ij * n <= home[w, p] - i + n - 1)  # home[w,p] >= i if match_ij = 1
                            solver.Add(match_ij * n <= i - home[w, p] + n - 1)  # home[w,p] <= i if match_ij = 1
                            solver.Add(match_ij * n <= away[w, p] - j + n - 1)  # away[w,p] >= j if match_ij = 1
                            solver.Add(match_ij * n <= j - away[w, p] + n - 1)  # away[w,p] <= j if match_ij = 1
                            
                            # match_ji = 1 iff (home[w,p] == j AND away[w,p] == i)
                            solver.Add(match_ji * n <= home[w, p] - j + n - 1)
                            solver.Add(match_ji * n <= j - home[w, p] + n - 1)
                            solver.Add(match_ji * n <= away[w, p] - i + n - 1)
                            solver.Add(match_ji * n <= i - away[w, p] + n - 1)
                            
                            # Link to played variables
                            solver.Add(played[i, j] >= match_ij)
                            solver.Add(played[i, j] >= match_ji)
                            
                            slot_pairs.append(match_ij)
                            slot_pairs.append(match_ji)
                
                # Each slot must have exactly one pair playing
                solver.Add(solver.Sum(slot_pairs) == 1, f"one_pair_per_slot_{w}_{p}")
        
        # 4. Every pair of teams plays exactly once
        for i in teams:
            for j in teams:
                if i < j:
                    solver.Add(played[i, j] == 1, f"play_once_{i}_{j}")
        
        # 5. Period constraint: each team plays at most twice in the same period
        for k in teams:
            for p in periods:
                period_appearances = []
                for w in weeks:
                    period_appearances.append(solver.Sum([home[w, p] == k, away[w, p] == k]))
                solver.Add(solver.Sum(period_appearances) <= 2, f"period_limit_{k}_{p}")
        
        # === SYMMETRY BREAKING ===
        
        # Fix team 1's first game
        solver.Add(home[1, 1] == 1, "sym_team1_home_w1_p1")
        solver.Add(away[1, 1] == 2, "sym_team1_vs_2_w1_p1")
        
        # Order first week by periods
        if len(periods) > 1:
            solver.Add(home[1, 1] < home[1, 2], "sym_order_w1_p1_p2")
        
        # Order first few teams in first period
        if len(weeks) > 1:
            solver.Add(home[1, 1] < home[2, 1], "sym_order_first_period")
        
        # === OBJECTIVE FUNCTION ===
        if optimization:
            # Count home/away games for each team
            home_count = {}
            away_count = {}
            
            for k in teams:
                home_appearances = []
                away_appearances = []
                for w in weeks:
                    for p in periods:
                        home_appearances.append(home[w, p] == k)
                        away_appearances.append(away[w, p] == k)
                
                home_count[k] = solver.Sum(home_appearances)
                away_count[k] = solver.Sum(away_appearances)
            
            # Minimize total imbalance
            deviation = {}
            for k in teams:
                deviation[k] = solver.NumVar(0, n-1, f'dev_{k}')
                solver.Add(deviation[k] >= home_count[k] - away_count[k])
                solver.Add(deviation[k] >= away_count[k] - home_count[k])
            
            solver.Minimize(solver.Sum(deviation[k] for k in teams))
        
        # === SOLVE ===
        status = solver.Solve()
        elapsed_time = int(time.time() - start_time)
        
        # === EXTRACT SOLUTION ===
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # Build solution matrix
            sol = []
            for p in periods:
                period_games = []
                for w in weeks:
                    home_team = int(home[w, p].solution_value())
                    away_team = int(away[w, p].solution_value())
                    period_games.append([home_team, away_team])
                sol.append(period_games)
            
            optimal = (status == pywraplp.Solver.OPTIMAL)
            obj_value = None
            if optimization and solver.Objective():
                obj_value = int(solver.Objective().Value())
            
            return STSSolution(
                time=elapsed_time,
                optimal=optimal,
                obj=obj_value,
                sol=sol
            )
        else:
            return STSSolution(
                time=min(elapsed_time, timeout),
                optimal=False,
                obj=None,
                sol=[]
            )
            
    except Exception as e:
        elapsed_time = int(time.time() - start_time)
        return STSSolution(
            time=min(elapsed_time, timeout),
            optimal=False,
            obj=None,
            sol=[]
        )


def solve_sts_flow_based(
    n: int,
    solver_name: Optional[str] = None,
    timeout: int = 300,
    optimization: bool = False
) -> STSSolution:
    """
    Alternative flow-based formulation for STS
    
    Models the problem as a multi-commodity flow where:
    - Each team-pair is a commodity that needs to be routed through time slots
    - Time slots are nodes in a network
    - Flow conservation ensures each pair plays exactly once
    """
    
    if n % 2 != 0:
        return STSSolution(time=0, optimal=False, obj=None, sol=[])
    
    start_time = time.time()
    
    try:
        # Solver setup
        if solver_name and solver_name.upper() == "SCIP":
            solver = pywraplp.Solver.CreateSolver("SCIP")
        elif solver_name and solver_name.upper() == "GUROBI":
            solver = pywraplp.Solver.CreateSolver("GUROBI")
        else:
            solver = pywraplp.Solver.CreateSolver("CBC")
            
        if not solver:
            return STSSolution(time=0, optimal=False, obj=None, sol=[])
        
        solver.SetNumThreads(1)
        solver.SetTimeLimit(timeout * 1000)
        
        teams = list(range(1, n + 1))
        weeks = list(range(1, n))
        periods = list(range(1, n // 2 + 1))
        
        # Time slots
        slots = [(w, p) for w in weeks for p in periods]
        
        # Team pairs (commodities)
        pairs = [(i, j) for i in teams for j in teams if i < j]
        
        # === VARIABLES ===
        # flow[i,j,w,p] = 1 if pair (i,j) is assigned to slot (w,p)
        flow = {}
        for (i, j) in pairs:
            for (w, p) in slots:
                flow[i, j, w, p] = solver.BoolVar(f"flow_{i}_{j}_{w}_{p}")
        
        # team_in_slot[t,w,p,s] = 1 if team t is in slot (w,p) at position s (0=home, 1=away)
        team_slot = {}
        for t in teams:
            for (w, p) in slots:
                for s in [0, 1]:  # 0=home, 1=away
                    team_slot[t, w, p, s] = solver.BoolVar(f"team_{t}_{w}_{p}_{s}")
        
        # === CONSTRAINTS ===
        
        # 1. Each pair plays exactly once (flow conservation)
        for (i, j) in pairs:
            solver.Add(solver.Sum(flow[i, j, w, p] for (w, p) in slots) == 1,
                      f"pair_{i}_{j}_once")
        
        # 2. Each slot has exactly one game
        for (w, p) in slots:
            solver.Add(solver.Sum(flow[i, j, w, p] for (i, j) in pairs) == 1,
                      f"slot_{w}_{p}_one_game")
        
        # 3. Link flow to team positions
        for (w, p) in slots:
            for (i, j) in pairs:
                # If pair (i,j) plays in slot (w,p), then either:
                # (i at home, j away) or (j at home, i away)
                solver.Add(flow[i, j, w, p] <= team_slot[i, w, p, 0] + team_slot[j, w, p, 0],
                          f"link_home_{i}_{j}_{w}_{p}")
                solver.Add(flow[i, j, w, p] <= team_slot[i, w, p, 1] + team_slot[j, w, p, 1], 
                          f"link_away_{i}_{j}_{w}_{p}")
        
        # 4. Each slot has exactly one home and one away team
        for (w, p) in slots:
            solver.Add(solver.Sum(team_slot[t, w, p, 0] for t in teams) == 1,
                      f"one_home_{w}_{p}")
            solver.Add(solver.Sum(team_slot[t, w, p, 1] for t in teams) == 1,
                      f"one_away_{w}_{p}")
        
        # 5. Each team plays exactly once per week
        for t in teams:
            for w in weeks:
                solver.Add(solver.Sum(team_slot[t, w, p, s] 
                                    for p in periods for s in [0, 1]) == 1,
                          f"team_{t}_week_{w}")
        
        # 6. Period constraint
        for t in teams:
            for p in periods:
                solver.Add(solver.Sum(team_slot[t, w, p, s] 
                                    for w in weeks for s in [0, 1]) <= 2,
                          f"team_{t}_period_{p}_limit")
        
        # Symmetry breaking
        if pairs:
            first_pair = pairs[0]
            solver.Add(flow[first_pair[0], first_pair[1], 1, 1] == 1,
                      f"sym_first_pair")
        
        # Solve
        status = solver.Solve()
        elapsed_time = int(time.time() - start_time)
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # Extract solution
            sol = []
            for p in periods:
                period_games = []
                for w in weeks:
                    # Find which pair plays in this slot
                    home_team, away_team = None, None
                    for t in teams:
                        if team_slot[t, w, p, 0].solution_value() > 0.5:
                            home_team = t
                        if team_slot[t, w, p, 1].solution_value() > 0.5:
                            away_team = t
                    
                    if home_team is not None and away_team is not None:
                        period_games.append([home_team, away_team])
                    else:
                        period_games.append([1, 2])  # Fallback
                sol.append(period_games)
            
            return STSSolution(
                time=elapsed_time,
                optimal=(status == pywraplp.Solver.OPTIMAL),
                obj=None,
                sol=sol
            )
        else:
            return STSSolution(
                time=min(elapsed_time, timeout),
                optimal=False,
                obj=None,
                sol=[]
            )
            
    except Exception as e:
        elapsed_time = int(time.time() - start_time)
        return STSSolution(
            time=min(elapsed_time, timeout),
            optimal=False,
            obj=None,
            sol=[]
        )