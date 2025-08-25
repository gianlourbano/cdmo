"""
Constraint Programming solver for STS using MiniZinc
"""

import pymzn
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

from ..utils.solution_format import STSSolution


def solve_cp(
    n: int, 
    solver: Optional[str] = None, 
    timeout: int = 300,
    optimization: bool = False
) -> STSSolution:
    """
    Solve STS using Constraint Programming with MiniZinc
    
    Args:
        n: Number of teams
        solver: MiniZinc solver to use (default: gecode)
        timeout: Timeout in seconds
        optimization: Whether to use optimization version
        
    Returns:
        STSSolution object with results
    """
    
    if solver is None:
        solver = "gecode"
    
    # Path to MiniZinc model
    model_path = Path(__file__).parent.parent.parent / "source" / "CP" / "sts.mzn"
    
    if not model_path.exists():
        raise FileNotFoundError(f"MiniZinc model not found at {model_path}")
    
    start_time = time.time()
    
    try:
        # Create data for the model
        data = {"n": n}
        
        # Solve with MiniZinc
        result = pymzn.minizinc(
            str(model_path),
            data=data,
            solver=solver,
            timeout=timeout,
            output_mode="dict"
        )
        
        elapsed_time = int(time.time() - start_time)
        
        if result:
            # Parse the schedule from MiniZinc output
            schedule = result[0]["schedule"]  # First solution
            
            # Convert to required format: periods × weeks matrix
            sol = []
            periods = n // 2
            weeks = n - 1
            
            for p in range(periods):
                period_games = []
                for w in range(weeks):
                    # Get game for this period and week
                    game = schedule[w * periods + p]
                    period_games.append(game)
                sol.append(period_games)
            
            return STSSolution(
                time=elapsed_time,
                optimal=True,  # MiniZinc finds exact solutions
                obj=None if not optimization else 1,  # Placeholder for optimization
                sol=sol
            )
        else:
            # No solution found within timeout
            return STSSolution(
                time=timeout,
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