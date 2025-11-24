"""
Solution format utilities for STS problem

Provides standardized solution representation and JSON serialization
as specified in the project documentation.
"""

from typing import List, Optional, Dict, Any, Union
import json
from pathlib import Path


class CompactJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that keeps matrix rows on single lines"""
    
    def encode(self, obj):
        if isinstance(obj, dict):
            # Handle the top-level dictionary
            items = []
            for key, value in obj.items():
                encoded_value = self._encode_solution_dict(value)
                items.append(f'  "{key}": {encoded_value}')
            return "{\n" + ",\n".join(items) + "\n}"
        return super().encode(obj)
    
    def _encode_solution_dict(self, obj):
        """Encode a solution dictionary with compact sol format"""
        if not isinstance(obj, dict):
            return json.dumps(obj)
        
        parts = []
        parts.append("{\n")
        
        for key, value in obj.items():
            if key == "sol" and isinstance(value, list):
                # Special handling for sol: each period on one line
                parts.append(f'    "{key}": [\n')
                period_lines = []
                for period in value:
                    # Each period (list of games) on one line
                    period_lines.append(f'      {json.dumps(period)}')
                parts.append(",\n".join(period_lines))
                parts.append("\n    ]")
            else:
                # Regular fields
                parts.append(f'    "{key}": {json.dumps(value)}')
            
            # Add comma if not last item
            if key != list(obj.keys())[-1]:
                parts.append(",\n")
            else:
                parts.append("\n")
        
        parts.append("  }")
        return "".join(parts)
    
    def iterencode(self, obj, _one_shot=False):
        """Override to use our custom encoding"""
        yield self.encode(obj)


class STSSolution:
    """Represents a solution to the STS problem"""
    
    def __init__(
        self, 
        time: int,
        optimal: bool, 
        obj: Optional[int],
        sol: List[List[List[int]]]
    ):
        self.time = time
        self.optimal = optimal
        self.obj = obj
        self.sol = sol
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert solution to dictionary format"""
        return {
            "time": self.time,
            "optimal": self.optimal,
            "obj": self.obj,
            "sol": self.sol
        }


def save_results(
    instance_id: int,
    approach: str,
    results: Dict[str, STSSolution],
    output_dir: Path
) -> None:
    """
    Save results in the required JSON format, merging with existing approaches
    
    Args:
        instance_id: Instance number
        approach: Optimization approach (CP, SAT, SMT, MIP)
        results: Dictionary mapping solver names to solutions
        output_dir: Output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{instance_id}.json"
    
    # Load existing data if file exists
    existing_data = {}
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing_data = {}
    
    # Merge new results with existing data
    for solver_name, solution in results.items():
        existing_data[solver_name] = solution.to_dict()
    
    # Write back the merged data with compact sol format
    with open(output_file, 'w') as f:
        f.write(CompactJSONEncoder().encode(existing_data))


def validate_solution(n: int, sol: List[List[List[int]]]) -> bool:
    """
    Validate solution format and basic constraints
    
    Args:
        n: Number of teams
        sol: Solution matrix
        
    Returns:
        True if solution format is valid
    """
    if not isinstance(sol, list):
        return False
    
    # Check dimensions: n/2 periods × (n-1) weeks
    if len(sol) != n // 2:
        return False
    
    for period in sol:
        if not isinstance(period, list) or len(period) != n - 1:
            return False
        for week in period:
            if not isinstance(week, list) or len(week) != 2:
                return False
            if not all(isinstance(team, int) and 1 <= team <= n for team in week):
                return False
    
    return True