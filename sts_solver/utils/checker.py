"""
Solution checker interface for STS solutions

This module provides integration with the official solution_checker.py
provided by the course instructors.
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Import the official checker directly
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from solution_checker import check_solution
    CHECKER_AVAILABLE = True
except ImportError:
    CHECKER_AVAILABLE = False


def check_solution_direct(solution: List, obj, time: int, optimal: bool) -> Tuple[bool, str]:
    """
    Check solution directly using the official checker function
    
    Args:
        solution: Solution matrix
        obj: Objective value
        time: Runtime in seconds
        optimal: Whether solution is optimal
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not CHECKER_AVAILABLE:
        return False, "Official checker not available"
    
    try:
        result = check_solution(solution, obj, time, optimal)
        
        if isinstance(result, str) and result == "Valid solution":
            return True, "Valid solution"
        elif isinstance(result, list):
            return False, "; ".join(result)
        else:
            return False, str(result)
            
    except Exception as e:
        return False, f"Checker error: {e}"


def check_solution_file(
    solution_file: Path, 
    checker_path: Path = Path("solution_checker.py")
) -> Tuple[bool, str]:
    """
    Check solution file using the official checker
    
    Args:
        solution_file: Path to the JSON solution file
        checker_path: Path to the official solution_checker.py
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not solution_file.exists():
        return False, f"Solution file not found at {solution_file}"
    
    # First try direct import if available
    if CHECKER_AVAILABLE:
        try:
            with open(solution_file, 'r') as f:
                data = json.load(f)
            
            all_valid = True
            messages = []
            
            for approach, result in data.items():
                sol = result.get("sol", [])
                time = result.get("time", 300)
                opt = result.get("optimal", False)
                obj = result.get("obj")
                
                is_valid, message = check_solution_direct(sol, obj, time, opt)
                if not is_valid:
                    all_valid = False
                    messages.append(f"{approach}: {message}")
                else:
                    messages.append(f"{approach}: Valid")
            
            return all_valid, "; ".join(messages)
            
        except Exception as e:
            return False, f"Error checking solution: {e}"
    
    # Fallback to subprocess call
    if not checker_path.exists():
        return False, f"Checker not found at {checker_path}"
    
    try:
        # The official checker expects a directory, so we need to create a temp directory
        # or modify our approach. For now, we'll use subprocess with the parent directory
        parent_dir = solution_file.parent
        result = subprocess.run(
            ["python3", str(checker_path), str(parent_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            # Parse output to determine if our specific file is valid
            if "VALID" in result.stdout:
                return True, "Solution is valid"
            else:
                return False, result.stdout
        else:
            return False, result.stderr or result.stdout
            
    except subprocess.TimeoutExpired:
        return False, "Checker timed out"
    except Exception as e:
        return False, f"Error running checker: {e}"


def validate_all_results(
    results_dir: Path = Path("res"),
    checker_path: Path = Path("solution_checker.py")
) -> Dict[str, Dict[str, Tuple[bool, str]]]:
    """
    Validate all result files in the results directory
    
    Args:
        results_dir: Base results directory
        checker_path: Path to the official checker
        
    Returns:
        Dictionary mapping approach -> instance -> (is_valid, message)
    """
    validation_results = {}
    
    for approach_dir in results_dir.iterdir():
        if not approach_dir.is_dir():
            continue
            
        approach_name = approach_dir.name
        validation_results[approach_name] = {}
        
        for result_file in approach_dir.glob("*.json"):
            instance_name = result_file.stem
            is_valid, message = check_solution_file(result_file, checker_path)
            validation_results[approach_name][instance_name] = (is_valid, message)
    
    return validation_results


def validate_directory_with_official_checker(
    results_dir: Path = Path("res"),
    checker_path: Path = Path("solution_checker.py")
) -> None:
    """
    Validate all results using the official checker's directory-based approach
    
    Args:
        results_dir: Base results directory  
        checker_path: Path to the official solution_checker.py
    """
    if not checker_path.exists():
        print(f"Error: Checker not found at {checker_path}")
        return
    
    for approach_dir in results_dir.iterdir():
        if not approach_dir.is_dir() or not list(approach_dir.glob("*.json")):
            continue
            
        print(f"\nValidating {approach_dir.name} approach:")
        print("=" * 50)
        
        try:
            result = subprocess.run(
                ["python3", str(checker_path), str(approach_dir)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"Error validating {approach_dir.name}:")
                print(result.stderr or result.stdout)
                
        except subprocess.TimeoutExpired:
            print(f"Validation timed out for {approach_dir.name}")
        except Exception as e:
            print(f"Error running checker on {approach_dir.name}: {e}")


def print_validation_report(validation_results: Dict[str, Dict[str, Tuple[bool, str]]]):
    """Print a formatted validation report"""
    
    print("=" * 60)
    print("SOLUTION VALIDATION REPORT")
    print("=" * 60)
    
    total_valid = 0
    total_invalid = 0
    
    for approach, instances in validation_results.items():
        print(f"\n{approach} Approach:")
        print("-" * 20)
        
        for instance, (is_valid, message) in instances.items():
            status = "✓ VALID" if is_valid else "✗ INVALID"
            print(f"  Instance {instance}: {status}")
            if not is_valid:
                print(f"    Error: {message}")
                total_invalid += 1
            else:
                total_valid += 1
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {total_valid} valid, {total_invalid} invalid")
    print("=" * 60)