"""Validate commands (modular CLI)"""

from pathlib import Path
from typing import Optional
import json
import click


@click.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--checker", "-c", type=click.Path(), help="Path to solution_checker.py")
def validate(results_file: str, checker: Optional[str]):
    """Validate solution file format and correctness."""
    from ...utils.checker import check_solution_file

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)

        for solver_name, solution in data.items():
            for field in ("time", "optimal", "obj", "sol"):
                if field not in solution:
                    click.echo(f"Error: Missing field '{field}' in solver '{solver_name}'", err=True)
                    raise SystemExit(1)

        click.echo("Solution file format is valid")

        checker_path = Path(checker) if checker else Path("solution_checker.py")
        is_valid, message = check_solution_file(Path(results_file), checker_path)
        if is_valid:
            click.echo("Solution correctness validated")
        else:
            click.echo(f"Solution validation failed: {message}", err=True)

    except json.JSONDecodeError as e:
        click.echo(f"Error: Invalid JSON format: {e}", err=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@click.command(name="validate-all")
@click.option("--checker", "-c", type=click.Path(), help="Path to solution_checker.py")
@click.option("--results-dir", "-r", type=click.Path(), default="res", help="Results directory")
@click.option("--official", is_flag=True, help="Use official checker's directory-based validation")
def validate_all(checker: Optional[str], results_dir: str, official: bool):
    """Validate all solution files in a directory."""

    checker_path = Path(checker) if checker else Path("solution_checker.py")
    results_path = Path(results_dir)

    if official:
        from ...utils.checker import validate_directory_with_official_checker
        click.echo("Using official checker validation...")
        validate_directory_with_official_checker(results_path, checker_path)
    else:
        from ...utils.checker import validate_all_results, print_validation_report
        click.echo("Validating all solution files...")
        validation_results = validate_all_results(results_path, checker_path)
        print_validation_report(validation_results)


@click.command(name="validate-results")
@click.option("--results-dir", "-r", type=click.Path(), default="res", help="Results directory")
def validate_results(results_dir: str):
    """Validate all benchmark results and show detailed error report."""

    from ...utils.analytics import validate_and_report_errors

    results_path = Path(results_dir)
    validate_and_report_errors(results_path)
