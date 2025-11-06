"""Cleaning utilities commands (modular CLI)"""

import json
from pathlib import Path
from typing import Optional
import click


@click.command(name="clean-results")
@click.option("--results-dir", "-r", type=click.Path(), default=None, help="Results directory (defaults from config)")
@click.option("--no-dry-run", is_flag=True, help="Actually clean files (default is dry-run mode)")
@click.option("--backup", is_flag=True, help="Create backup before cleaning")
def clean_results(results_dir: str, no_dry_run: bool, backup: bool):
    """Clean invalid solutions from result files."""
    from ...utils.analytics import clean_invalid_solutions, backup_results
    from ...config import get_config

    cfg = get_config()
    results_path = Path(results_dir) if results_dir else cfg.results_dir
    if not results_path.exists():
        click.echo(f"Error: Results directory {results_path} not found", err=True)
        raise SystemExit(1)

    if backup and not no_dry_run:
        backup_results(results_path)
        click.echo()

    dry_run = not no_dry_run
    clean_invalid_solutions(results_path, dry_run=dry_run)


@click.command(name="remove-solver")
@click.option("--results-dir", "-r", type=click.Path(), default=None, help="Results directory (defaults from config)")
@click.option("--solver", help="Specific solver to remove (e.g., 'GLOP')")
@click.option("--no-dry-run", is_flag=True, help="Actually remove solutions (default is dry-run)")
def remove_solver(results_dir: str, solver: Optional[str], no_dry_run: bool):
    """Remove all solutions from a specific solver across results."""
    from ...config import get_config
    cfg = get_config()
    results_path = Path(results_dir) if results_dir else cfg.results_dir
    if not results_path.exists():
        click.echo(f"Error: Results directory {results_path} not found", err=True)
        raise SystemExit(1)

    if not solver:
        click.echo("Error: --solver parameter is required", err=True)
        raise SystemExit(1)

    dry_run = not no_dry_run
    mode_str = "DRY RUN" if dry_run else "REMOVING"

    click.echo(f"SOLVER REMOVAL UTILITY ({mode_str})")
    click.echo("=" * 50)
    click.echo(f"Target solver: {solver}")
    click.echo(f"Directory: {results_path}")
    click.echo()

    files_modified = 0
    solutions_removed = 0

    for approach_dir in results_path.iterdir():
        if not approach_dir.is_dir():
            continue

        approach = approach_dir.name
        click.echo(f"Checking {approach} approach...")

        for result_file in approach_dir.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)

                if solver in data:
                    click.echo(f"  📁 {result_file.relative_to(results_path)}: Found {solver}")

                    if not dry_run:
                        del data[solver]

                        if data:
                            with open(result_file, 'w') as f:
                                json.dump(data, f, indent=2)
                        else:
                            result_file.unlink()
                            click.echo(f"    Removed empty file")

                    files_modified += 1
                    solutions_removed += 1

            except (json.JSONDecodeError, FileNotFoundError):
                continue

    click.echo("\n" + "=" * 50)
    if dry_run:
        click.echo(f"Would remove {solutions_removed} instances of '{solver}' from {files_modified} files")
        click.echo("Use --no-dry-run to actually remove")
    else:
        click.echo(f"Removed {solutions_removed} instances of '{solver}' from {files_modified} files")
        click.echo("Cleaning completed!")
