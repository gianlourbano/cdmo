"""Backup command for results (modular CLI)"""

from pathlib import Path
import click


@click.command(name="backup-results")
@click.option("--results-dir", "-r", type=click.Path(), default=None, help="Results directory (defaults from config)")
def backup_results_cmd(results_dir: str):
    """Create a timestamped backup of the results directory."""
    from ...utils.analytics import backup_results
    from ...config import get_config

    cfg = get_config()
    results_path = Path(results_dir) if results_dir else cfg.results_dir
    if not results_path.exists():
        click.echo(f"Error: Results directory {results_path} not found", err=True)
        raise SystemExit(1)

    backup_path = backup_results(results_path)
    click.echo(f"Backup created at: {backup_path}")
