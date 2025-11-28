"""Main CLI entry point (modular)"""

import click

# Commands will be imported and registered below
from .commands import solve as solve_cmd
from .commands import list_solvers as list_models_cmd
from .commands import list_backends as list_backends_cmd
from .commands import validate as validate_cmd
from .commands import benchmark as bench_cmd
from .commands import analyze as analyze_cmd
from .commands import clean as clean_cmd
from .commands import backup as backup_cmd


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Sports Tournament Scheduling (STS) Problem Solver"""
    pass


# Register modular commands
cli.add_command(solve_cmd.solve)
cli.add_command(list_models_cmd.list_models)
cli.add_command(list_backends_cmd.list_backends)
cli.add_command(validate_cmd.validate)
cli.add_command(validate_cmd.validate_all)
cli.add_command(validate_cmd.validate_results)
cli.add_command(bench_cmd.benchmark)
cli.add_command(bench_cmd.comprehensive_benchmark)
cli.add_command(analyze_cmd.analyze)
cli.add_command(analyze_cmd.compare_instance)
cli.add_command(clean_cmd.clean_results)
cli.add_command(clean_cmd.remove_solver)
cli.add_command(backup_cmd.backup_results_cmd)


if __name__ == "__main__":
    cli()
