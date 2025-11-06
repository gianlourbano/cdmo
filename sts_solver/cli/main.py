"""Main CLI entry point (modular)"""

import click

# Commands will be imported and registered below
from .commands import solve as solve_cmd
from .commands import list_solvers as list_cmd


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Sports Tournament Scheduling (STS) Problem Solver"""
    pass


# Register modular commands
cli.add_command(solve_cmd.solve)
cli.add_command(list_cmd.list_solvers)


if __name__ == "__main__":
    cli()
