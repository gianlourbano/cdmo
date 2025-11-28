"""List available backend engines per approach.

- CP: MiniZinc backends (gecode, chuffed)
- MIP: OR-Tools backends (CBC, SCIP, GUROBI)
- SAT/SMT: No external backends (use --model only)
"""

import click


@click.command(name="list-backends")
@click.option("--approach", "-a", type=click.Choice(["CP", "SAT", "SMT", "MIP"]), required=False, help="Filter by approach")
@click.option("--verbose", "-v", is_flag=True, help="Show additional notes")
def list_backends(approach: str | None, verbose: bool):
    """List supported backend engines by approach."""
    mapping = {
        "CP": ["gecode", "chuffed"],
        "MIP": ["CBC", "SCIP", "GUROBI"],
        "SAT": [],
        "SMT": [],
    }

    def print_one(app: str):
        items = mapping.get(app, [])
        click.echo(f"=== {app} Backends ===")
        if items:
            for b in items:
                click.echo(f"  - {b}")
            if verbose and app == "MIP":
                click.echo("  Note: GUROBI requires a license and may be unavailable in Docker.")
        else:
            click.echo("  (none) — backends not applicable; use --model only")
        click.echo()

    if approach:
        print_one(approach)
    else:
        for app in ("CP", "SAT", "SMT", "MIP"):
            print_one(app)
