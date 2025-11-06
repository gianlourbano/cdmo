"""Analyze commands for results (modular CLI)"""

from dataclasses import asdict
from pathlib import Path
import json
import csv
import click


@click.command()
@click.option("--results-dir", "-r", type=click.Path(), default=None, help="Results directory (defaults from config)")
@click.option("--format", type=click.Choice(["console", "json", "csv"]), default="console", help="Output format")
def analyze(results_dir: str, format: str):
    """Analyze benchmark results and generate comprehensive statistics."""
    from ...utils.analytics import analyze_benchmark_results, print_comprehensive_report, validate_and_report_errors, load_all_results
    from ...config import get_config

    cfg = get_config()
    results_path = Path(results_dir) if results_dir else cfg.results_dir
    if not results_path.exists():
        click.echo(f"Error: Results directory {results_path} not found", err=True)
        raise SystemExit(1)

    click.echo("Analyzing benchmark results...")

    # Validate all solutions first
    validate_and_report_errors(results_path)

    # Generate analysis
    stats = analyze_benchmark_results(results_path)

    if format == "console":
        print_comprehensive_report(stats, results_path)
    elif format == "json":
        json_stats = {approach: asdict(stat) for approach, stat in stats.items()}
        output_file = results_path / "benchmark_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(json_stats, f, indent=2, default=str)
        click.echo(f"Analysis saved to {output_file}")
    elif format == "csv":
        output_file = results_path / "benchmark_summary.csv"
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Approach', 'Solver', 'Instance', 'Time', 'Optimal', 'Valid', 'Objective'])
            all_results = load_all_results(results_path)
            for result in all_results:
                writer.writerow([
                    result.approach, result.solver, result.instance,
                    result.time, result.optimal, result.is_valid, result.obj
                ])
        click.echo(f"CSV data saved to {output_file}")


@click.command(name="compare-instance")
@click.argument("instance_size", type=int)
@click.option("--results-dir", "-r", type=click.Path(), default="res", help="Results directory")
def compare_instance(instance_size: int, results_dir: str):
    """Compare solver performance on a specific instance size."""
    from ...utils.analytics import load_all_results

    results_path = Path(results_dir)
    all_results = load_all_results(results_path)
    instance_results = [r for r in all_results if r.instance == instance_size]

    if not instance_results:
        click.echo(f"No results found for instance size {instance_size}")
        return

    click.echo(f"PERFORMANCE COMPARISON: {instance_size} teams")
    click.echo("=" * 60)

    by_approach = {}
    for result in instance_results:
        by_approach.setdefault(result.approach, []).append(result)

    for approach, results in by_approach.items():
        click.echo(f"\n{approach} Approach:")
        click.echo("-" * 30)
        results.sort(key=lambda r: (not bool(r.sol), r.time))
        for result in results:
            status = "PASS" if result.is_valid else "FAIL" if result.sol else "TIMEOUT"
            opt_str = " (OPT)" if result.optimal else ""
            click.echo(f"  {status} {result.solver:<20} {result.time:>4}s{opt_str}")
