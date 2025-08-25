#!/bin/bash

# Automated script to run all STS models on all instances
# As required by the project documentation

set -e

# Configuration
MAX_TEAMS=20
TIMEOUT=300
RESULTS_DIR="res"

echo "======================================"
echo "STS Solver - Comprehensive Benchmark"
echo "======================================"
echo "Max teams: $MAX_TEAMS"
echo "Timeout: ${TIMEOUT}s per solver"
echo "Results directory: $RESULTS_DIR"
echo "======================================"

# Create results directories
mkdir -p $RESULTS_DIR/{CP,SAT,SMT,MIP}

echo ""
echo "Running comprehensive benchmark with all solvers..."
echo "This will test multiple formulations for each approach."
echo ""

# Run comprehensive benchmark
if timeout $((TIMEOUT * 100)) uv run sts-solve comprehensive-benchmark $MAX_TEAMS --timeout $TIMEOUT; then
    echo ""
    echo "✓ Comprehensive benchmark completed successfully"
else
    echo ""
    echo "✗ Comprehensive benchmark encountered issues"
fi

echo ""
echo "======================================"
echo "Benchmark completed!"
echo "======================================"

# Count results
total_results=0
for approach in "${APPROACHES[@]}"; do
    count=$(ls -1 $RESULTS_DIR/$approach/*.json 2>/dev/null | wc -l || echo 0)
    echo "$approach: $count results"
    total_results=$((total_results + count))
done

echo "Total results: $total_results"

echo ""
echo "Running post-benchmark analysis..."
echo "======================================"

# Run comprehensive analysis
if command -v uv >/dev/null 2>&1; then
    echo "📊 Generating benchmark analysis report..."
    uv run sts-solve analyze --format console
    
    echo ""
    echo "📈 Exporting results for further analysis..."
    uv run sts-solve analyze --format json
    uv run sts-solve analyze --format csv
    
    echo ""
    echo "✅ Analysis complete!"
    echo "  - Console report shown above"
    echo "  - JSON data: res/benchmark_analysis.json"  
    echo "  - CSV data: res/benchmark_summary.csv"
else
    echo "To run analysis manually:"
    echo "  uv run sts-solve analyze --format console"
    echo "  uv run sts-solve validate-results"
fi