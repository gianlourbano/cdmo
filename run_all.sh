#!/bin/bash

# Run benchmarks based on res/max_n_summary.txt
# Each line format: (approach, model_name, max_n, optimization, solver_name)

set -euo pipefail

SUMMARY_FILE="res/max_n_summary.txt"

if [[ ! -f "$SUMMARY_FILE" ]]; then
	echo "Summary file not found: $SUMMARY_FILE"
	echo "Run: uv run ./scripts/max_n_report.py"
	exit 1
fi

while IFS= read -r line; do
	# skip empty lines
	[[ -z "$line" ]] && continue

	# strip parentheses
	clean=${line#(}
	clean=${clean%)}
	# split by comma into array
	IFS=',' read -r approach model_name max_n optimization solver_name <<< "$clean"

	# trim spaces
	approach=$(echo "$approach" | xargs)
	model_name=$(echo "$model_name" | xargs)
	max_n=$(echo "$max_n" | xargs)
	optimization=$(echo "$optimization" | xargs)
	solver_name=$(echo "$solver_name" | xargs)

	# compute target n = max_n + 2
	if [[ "$max_n" =~ ^[0-9]+$ ]]; then
		target_n=$((max_n + 2))
	else
		echo "Invalid max_n in line: $line"
		continue
	fi

	cmd=(uv run sts benchmark "$target_n" --approach "$approach" --model "$model_name")

	# add flags conditionally
	if [[ "$optimization" == "True" || "$optimization" == "true" ]]; then
		cmd+=(--optimization)
	fi
	if [[ "$solver_name" != "None" && "$solver_name" != "null" && -n "$solver_name" ]]; then
		cmd+=(--solver "$solver_name")
	fi

	echo "Running: ${cmd[*]}"
	"${cmd[@]}"
done < "$SUMMARY_FILE"
