#!/usr/bin/env bash
# Run the test suite with pytest, using xdist if available.

set -e

# Resolve script directory to allow running from anywhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

PYTEST_ARGS=()
if python -c "import xdist" &>/dev/null; then
    PYTEST_ARGS+=("-n" "4")
fi

echo "Running pytest with: ${PYTEST_ARGS[@]} $@"
python -m pytest "${PYTEST_ARGS[@]}" tests/ "$@"
