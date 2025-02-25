#!/usr/bin/env bash
# Adapted from https://github.com/woven-planet/l5kit/blob/master/l5kit/run_tests.sh

# Exit on error
set -e

TEST_TYPE=${1:-"all"}

echo "------------- Running formatting -------------"

sort_imports() {
    echo "sorting imports..."
    uvx ruff check . --select I --fix
    echo "\n"
}

format() {
    echo "formatting..."
    uvx ruff format .
    echo "\n"
}

sort_imports
format
