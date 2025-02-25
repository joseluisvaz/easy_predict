#!/usr/bin/env bash
# Adapted from https://github.com/woven-planet/l5kit/blob/master/l5kit/run_tests.sh

# Exit on error
set -e

TEST_TYPE=${1:-"all"}

echo "------------- Running checks -------------"

lint() {
    echo "linting..."
    uvx ruff check .
    echo "\n"
}

check_imports() {
    echo "checking imports..."
    uvx ruff check . --select I 
    echo "\n"
}

format() {
    echo "formatting..."
    uvx ruff format --check .
    echo "\n"
}

types() {
    echo "type checking..."
    uvx mypy . 
    echo "\n"
}

test() {
    echo "testing.."
    uv run python -m pytest
}

lint
check_imports
format
types
test
