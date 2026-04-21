#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dinov3_stack}"

cd "${REPO_ROOT}"

conda run -n "${CONDA_ENV_NAME}" python \
  dinov3/launchers/b200/recipe_search/analysis/search_b200_vit7b_recipe.py \
  "$@"
