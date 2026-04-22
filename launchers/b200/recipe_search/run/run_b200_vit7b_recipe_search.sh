#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dinov3_stack}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/home/gmail_asse/anaconda3/envs/${CONDA_ENV_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PREFIX}/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

"${PYTHON_BIN}" \
  dinov3/launchers/b200/recipe_search/analysis/search_b200_vit7b_recipe.py \
  "$@"
