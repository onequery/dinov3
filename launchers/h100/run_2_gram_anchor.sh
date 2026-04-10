#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

cd "${DINO_ROOT}"
require_model

STAGE1_CKPT="${STAGE1_TEACHER_CKPT:-$(default_stage1_teacher_ckpt)}"
require_file_unless_dry_run "${STAGE1_CKPT}"

launch_h100_stage \
  "stage2" \
  "gram.ckpt=${STAGE1_CKPT}" \
  "student.resume_from_teacher_chkpt=${STAGE1_CKPT}"

