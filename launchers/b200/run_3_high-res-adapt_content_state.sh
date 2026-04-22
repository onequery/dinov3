#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

cd "${DINO_ROOT}"
require_model

STAGE2_CKPT="${STAGE2_TEACHER_CKPT:-$(default_stage2_teacher_ckpt)}"
require_file_unless_dry_run "${STAGE2_CKPT}"

launch_b200_stage \
  "stage3" \
  "gram.ckpt=${STAGE2_CKPT}" \
  "student.resume_from_teacher_chkpt=${STAGE2_CKPT}"
