# Launcher Layout

- `a6000/`
  - Legacy pretraining launchers used for the A6000-era experiments.
  - These scripts preserve the original dataset paths, checkpoint paths, and output layout.
- `h100/`
  - Canonical H100 launchers for the CAG content-state backbone scaling runs.
  - These scripts use the adopted 2-GPU H100 common recipe on `CUDA_VISIBLE_DEVICES=5,6`:
    - `stage1/stage2/stage3 epochs=100/100/30`
    - `stage1 batch_per_gpu=80`
    - `stage2 batch_per_gpu=80`
    - `stage3 batch_per_gpu=28`
  - The launchers now default to `CUDA_VISIBLE_DEVICES=5,6` and infer `torchrun --nproc_per_node` from the visible GPU count.
  - Recipe-search utilities live under `h100/recipe_search/`.
- `../run_knn.sh`
  - Evaluation launcher.
  - It is intentionally kept outside this folder because it is not part of pretraining orchestration.

## H100 Usage

- Stage launchers:
  - `dinov3/launchers/h100/run_1_pretraining.sh`
  - `dinov3/launchers/h100/run_2_gram_anchor.sh`
  - `dinov3/launchers/h100/run_3_high-res-adapt_content_state.sh`
- Full pipeline wrapper:
  - `dinov3/launchers/h100/run_full_content_state_pipeline.sh`
- Recipe-search wrapper:
  - `dinov3/launchers/h100/recipe_search/run/run_h100_common_recipe_search.sh`

## H100 Environment Variables

- `MODEL`
  - Required for the stage launchers.
  - One of `vits`, `vitb`, `vitl`.
- `MODELS`
  - Used by the full wrapper.
  - Defaults to `"vits vitb vitl"`.
- `CONDA_ENV_NAME`
  - Defaults to `dinov3_stack`.
- `CUDA_VISIBLE_DEVICES`
  - Defaults to `5,6`.
- `NPROC_PER_NODE`
  - Optional override for distributed launch size.
  - By default, the H100 launcher infers this from `CUDA_VISIBLE_DEVICES`.
- `DATA_ROOT`
  - Defaults to `/mnt/30TB_SSD/heesu/cag_vision_fm/images`.
- `OUT_ROOT`
  - Defaults to `output/h100/1_pretrain`.
- `TRAIN_NUM_WORKERS`
  - Defaults to `8`.
- `DRY_RUN`
  - Set to `1` to print the resolved command instead of executing it.
- `OMP_NUM_THREADS`
  - Defaults to `1`.
- `ENABLE_DISTRIBUTED_DIAGNOSTICS`
  - Defaults to `1`.
  - When enabled, the launcher adds distributed/NCCL debug environment variables for postmortem diagnosis without changing the training recipe.
- `TORCH_DISTRIBUTED_DEBUG_LEVEL`
  - Defaults to `DETAIL`.
- `NCCL_DEBUG_LEVEL`
  - Defaults to `INFO`.
- `NCCL_DEBUG_SUBSYS_LIST`
  - Defaults to `INIT,COLL`.
- `ENABLE_NCCL_DEBUG_FILE`
  - Defaults to `1`.
  - When enabled, NCCL writes per-process debug logs under each stage output `logs/` directory using `nccl.%h.%p.log`.
- `TORCH_SHOW_CPP_STACKTRACES`
  - Defaults to `1`.
- `PYTHONFAULTHANDLER`
  - Defaults to `1`.

## H100 Checkpoints

- Exact training resume checkpoints are stored under each stage's `ckpt/<iteration>/` directory.
- H100 multi-GPU launchers save eval teacher checkpoints as sharded directories named `sharded_teacher_checkpoint/`.
- Stage2 and stage3 default checkpoint wiring expects those sharded eval directories rather than a single `teacher_checkpoint.pth` file.
