# Launcher Layout

- `a6000/`
  - Legacy pretraining launchers used for the A6000-era experiments.
  - These scripts preserve the original dataset paths, checkpoint paths, and output layout.
- `h100/`
  - Canonical H100 launchers for the CAG content-state backbone scaling runs.
  - These scripts use the adopted single-H100 common recipe:
    - `stage1/stage2/stage3 epochs=100/100/30`
    - `stage1 batch_per_gpu=72`
    - `stage2 batch_per_gpu=72`
    - `stage3 batch_per_gpu=24`
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
  - Defaults to `6`.
- `DATA_ROOT`
  - Defaults to `/mnt/30TB_SSD/heesu/cag_vision_fm/images`.
- `OUT_ROOT`
  - Defaults to `output/h100/1_pretrain`.
- `TRAIN_NUM_WORKERS`
  - Defaults to `8`.
- `DRY_RUN`
  - Set to `1` to print the resolved command instead of executing it.
