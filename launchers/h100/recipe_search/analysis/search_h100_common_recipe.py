#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[5]
DINO_ROOT = REPO_ROOT / "dinov3"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "h100_recipe_search"
DOC_PATH = REPO_ROOT / "docs" / "experiments" / "h100_common_recipe_search.md"
DEFAULT_DATA_ROOT = Path("/mnt/30TB_SSD/heesu/cag_vision_fm/images")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    arch: str
    embed_dim: int
    decoder_hidden_dim: int


MODELS: List[ModelSpec] = [
    ModelSpec("vitl", "vit_large", 1024, 2048),
    ModelSpec("vitb", "vit_base", 768, 1536),
    ModelSpec("vits", "vit_small", 384, 768),
]

STAGES = {
    "stage1": {
        "config": DINO_ROOT / "dinov3" / "configs" / "train" / "dinov3_h100_hybrid_cag_pretrain.yaml",
        "dataset": "CAGContrastFM3M:split=TRAIN:root={root}",
        "candidates": [80, 72, 64, 56, 48, 32, 24, 16],
        "safe_batch": 16,
        "stability_iters": 50,
        "base_lr_peak": 5.0e-05,
        "base_lr_end": 5.0e-05,
        "base_lr_warmup": 30,
        "base_teacher_warmup": 30,
    },
    "stage2": {
        "config": DINO_ROOT / "dinov3" / "configs" / "train" / "dinov3_h100_hybrid_cag_gram_anchor.yaml",
        "dataset": "CAGContrastFM3M:split=TRAIN:root={root}",
        "candidates": [80, 76, 72, 68, 64, 56, 48, 32, 24, 16],
        "safe_batch": 16,
        "stability_iters": 50,
        "reference_epochs": 100,
        "base_lr_peak": 3.0e-05,
        "base_lr_end": 3.0e-05,
        "base_lr_warmup": 8,
        "base_teacher_warmup": 8,
        "base_local_loss_warmup": 83,
        "base_gram_loss_warmup": 83,
    },
    "stage3": {
        "config": DINO_ROOT
        / "dinov3"
        / "configs"
        / "train"
        / "dinov3_h100_hybrid_cag_high_res_adapt_content_state.yaml",
        "dataset": "CAGContrastFMContinuityV1:split=TRAIN:root={root}",
        "candidates": [40, 36, 32, 28, 24, 20, 16, 12, 8],
        "safe_batch": 8,
        "stability_iters": 100,
        "base_lr_peak": 0.0,
        "base_lr_end": 1.25e-05,
        "base_lr_warmup": 0,
        "base_teacher_warmup": 0,
    },
}

GPU_MODES = {
    "single": {
        "gpu_ids": [5],
        "cuda_visible_devices": "5",
        "num_workers": 8,
    },
    "two": {
        "gpu_ids": [5, 6],
        "cuda_visible_devices": "5,6",
        "num_workers": 8,
    },
}

LR_SCALES = [1.0, 0.5, 0.25]
WARMUP_CANDIDATES = {
    "stage1": [30, 40, 50],
    "stage2": [8, 12, 16],
    "stage3": [0, 2, 5],
}


def bool_str(value: bool) -> str:
    return "true" if value else "false"


def run_checked(cmd: List[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def preflight(data_root: Path) -> dict:
    result = {
        "repo_root": str(REPO_ROOT),
        "dino_root": str(DINO_ROOT),
        "data_root": str(data_root),
        "gpu_status": {},
        "continuity_cache_ok": False,
        "python": sys.executable,
    }
    nvidia = run_checked(["nvidia-smi"], cwd=REPO_ROOT)
    if nvidia.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed:\n{nvidia.stderr}")
    result["nvidia_smi"] = nvidia.stdout
    required_gpu_ids = sorted({gpu_id for spec in GPU_MODES.values() for gpu_id in spec["gpu_ids"]})
    for gpu_id in required_gpu_ids:
        query = run_checked(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            cwd=REPO_ROOT,
        )
        if query.returncode != 0:
            raise RuntimeError(f"failed to query GPU {gpu_id}:\n{query.stderr}")
        result["gpu_status"][str(gpu_id)] = query.stdout.strip()

    required = [
        data_root / "train",
        data_root / "cache" / "continuity_v1" / "image_relpaths_train.txt",
        data_root / "cache" / "continuity_v1" / "unique_dicom_keys_train.txt",
        data_root / "cache" / "continuity_v1" / "sample_to_dicom_idx_train.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"dataset preflight failed, missing: {missing}")
    result["continuity_cache_ok"] = True
    return result


class GPUMonitor:
    def __init__(self, gpu_ids: List[int], interval_sec: float = 1.0):
        self.gpu_ids = gpu_ids
        self.interval_sec = interval_sec
        self.max_used = {gpu_id: 0 for gpu_id in gpu_ids}
        self.total = {gpu_id: 0 for gpu_id in gpu_ids}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=2)
        headroom = {
            gpu_id: (self.total[gpu_id] - self.max_used[gpu_id]) if self.total[gpu_id] else None
            for gpu_id in self.gpu_ids
        }
        return {
            "max_used_mib": self.max_used,
            "total_mib": self.total,
            "headroom_mib": headroom,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            query = run_checked(
                [
                    "nvidia-smi",
                    f"--id={','.join(str(gpu_id) for gpu_id in self.gpu_ids)}",
                    "--query-gpu=index,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                cwd=REPO_ROOT,
            )
            if query.returncode == 0:
                for line in query.stdout.strip().splitlines():
                    if not line.strip():
                        continue
                    index_s, used_s, total_s = [part.strip() for part in line.split(",")]
                    gpu_id = int(index_s)
                    used = int(used_s)
                    total = int(total_s)
                    if gpu_id in self.max_used:
                        self.max_used[gpu_id] = max(self.max_used[gpu_id], used)
                        self.total[gpu_id] = total
            self._stop.wait(self.interval_sec)


def stage_overrides(
    stage: str,
    model: ModelSpec,
    *,
    data_root: Path,
    fp8: bool,
    batch_per_gpu: int,
    num_workers: int,
) -> List[str]:
    spec = STAGES[stage]
    overrides = [
        f"train.dataset_path={spec['dataset'].format(root=data_root)}",
        f"train.batch_size_per_gpu={batch_per_gpu}",
        f"train.num_workers={num_workers}",
        "train.pin_memory=false",
        "train.sharded_eval_checkpoint=true",
        f"student.arch={model.arch}",
        f"student.fp8_enabled={bool_str(fp8)}",
        "student.fp8_filter=blocks",
    ]
    if stage == "stage3":
        overrides.extend(
            [
                f"content_state.head_hidden_dim={model.embed_dim}",
                f"content_state.head_out_dim={model.embed_dim}",
                f"content_state.decoder_hidden_dim={model.decoder_hidden_dim}",
            ]
        )
    return overrides


def schedule_overrides(stage: str, *, lr_scale: float, lr_warmup: int, teacher_warmup: int) -> List[str]:
    spec = STAGES[stage]
    return [
        f"schedules.lr.peak={spec['base_lr_peak'] * lr_scale}",
        f"schedules.lr.end={spec['base_lr_end'] * lr_scale}",
        f"schedules.lr.warmup_epochs={lr_warmup}",
        f"schedules.teacher_temp.warmup_epochs={teacher_warmup}",
    ]


def fit_probe_schedule_overrides(stage: str) -> List[str]:
    overrides = [
        "schedules.lr.warmup_epochs=0",
        "schedules.teacher_temp.warmup_epochs=0",
    ]
    if stage in ("stage2", "stage3"):
        overrides.extend(
            [
                "dino.local_loss_weight_schedule.warmup_epochs=0",
                "dino.local_loss_weight_schedule.cosine_epochs=1",
                "gram.loss_weight_schedule.warmup_epochs=0",
                "gram.loss_weight_schedule.cosine_epochs=1",
            ]
        )
    if stage == "stage3":
        overrides.append("schedules.lr.cosine_epochs=1")
    return overrides


def make_output_dir(phase: str, gpu_mode: str, stage: str, model: str, suffix: str) -> Path:
    path = OUTPUT_ROOT / phase / gpu_mode / stage / model / suffix
    path.mkdir(parents=True, exist_ok=True)
    return path


def teacher_ckpt(output_dir: Path) -> Path:
    return output_dir / "eval" / "training_0" / "sharded_teacher_checkpoint"


def classify_failure(stdout: str, stderr: str) -> str:
    haystack = f"{stdout}\n{stderr}".lower()
    if "out of memory" in haystack or "cuda error: out of memory" in haystack:
        return "oom"
    if "nan" in haystack:
        return "nan"
    return "runtime_error"


def run_train(
    *,
    stage: str,
    model: ModelSpec,
    gpu_mode: str,
    batch_per_gpu: int,
    fp8: bool,
    compile_enabled: bool,
    epochs: int,
    official_epoch_length: int,
    eval_period: int,
    output_dir: Path,
    data_root: Path,
    extra_overrides: List[str],
) -> dict:
    gpu_spec = GPU_MODES[gpu_mode]
    config_path = STAGES[stage]["config"]
    overrides = stage_overrides(
        stage,
        model,
        data_root=data_root,
        fp8=fp8,
        batch_per_gpu=batch_per_gpu,
        num_workers=gpu_spec["num_workers"],
    )
    overrides.extend(
        [
            f"train.compile={bool_str(compile_enabled)}",
            f"optim.epochs={epochs}",
            f"train.OFFICIAL_EPOCH_LENGTH={official_epoch_length}",
            f"evaluation.eval_period_iterations={eval_period}",
            "checkpointing.period=1000000000",
        ]
    )
    overrides.extend(extra_overrides)

    if gpu_mode == "single":
        cmd = [
            sys.executable,
            "dinov3/train/train.py",
            "--config-file",
            str(config_path),
            "--output-dir",
            str(output_dir),
            *overrides,
        ]
    else:
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "dinov3.train.train",
            "--config-file",
            str(config_path),
            "--output-dir",
            str(output_dir),
            *overrides,
        ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_spec["cuda_visible_devices"]
    env["PYTHONPATH"] = str(DINO_ROOT)
    env.setdefault("OMP_NUM_THREADS", "1")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "stdout.log"
    monitor = GPUMonitor(gpu_spec["gpu_ids"])
    monitor.start()
    start = time.time()
    with log_path.open("w") as log_file:
        process = subprocess.run(
            cmd,
            cwd=DINO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    monitor_stats = monitor.stop()
    duration = time.time() - start
    stdout = log_path.read_text()
    status = "success" if process.returncode == 0 else classify_failure(stdout, "")
    return {
        "stage": stage,
        "model": model.key,
        "gpu_mode": gpu_mode,
        "batch_per_gpu": batch_per_gpu,
        "fp8": fp8,
        "compile": compile_enabled,
        "epochs": epochs,
        "official_epoch_length": official_epoch_length,
        "eval_period": eval_period,
        "cmd": cmd,
        "status": status,
        "returncode": process.returncode,
        "duration_sec": duration,
        "log_path": str(log_path),
        "output_dir": str(output_dir),
        "monitor": monitor_stats,
    }


def ensure_stage_bootstrap_ckpt(
    model: ModelSpec,
    gpu_mode: str,
    fp8: bool,
    target_stage: str,
    data_root: Path,
    results: dict,
) -> Path | None:
    if target_stage not in ("stage1", "stage2"):
        raise ValueError(f"bootstrap only supports stage1/stage2, got {target_stage}")
    cache_key = f"{gpu_mode}:{target_stage}:{model.key}:fp8={int(fp8)}"
    if cache_key in results["bootstrap_ckpts"]:
        return Path(results["bootstrap_ckpts"][cache_key])

    prereq = None
    if target_stage == "stage2":
        prereq = ensure_stage_bootstrap_ckpt(model, gpu_mode, fp8, "stage1", data_root, results)
        if prereq is None:
            return None

    safe_batch = STAGES[target_stage]["safe_batch"]
    suffix = f"bootstrap_b{safe_batch}_fp8{int(fp8)}"
    output_dir = make_output_dir("bootstrap", gpu_mode, target_stage, model.key, suffix)
    extra = fit_probe_schedule_overrides(target_stage)
    if target_stage == "stage2":
        extra.extend([f"gram.ckpt={prereq}", f"student.resume_from_teacher_chkpt={prereq}"])

    result = run_train(
        stage=target_stage,
        model=model,
        gpu_mode=gpu_mode,
        batch_per_gpu=safe_batch,
        fp8=fp8,
        compile_enabled=False,
        epochs=1,
        official_epoch_length=1,
        eval_period=1,
        output_dir=output_dir,
        data_root=data_root,
        extra_overrides=extra,
    )
    record_run(results, result)
    ckpt = teacher_ckpt(output_dir)
    if result["status"] != "success" or not ckpt.exists():
        return None
    results["bootstrap_ckpts"][cache_key] = str(ckpt)
    return ckpt


def find_common_batch(stage: str, gpu_mode: str, fp8: bool, data_root: Path, results: dict) -> dict:
    common = {"batch_per_gpu": None, "runs": []}
    for candidate in STAGES[stage]["candidates"]:
        candidate_ok = True
        candidate_runs = []
        for model in MODELS:
            prereq_ckpt = None
            if stage == "stage2":
                prereq_ckpt = ensure_stage_bootstrap_ckpt(model, gpu_mode, fp8, "stage1", data_root, results)
            elif stage == "stage3":
                prereq_ckpt = ensure_stage_bootstrap_ckpt(model, gpu_mode, fp8, "stage2", data_root, results)
            if stage != "stage1" and prereq_ckpt is None:
                candidate_ok = False
                break
            extra = fit_probe_schedule_overrides(stage)
            if prereq_ckpt is not None:
                extra.extend([f"gram.ckpt={prereq_ckpt}", f"student.resume_from_teacher_chkpt={prereq_ckpt}"])
            suffix = f"fit_b{candidate}_fp8{int(fp8)}"
            output_dir = make_output_dir("probes", gpu_mode, stage, model.key, suffix)
            run = run_train(
                stage=stage,
                model=model,
                gpu_mode=gpu_mode,
                batch_per_gpu=candidate,
                fp8=fp8,
                compile_enabled=False,
                epochs=1,
                official_epoch_length=1,
                eval_period=1,
                output_dir=output_dir,
                data_root=data_root,
                extra_overrides=extra,
            )
            record_run(results, run)
            candidate_runs.append(run)
            if run["status"] != "success":
                candidate_ok = False
                break
        common["runs"].extend(candidate_runs)
        if candidate_ok:
            common["batch_per_gpu"] = candidate
            break
    return common


def fit_batch_candidate(stage: str, gpu_mode: str, fp8: bool, batch_per_gpu: int, data_root: Path, results: dict) -> bool:
    for model in MODELS:
        prereq_ckpt = None
        if stage == "stage2":
            prereq_ckpt = ensure_stage_bootstrap_ckpt(model, gpu_mode, fp8, "stage1", data_root, results)
        elif stage == "stage3":
            prereq_ckpt = ensure_stage_bootstrap_ckpt(model, gpu_mode, fp8, "stage2", data_root, results)
        if stage != "stage1" and prereq_ckpt is None:
            return False
        extra = fit_probe_schedule_overrides(stage)
        if prereq_ckpt is not None:
            extra.extend([f"gram.ckpt={prereq_ckpt}", f"student.resume_from_teacher_chkpt={prereq_ckpt}"])
        suffix = f"fit_b{batch_per_gpu}_fp8{int(fp8)}"
        output_dir = make_output_dir("probes", gpu_mode, stage, model.key, suffix)
        run = run_train(
            stage=stage,
            model=model,
            gpu_mode=gpu_mode,
            batch_per_gpu=batch_per_gpu,
            fp8=fp8,
            compile_enabled=False,
            epochs=1,
            official_epoch_length=1,
            eval_period=1,
            output_dir=output_dir,
            data_root=data_root,
            extra_overrides=extra,
        )
        record_run(results, run)
        if run["status"] != "success":
            return False
    return True


def stage_short_stability(
    stage: str,
    gpu_mode: str,
    fp8: bool,
    batch_per_gpu: int,
    data_root: Path,
    results: dict,
) -> dict | None:
    chosen = None
    for lr_scale in LR_SCALES:
        for warmup in WARMUP_CANDIDATES[stage]:
            candidate = {
                "lr_scale": lr_scale,
                "lr_peak": STAGES[stage]["base_lr_peak"] * lr_scale,
                "lr_end": STAGES[stage]["base_lr_end"] * lr_scale,
                "lr_warmup_epochs": warmup,
                "teacher_temp_warmup_epochs": warmup,
            }
            vitl = next(model for model in MODELS if model.key == "vitl")
            prereq_ckpt = None
            if stage == "stage2":
                prereq_ckpt = ensure_stage_bootstrap_ckpt(vitl, gpu_mode, fp8, "stage1", data_root, results)
            elif stage == "stage3":
                prereq_ckpt = ensure_stage_bootstrap_ckpt(vitl, gpu_mode, fp8, "stage2", data_root, results)
            if stage != "stage1" and prereq_ckpt is None:
                return None
            extra = schedule_overrides(
                stage,
                lr_scale=lr_scale,
                lr_warmup=warmup,
                teacher_warmup=warmup,
            )
            if stage == "stage2":
                aux_warmup = int(
                    round(
                        STAGES[stage]["base_local_loss_warmup"]
                        * STAGES[stage]["stability_iters"]
                        / STAGES[stage]["reference_epochs"]
                    )
                )
                aux_warmup = max(0, min(STAGES[stage]["stability_iters"] - 1, aux_warmup))
                extra.extend(
                    [
                        f"dino.local_loss_weight_schedule.warmup_epochs={aux_warmup}",
                        "dino.local_loss_weight_schedule.cosine_epochs=1",
                        f"gram.loss_weight_schedule.warmup_epochs={aux_warmup}",
                        "gram.loss_weight_schedule.cosine_epochs=1",
                    ]
                )
            if prereq_ckpt is not None:
                extra.extend([f"gram.ckpt={prereq_ckpt}", f"student.resume_from_teacher_chkpt={prereq_ckpt}"])
            suffix = f"stability_vitl_b{batch_per_gpu}_fp8{int(fp8)}_lr{lr_scale}_wu{warmup}"
            run = run_train(
                stage=stage,
                model=vitl,
                gpu_mode=gpu_mode,
                batch_per_gpu=batch_per_gpu,
                fp8=fp8,
                compile_enabled=False,
                epochs=STAGES[stage]["stability_iters"],
                official_epoch_length=1,
                eval_period=0,
                output_dir=make_output_dir("stability", gpu_mode, stage, vitl.key, suffix),
                data_root=data_root,
                extra_overrides=extra,
            )
            record_run(results, run)
            if run["status"] == "success":
                chosen = candidate
                break
        if chosen is not None:
            break

    if chosen is None:
        return None

    for model in MODELS[1:]:
        prereq_ckpt = None
        if stage == "stage2":
            prereq_ckpt = ensure_stage_bootstrap_ckpt(model, gpu_mode, fp8, "stage1", data_root, results)
        elif stage == "stage3":
            prereq_ckpt = ensure_stage_bootstrap_ckpt(model, gpu_mode, fp8, "stage2", data_root, results)
        if stage != "stage1" and prereq_ckpt is None:
            return None
        extra = schedule_overrides(
            stage,
            lr_scale=chosen["lr_scale"],
            lr_warmup=chosen["lr_warmup_epochs"],
            teacher_warmup=chosen["teacher_temp_warmup_epochs"],
        )
        if stage == "stage2":
            aux_warmup = int(
                round(
                    STAGES[stage]["base_local_loss_warmup"]
                    * STAGES[stage]["stability_iters"]
                    / STAGES[stage]["reference_epochs"]
                )
            )
            aux_warmup = max(0, min(STAGES[stage]["stability_iters"] - 1, aux_warmup))
            extra.extend(
                [
                    f"dino.local_loss_weight_schedule.warmup_epochs={aux_warmup}",
                    "dino.local_loss_weight_schedule.cosine_epochs=1",
                    f"gram.loss_weight_schedule.warmup_epochs={aux_warmup}",
                    "gram.loss_weight_schedule.cosine_epochs=1",
                ]
            )
        if prereq_ckpt is not None:
            extra.extend([f"gram.ckpt={prereq_ckpt}", f"student.resume_from_teacher_chkpt={prereq_ckpt}"])
        suffix = (
            f"stability_{model.key}_b{batch_per_gpu}_fp8{int(fp8)}"
            f"_lr{chosen['lr_scale']}_wu{chosen['lr_warmup_epochs']}"
        )
        run = run_train(
            stage=stage,
            model=model,
            gpu_mode=gpu_mode,
            batch_per_gpu=batch_per_gpu,
            fp8=fp8,
            compile_enabled=False,
            epochs=STAGES[stage]["stability_iters"],
            official_epoch_length=1,
            eval_period=0,
            output_dir=make_output_dir("stability", gpu_mode, stage, model.key, suffix),
            data_root=data_root,
            extra_overrides=extra,
        )
        record_run(results, run)
        if run["status"] != "success":
            return None
    return chosen


def find_stage_recipe(stage: str, gpu_mode: str, fp8: bool, data_root: Path, results: dict) -> dict | None:
    for candidate in STAGES[stage]["candidates"]:
        if not fit_batch_candidate(stage, gpu_mode, fp8, candidate, data_root, results):
            continue
        stability = stage_short_stability(stage, gpu_mode, fp8, candidate, data_root, results)
        if stability is not None:
            return {
                "batch_per_gpu": candidate,
                **stability,
            }
    return None


def write_results_json(results: dict) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "results.json").open("w") as f:
        json.dump(results, f, indent=2)


def record_run(results: dict, run: dict) -> None:
    results["runs"].append(run)
    write_results_json(results)


def write_markdown(results: dict) -> None:
    lines = [
        "# H100 Common Recipe Search",
        "",
        "This file is generated by `dinov3/launchers/h100/recipe_search/analysis/search_h100_common_recipe.py`.",
        "",
        "## Outcome",
        "",
        f"- Adopted GPU mode: `{results.get('adopted_gpu_mode', 'unresolved')}`",
        f"- FP8 fallback used: `{results.get('fp8_used', False)}`",
        f"- Data root: `{results['preflight']['data_root']}`",
        "",
        "## Stage Recipes",
        "",
        "| Stage | Batch / GPU | Global Batch | LR Peak | LR End | LR Warmup | Teacher Warmup | FP8 | Compile | Num Workers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for stage in ("stage1", "stage2", "stage3"):
        stage_result = results["stage_results"].get(stage)
        if not stage_result:
            lines.append(f"| {stage} | unresolved | unresolved | unresolved | unresolved | unresolved | unresolved | unresolved | unresolved | unresolved |")
            continue
        gpu_mode = results["adopted_gpu_mode"]
        world_size = len(GPU_MODES[gpu_mode]["gpu_ids"])
        lines.append(
            "| {stage} | {batch} | {global_batch} | {lr_peak:.6g} | {lr_end:.6g} | {lr_warmup} | {teacher_warmup} | {fp8} | false | {workers} |".format(
                stage=stage,
                batch=stage_result["batch_per_gpu"],
                global_batch=stage_result["batch_per_gpu"] * world_size,
                lr_peak=stage_result["lr_peak"],
                lr_end=stage_result["lr_end"],
                lr_warmup=stage_result["lr_warmup_epochs"],
                teacher_warmup=stage_result["teacher_temp_warmup_epochs"],
                fp8=results["fp8_used"],
                workers=GPU_MODES[gpu_mode]["num_workers"],
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Stage3 local batch floor was enforced at `8` per GPU.",
            "- Raw probe logs live under `outputs/h100_recipe_search/`.",
            "- Canonical H100 training launchers live under `dinov3/launchers/h100/`.",
            "- H100 recipe search scripts live under `dinov3/launchers/h100/recipe_search/`.",
        ]
    )
    if results.get("reused_stage_results"):
        lines.append(f"- Reused stage results: `{', '.join(results['reused_stage_results'])}`.")
    lines.extend(
        [
            "",
            "## Failed Candidates",
            "",
        ]
    )
    failed = [run for run in results["runs"] if run["status"] != "success"]
    if not failed:
        lines.append("- None")
    else:
        for run in failed:
            lines.append(
                f"- `{run['gpu_mode']}` `{run['stage']}` `{run['model']}` "
                f"`b{run['batch_per_gpu']}` `fp8={int(run['fp8'])}` -> `{run['status']}`"
            )

    DOC_PATH.write_text("\n".join(lines) + "\n")


def search_gpu_mode(
    gpu_mode: str,
    data_root: Path,
    results: dict,
    *,
    stage_order: List[str],
    preset_stage_results: Dict[str, dict] | None = None,
) -> bool:
    fp8_candidates = [False, True]
    for fp8 in fp8_candidates:
        if gpu_mode == "single" and fp8:
            # Only use fp8 fallback after fp8=false failed single-GPU gating.
            pass
        stage_results = dict(preset_stage_results or {})
        for stage in stage_order:
            if stage in stage_results:
                continue
            recipe = find_stage_recipe(stage, gpu_mode, fp8, data_root, results)
            if recipe is None:
                stage_results = {}
                break
            stage_results[stage] = recipe
        if stage_results:
            results["adopted_gpu_mode"] = gpu_mode
            results["fp8_used"] = fp8
            results["stage_results"] = stage_results
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Search H100-common CAG FM recipe.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--gpu-mode",
        choices=("auto", "single", "two"),
        default="auto",
        help="Search only the requested GPU mode, or try single then two when set to auto.",
    )
    parser.add_argument(
        "--search-stages",
        nargs="+",
        choices=("stage1", "stage2", "stage3"),
        default=("stage3", "stage2", "stage1"),
        help="Stages to actively search, in search order.",
    )
    parser.add_argument(
        "--reuse-stage-results",
        nargs="*",
        choices=("stage1", "stage2", "stage3"),
        default=(),
        help="Reuse stage results from an existing results JSON instead of re-searching them.",
    )
    parser.add_argument(
        "--reuse-results-from",
        type=Path,
        default=OUTPUT_ROOT / "results.json",
        help="Results JSON used when --reuse-stage-results is provided.",
    )
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": [],
        "bootstrap_ckpts": {},
        "stage_results": {},
        "reused_stage_results": [],
    }
    results["preflight"] = preflight(args.data_root)

    preset_stage_results: Dict[str, dict] = {}
    if args.reuse_stage_results:
        if not args.reuse_results_from.exists():
            raise FileNotFoundError(f"reuse results file not found: {args.reuse_results_from}")
        previous = json.loads(args.reuse_results_from.read_text())
        previous_stage_results = previous.get("stage_results", {})
        for stage in args.reuse_stage_results:
            if stage not in previous_stage_results:
                raise ValueError(f"stage result {stage} not found in {args.reuse_results_from}")
            preset_stage_results[stage] = previous_stage_results[stage]
        results["stage_results"].update(preset_stage_results)
        results["reused_stage_results"] = list(args.reuse_stage_results)

    write_results_json(results)

    if args.gpu_mode == "auto":
        success = search_gpu_mode(
            "single",
            args.data_root,
            results,
            stage_order=list(args.search_stages),
            preset_stage_results=preset_stage_results,
        )
        if not success:
            success = search_gpu_mode(
                "two",
                args.data_root,
                results,
                stage_order=list(args.search_stages),
                preset_stage_results=preset_stage_results,
            )
    else:
        success = search_gpu_mode(
            args.gpu_mode,
            args.data_root,
            results,
            stage_order=list(args.search_stages),
            preset_stage_results=preset_stage_results,
        )

    if not success:
        results["adopted_gpu_mode"] = "unresolved"
        results["fp8_used"] = False

    write_results_json(results)
    write_markdown(results)

    summary = textwrap.dedent(
        f"""
        Search finished.
        Adopted GPU mode: {results.get('adopted_gpu_mode', 'unresolved')}
        FP8 used: {results.get('fp8_used', False)}
        Results JSON: {OUTPUT_ROOT / 'results.json'}
        Markdown summary: {DOC_PATH}
        """
    ).strip()
    print(summary)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
