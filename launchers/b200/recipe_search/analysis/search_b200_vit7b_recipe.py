#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
DINO_ROOT = REPO_ROOT / "dinov3"
DEFAULT_OUTPUT_ROOT = DINO_ROOT / "output" / "b200_recipe_search"
DEFAULT_DOC_PATH = REPO_ROOT / "docs" / "experiments" / "b200_vit7b_recipe_search.md"
DEFAULT_RECIPE_ENV_PATH = DINO_ROOT / "launchers" / "b200" / "recipe.env"
DEFAULT_DATA_ROOT = Path("/NHNHOME/WORKSPACE/0526040018_A/heesu/cag_vision_fm/images")


@dataclass(frozen=True)
class StageSpec:
    name: str
    config: Path
    dataset: str
    default_batch: int
    epochs: int
    epoch_length: int
    eval_period: int
    requires_ckpt: str | None
    min_batch: int
    even_batch: bool

    @property
    def final_iteration(self) -> int:
        return self.epochs * self.epoch_length - 1


STAGES: dict[str, StageSpec] = {
    "stage1": StageSpec(
        name="stage1",
        config=DINO_ROOT / "dinov3" / "configs" / "train" / "dinov3_b200_vit7b16_cag_pretrain.yaml",
        dataset="CAGContrastFM3M:split=TRAIN:root={root}",
        default_batch=16,
        epochs=100,
        epoch_length=1250,
        eval_period=5000,
        requires_ckpt=None,
        min_batch=1,
        even_batch=False,
    ),
    "stage2": StageSpec(
        name="stage2",
        config=DINO_ROOT / "dinov3" / "configs" / "train" / "dinov3_b200_vit7b16_cag_gram_anchor.yaml",
        dataset="CAGContrastFM3M:split=TRAIN:root={root}",
        default_batch=16,
        epochs=100,
        epoch_length=1000,
        eval_period=5000,
        requires_ckpt="stage1",
        min_batch=1,
        even_batch=False,
    ),
    "stage3": StageSpec(
        name="stage3",
        config=DINO_ROOT
        / "dinov3"
        / "configs"
        / "train"
        / "dinov3_b200_vit7b16_cag_high_res_adapt_content_state.yaml",
        dataset="CAGContrastFMContinuityV1:split=TRAIN:root={root}",
        default_batch=8,
        epochs=30,
        epoch_length=1000,
        eval_period=5000,
        requires_ckpt="stage2",
        min_batch=2,
        even_batch=True,
    ),
}


@dataclass(frozen=True)
class Recipe:
    stage: str
    batch_per_gpu: int
    num_workers: int
    pin_memory: bool
    fp8: bool
    compile: bool
    lr_scale: float
    throughput_images_per_sec: float | None
    iter_time_sec: float | None
    max_memory_mib: int | None
    mean_gpu_util: float | None
    checkpointing_full: bool | None = None


def bool_str(value: bool) -> str:
    return "true" if value else "false"


def run_checked(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, check=False)


def preflight(data_root: Path, cuda_visible_devices: str, nproc_per_node: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "dino_root": str(DINO_ROOT),
        "data_root": str(data_root),
        "python": sys.executable,
        "cuda_visible_devices": cuda_visible_devices,
        "nproc_per_node": nproc_per_node,
    }
    torch_check = run_checked(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "print(torch.__version__, torch.version.cuda); "
                "print(torch.cuda.is_available(), torch.cuda.device_count()); "
                "print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
            ),
        ],
        cwd=REPO_ROOT,
    )
    if torch_check.returncode != 0:
        raise RuntimeError(f"PyTorch preflight failed:\n{torch_check.stderr}")
    result["torch_check"] = torch_check.stdout.strip()
    if "(10, 0)" not in torch_check.stdout:
        raise RuntimeError(f"Expected B200 sm_100 capability in torch check, got:\n{torch_check.stdout}")

    nvidia = run_checked(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=REPO_ROOT,
    )
    if nvidia.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed:\n{nvidia.stderr}")
    result["nvidia_smi"] = nvidia.stdout.strip()

    required = [
        data_root / "train",
        data_root / "cache" / "cagcontrastfm3m-relpaths-TRAIN.txt",
        data_root / "cache" / "continuity_v1" / "image_relpaths_train.txt",
        data_root / "cache" / "continuity_v1" / "unique_dicom_keys_train.txt",
        data_root / "cache" / "continuity_v1" / "sample_to_dicom_idx_train.npy",
        data_root / "cache" / "continuity_v1" / "positive_adjacent_indices_train.npy",
        data_root / "cache" / "continuity_v1" / "positive_adjacent_offsets_train.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Dataset preflight failed, missing: {missing}")
    result["dataset_ok"] = True
    return result


class GPUMonitor:
    def __init__(self, gpu_ids: list[int], interval_sec: float = 1.0):
        self.gpu_ids = gpu_ids
        self.interval_sec = interval_sec
        self.max_used = {gpu_id: 0 for gpu_id in gpu_ids}
        self.total = {gpu_id: 0 for gpu_id in gpu_ids}
        self.util_samples: dict[int, list[int]] = {gpu_id: [] for gpu_id in gpu_ids}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=2)
        return {
            "max_used_mib": self.max_used,
            "total_mib": self.total,
            "headroom_mib": {
                gpu_id: (self.total[gpu_id] - self.max_used[gpu_id]) if self.total[gpu_id] else None
                for gpu_id in self.gpu_ids
            },
            "mean_utilization_gpu": {
                gpu_id: (sum(values) / len(values) if values else None)
                for gpu_id, values in self.util_samples.items()
            },
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            query = run_checked(
                [
                    "nvidia-smi",
                    f"--id={','.join(str(gpu_id) for gpu_id in self.gpu_ids)}",
                    "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                cwd=REPO_ROOT,
            )
            if query.returncode == 0:
                for line in query.stdout.strip().splitlines():
                    if not line.strip():
                        continue
                    index_s, used_s, total_s, util_s = [part.strip() for part in line.split(",")]
                    gpu_id = int(index_s)
                    if gpu_id not in self.max_used:
                        continue
                    used = int(used_s)
                    total = int(total_s)
                    util = int(util_s)
                    self.max_used[gpu_id] = max(self.max_used[gpu_id], used)
                    self.total[gpu_id] = total
                    self.util_samples[gpu_id].append(util)
            self._stop.wait(self.interval_sec)


def gpu_ids_from_visible_devices(cuda_visible_devices: str) -> list[int]:
    ids = []
    for value in cuda_visible_devices.split(","):
        value = value.strip()
        if value:
            ids.append(int(value))
    return ids


def make_output_dir(output_root: Path, phase: str, stage: str, suffix: str) -> Path:
    return output_root / phase / stage / suffix


def teacher_ckpt(output_dir: Path, iteration: int) -> Path:
    return output_dir / "eval" / f"training_{iteration}" / "sharded_teacher_checkpoint"


def classify_failure(returncode: int, log_text: str) -> str:
    text = log_text.lower()
    if "loss group size" in text and "must be" in text:
        return "koleo_group_size_invalid"
    if "out of memory" in text or "cuda error: out of memory" in text:
        return "oom"
    if "nan loss detected" in text or "too many consecutive nans" in text:
        return "nan"
    if "torch._dynamo" in text or "compile" in text and returncode != 0:
        return "compile_or_inductor_error"
    if returncode != 0:
        return "runtime_error"
    if "nan loss detected" in text:
        return "nan"
    return "success"


def parse_metrics(output_dir: Path, *, global_batch: int, warmup_records: int) -> dict[str, float | None]:
    metrics_path = output_dir / "training_metrics.json"
    if not metrics_path.exists():
        return {"iter_time_sec": None, "throughput_images_per_sec": None}
    rows = []
    with metrics_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows = rows[warmup_records:] if len(rows) > warmup_records else rows
    iter_times = [float(row["iter_time"]) for row in rows if "iter_time" in row and float(row["iter_time"]) > 0]
    if not iter_times:
        return {"iter_time_sec": None, "throughput_images_per_sec": None}
    avg_iter_time = sum(iter_times) / len(iter_times)
    return {
        "iter_time_sec": avg_iter_time,
        "throughput_images_per_sec": global_batch / avg_iter_time,
    }


def build_overrides(
    *,
    stage: StageSpec,
    data_root: Path,
    batch_per_gpu: int,
    num_workers: int,
    pin_memory: bool,
    fp8: bool,
    compile_enabled: bool,
    checkpointing_full: bool | None,
    lr_scale: float,
    epochs: int,
    epoch_length: int,
    eval_period: int,
    checkpoint_period: int,
    prereq_ckpt: Path | None,
) -> list[str]:
    overrides = [
        f"train.dataset_path={stage.dataset.format(root=data_root)}",
        f"train.batch_size_per_gpu={batch_per_gpu}",
        f"train.num_workers={num_workers}",
        f"train.pin_memory={bool_str(pin_memory)}",
        "train.sharded_eval_checkpoint=true",
        f"train.compile={bool_str(compile_enabled)}",
        f"train.OFFICIAL_EPOCH_LENGTH={epoch_length}",
        f"optim.epochs={epochs}",
        f"evaluation.eval_period_iterations={eval_period}",
        f"checkpointing.period={checkpoint_period}",
        "checkpointing.max_to_keep=1",
        "checkpointing.keep_every=1000000000",
        "student.arch=vit_7b",
        f"student.fp8_enabled={bool_str(fp8)}",
        "student.fp8_filter=blocks",
        "schedules.lr.peak={:.12g}".format(stage_lr_peak(stage.name) * lr_scale),
        "schedules.lr.end={:.12g}".format(stage_lr_end(stage.name) * lr_scale),
        "schedules.lr.warmup_epochs=0",
        "schedules.lr.freeze_last_layer_epochs=0",
        "schedules.lr.cosine_epochs=1",
        "schedules.teacher_temp.warmup_epochs=0",
        "schedules.weight_decay.warmup_epochs=0",
    ]
    if checkpointing_full is not None:
        overrides.append(f"train.checkpointing_full={bool_str(checkpointing_full)}")
    if prereq_ckpt is not None:
        overrides.extend(
            [
                f"gram.ckpt={prereq_ckpt}",
                f"student.resume_from_teacher_chkpt={prereq_ckpt}",
            ]
        )
    if stage.name == "stage3":
        overrides.extend(
            [
                "dino.koleo_loss_distributed=false",
                "dino.koleo_distributed_loss_group_size=null",
                "dino.local_loss_weight_schedule.warmup_epochs=0",
                "dino.local_loss_weight_schedule.cosine_epochs=1",
                "gram.loss_weight_schedule.warmup_epochs=0",
                "gram.loss_weight_schedule.cosine_epochs=1",
                "content_state.head_hidden_dim=4096",
                "content_state.head_out_dim=4096",
                "content_state.decoder_hidden_dim=8192",
            ]
        )
    if stage.name == "stage2":
        overrides.extend(
            [
                "dino.local_loss_weight_schedule.warmup_epochs=0",
                "dino.local_loss_weight_schedule.cosine_epochs=1",
                "gram.loss_weight_schedule.warmup_epochs=0",
                "gram.loss_weight_schedule.cosine_epochs=1",
            ]
        )
    return overrides


def stage_lr_peak(stage: str) -> float:
    return {"stage1": 5.0e-05, "stage2": 3.0e-05, "stage3": 0.0}[stage]


def stage_lr_end(stage: str) -> float:
    return {"stage1": 5.0e-05, "stage2": 3.0e-05, "stage3": 1.25e-05}[stage]


def run_train(
    *,
    stage: StageSpec,
    data_root: Path,
    output_root: Path,
    phase: str,
    suffix: str,
    cuda_visible_devices: str,
    nproc_per_node: int,
    batch_per_gpu: int,
    num_workers: int,
    pin_memory: bool,
    fp8: bool,
    compile_enabled: bool,
    checkpointing_full: bool | None,
    lr_scale: float,
    iters: int,
    eval_period: int,
    checkpoint_period: int,
    prereq_ckpt: Path | None,
    warmup_records: int,
) -> dict[str, Any]:
    output_dir = make_output_dir(output_root, phase, stage.name, suffix)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overrides = build_overrides(
        stage=stage,
        data_root=data_root,
        batch_per_gpu=batch_per_gpu,
        num_workers=num_workers,
        pin_memory=pin_memory,
        fp8=fp8,
        compile_enabled=compile_enabled,
        checkpointing_full=checkpointing_full,
        lr_scale=lr_scale,
        epochs=1,
        epoch_length=iters,
        eval_period=eval_period,
        checkpoint_period=checkpoint_period,
        prereq_ckpt=prereq_ckpt,
    )
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        "-m",
        "dinov3.train.train",
        "--config-file",
        str(stage.config),
        "--output-dir",
        str(output_dir),
        *overrides,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env["PYTHONPATH"] = str(DINO_ROOT)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    # Search launches many short torchrun jobs. Isolating compiler caches per
    # probe avoids stale/cross-rank cache hits from turning into Triton failures.
    inductor_cache_dir = output_dir / "torchinductor_cache"
    triton_cache_dir = inductor_cache_dir / "triton"
    inductor_cache_dir.mkdir(parents=True, exist_ok=True)
    triton_cache_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor_cache_dir))
    env.setdefault("TRITON_CACHE_DIR", str(triton_cache_dir))
    env.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    env.setdefault("TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE", "1")

    log_path = output_dir / "stdout.log"
    monitor = GPUMonitor(gpu_ids_from_visible_devices(cuda_visible_devices))
    monitor.start()
    started = time.time()
    with log_path.open("w") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=DINO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    duration = time.time() - started
    monitor_stats = monitor.stop()
    log_text = log_path.read_text(errors="replace")
    status = classify_failure(proc.returncode, log_text)
    metrics = parse_metrics(
        output_dir,
        global_batch=batch_per_gpu * nproc_per_node,
        warmup_records=warmup_records,
    )
    max_memory = max(
        (value for value in monitor_stats["max_used_mib"].values() if value is not None),
        default=None,
    )
    util_values = [
        value
        for value in monitor_stats["mean_utilization_gpu"].values()
        if value is not None
    ]
    mean_util = sum(util_values) / len(util_values) if util_values else None
    return {
        "stage": stage.name,
        "phase": phase,
        "suffix": suffix,
        "status": status,
        "returncode": proc.returncode,
        "duration_sec": duration,
        "batch_per_gpu": batch_per_gpu,
        "global_batch": batch_per_gpu * nproc_per_node,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "fp8": fp8,
        "compile": compile_enabled,
        "checkpointing_full": checkpointing_full,
        "lr_scale": lr_scale,
        "iters": iters,
        "eval_period": eval_period,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "cmd": cmd,
        "monitor": monitor_stats,
        "max_memory_mib": max_memory,
        "mean_gpu_util": mean_util,
        **metrics,
    }


def is_success(run: dict[str, Any]) -> bool:
    return run["status"] == "success"


def next_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def normalize_batch(value: int, *, even: bool, min_batch: int) -> int:
    value = max(value, min_batch)
    return next_even(value) if even else value


def normalize_batch_down(value: int, *, even: bool, min_batch: int) -> int:
    value = max(value, min_batch)
    if even and value % 2:
        value -= 1
    return max(value, min_batch)


def record(results: dict[str, Any], run: dict[str, Any], output_root: Path) -> None:
    results["runs"].append(run)
    write_json(results, output_root)
    print(
        "[{stage}] {phase}/{suffix}: status={status}, batch={batch_per_gpu}, "
        "workers={num_workers}, pin={pin_memory}, fp8={fp8}, compile={compile}, "
        "ips={throughput_images_per_sec}".format(**run),
        flush=True,
    )


def same_float(left: Any, right: float) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def find_recorded_run(
    results: dict[str, Any],
    *,
    stage: StageSpec,
    phase: str,
    suffix: str,
    batch_per_gpu: int,
    num_workers: int,
    pin_memory: bool,
    fp8: bool,
    compile_enabled: bool,
    checkpointing_full: bool | None,
    lr_scale: float,
    iters: int,
) -> dict[str, Any] | None:
    for run in reversed(results.get("runs", [])):
        if run.get("stage") != stage.name:
            continue
        if run.get("phase") != phase or run.get("suffix") != suffix:
            continue
        if int(run.get("batch_per_gpu", -1)) != batch_per_gpu:
            continue
        if int(run.get("num_workers", -1)) != num_workers:
            continue
        if bool(run.get("pin_memory")) != pin_memory:
            continue
        if bool(run.get("fp8")) != fp8 or bool(run.get("compile")) != compile_enabled:
            continue
        if checkpointing_full is not None and bool(run.get("checkpointing_full")) != checkpointing_full:
            continue
        if checkpointing_full is None and run.get("checkpointing_full") is not None:
            continue
        if int(run.get("iters", -1)) != iters:
            continue
        if not same_float(run.get("lr_scale"), lr_scale):
            continue
        return run
    return None


def reuse_recorded_run(run: dict[str, Any]) -> dict[str, Any]:
    print(
        "[{stage}] {phase}/{suffix}: reuse status={status}, batch={batch_per_gpu}, "
        "workers={num_workers}, pin={pin_memory}, fp8={fp8}, compile={compile}, "
        "ips={throughput_images_per_sec}".format(**run),
        flush=True,
    )
    return run


def write_json(results: dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "results.json").open("w") as f:
        json.dump(results, f, indent=2)


def test_batch(
    *,
    stage: StageSpec,
    data_root: Path,
    output_root: Path,
    cuda_visible_devices: str,
    nproc_per_node: int,
    prereq_ckpt: Path | None,
    batch: int,
    num_workers: int,
    pin_memory: bool,
    fp8: bool,
    compile_enabled: bool,
    checkpointing_full: bool | None,
    fit_iters: int,
    results: dict[str, Any],
) -> dict[str, Any]:
    ckpt_suffix = "" if checkpointing_full is None else f"_ckptfull{int(checkpointing_full)}"
    suffix = f"b{batch}_w{num_workers}_pin{int(pin_memory)}_fp8{int(fp8)}_compile{int(compile_enabled)}{ckpt_suffix}"
    recorded = find_recorded_run(
        results,
        stage=stage,
        phase="fit",
        suffix=suffix,
        batch_per_gpu=batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        fp8=fp8,
        compile_enabled=compile_enabled,
        checkpointing_full=checkpointing_full,
        lr_scale=1.0,
        iters=fit_iters,
    )
    if recorded is not None:
        return reuse_recorded_run(recorded)
    run = run_train(
        stage=stage,
        data_root=data_root,
        output_root=output_root,
        phase="fit",
        suffix=suffix,
        cuda_visible_devices=cuda_visible_devices,
        nproc_per_node=nproc_per_node,
        batch_per_gpu=batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        fp8=fp8,
        compile_enabled=compile_enabled,
        checkpointing_full=checkpointing_full,
        lr_scale=1.0,
        iters=fit_iters,
        eval_period=0,
        checkpoint_period=1000000000,
        prereq_ckpt=prereq_ckpt,
        warmup_records=0,
    )
    record(results, run, output_root)
    return run


def find_max_batch(
    *,
    stage: StageSpec,
    data_root: Path,
    output_root: Path,
    cuda_visible_devices: str,
    nproc_per_node: int,
    prereq_ckpt: Path | None,
    num_workers: int,
    pin_memory: bool,
    fp8: bool,
    compile_enabled: bool,
    checkpointing_full: bool | None,
    fit_iters: int,
    max_batch_per_gpu: int,
    results: dict[str, Any],
) -> tuple[int | None, dict[str, Any] | None]:
    start = normalize_batch(stage.default_batch, even=stage.even_batch, min_batch=stage.min_batch)
    start = min(start, max_batch_per_gpu)
    lower_success: tuple[int, dict[str, Any]] | None = None
    first = test_batch(
        stage=stage,
        data_root=data_root,
        output_root=output_root,
        cuda_visible_devices=cuda_visible_devices,
        nproc_per_node=nproc_per_node,
        prereq_ckpt=prereq_ckpt,
        batch=start,
        num_workers=num_workers,
        pin_memory=pin_memory,
        fp8=fp8,
        compile_enabled=compile_enabled,
        checkpointing_full=checkpointing_full,
        fit_iters=fit_iters,
        results=results,
    )
    if is_success(first):
        lower_success = (start, first)
        probe = start * 2
        while probe <= max_batch_per_gpu:
            probe = normalize_batch(probe, even=stage.even_batch, min_batch=stage.min_batch)
            run = test_batch(
                stage=stage,
                data_root=data_root,
                output_root=output_root,
                cuda_visible_devices=cuda_visible_devices,
                nproc_per_node=nproc_per_node,
                prereq_ckpt=prereq_ckpt,
                batch=probe,
                num_workers=num_workers,
                pin_memory=pin_memory,
                fp8=fp8,
                compile_enabled=compile_enabled,
                checkpointing_full=checkpointing_full,
                fit_iters=fit_iters,
                results=results,
            )
            if is_success(run):
                lower_success = (probe, run)
                probe *= 2
                continue
            upper_failure = probe
            break
        else:
            return lower_success
    else:
        upper_failure = start
        probe = start // 2
        if stage.even_batch:
            probe = probe - (probe % 2)
        while probe >= stage.min_batch:
            run = test_batch(
                stage=stage,
                data_root=data_root,
                output_root=output_root,
                cuda_visible_devices=cuda_visible_devices,
                nproc_per_node=nproc_per_node,
                prereq_ckpt=prereq_ckpt,
                batch=probe,
                num_workers=num_workers,
                pin_memory=pin_memory,
                fp8=fp8,
                compile_enabled=compile_enabled,
                checkpointing_full=checkpointing_full,
                fit_iters=fit_iters,
                results=results,
            )
            if is_success(run):
                lower_success = (probe, run)
                break
            upper_failure = probe
            probe //= 2
            if stage.even_batch:
                probe = probe - (probe % 2)
        if lower_success is None:
            return None, None

    assert lower_success is not None
    low_batch, low_run = lower_success
    high_batch = upper_failure
    step = 2 if stage.even_batch else 1
    while high_batch - low_batch > step:
        mid = (low_batch + high_batch) // 2
        if stage.even_batch:
            mid = mid - (mid % 2)
            if mid <= low_batch:
                mid = low_batch + 2
        run = test_batch(
            stage=stage,
            data_root=data_root,
            output_root=output_root,
            cuda_visible_devices=cuda_visible_devices,
            nproc_per_node=nproc_per_node,
            prereq_ckpt=prereq_ckpt,
            batch=mid,
            num_workers=num_workers,
            pin_memory=pin_memory,
            fp8=fp8,
            compile_enabled=compile_enabled,
            checkpointing_full=checkpointing_full,
            fit_iters=fit_iters,
            results=results,
        )
        if is_success(run):
            low_batch, low_run = mid, run
        else:
            high_batch = mid
    return low_batch, low_run


def worker_candidates(max_workers: int) -> list[int]:
    candidates = {0, 1, max_workers}
    value = 2
    while value <= max_workers:
        candidates.add(value)
        value *= 2
    return sorted(candidates)


def benchmark_io_and_mode(
    *,
    stage: StageSpec,
    data_root: Path,
    output_root: Path,
    cuda_visible_devices: str,
    nproc_per_node: int,
    prereq_ckpt: Path | None,
    batch: int,
    fp8: bool,
    compile_enabled: bool,
    checkpointing_full: bool | None,
    max_workers: int,
    benchmark_iters: int,
    warmup_records: int,
    results: dict[str, Any],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    worker0_ooms = 0
    for workers in worker_candidates(max_workers):
        for pin_memory in (False, True):
            ckpt_suffix = "" if checkpointing_full is None else f"_ckptfull{int(checkpointing_full)}"
            suffix = f"b{batch}_w{workers}_pin{int(pin_memory)}_fp8{int(fp8)}_compile{int(compile_enabled)}{ckpt_suffix}"
            run = find_recorded_run(
                results,
                stage=stage,
                phase="benchmark",
                suffix=suffix,
                batch_per_gpu=batch,
                num_workers=workers,
                pin_memory=pin_memory,
                fp8=fp8,
                compile_enabled=compile_enabled,
                checkpointing_full=checkpointing_full,
                lr_scale=1.0,
                iters=benchmark_iters,
            )
            if run is not None:
                run = reuse_recorded_run(run)
            else:
                run = run_train(
                    stage=stage,
                    data_root=data_root,
                    output_root=output_root,
                    phase="benchmark",
                    suffix=suffix,
                    cuda_visible_devices=cuda_visible_devices,
                    nproc_per_node=nproc_per_node,
                    batch_per_gpu=batch,
                    num_workers=workers,
                    pin_memory=pin_memory,
                    fp8=fp8,
                    compile_enabled=compile_enabled,
                    checkpointing_full=checkpointing_full,
                    lr_scale=1.0,
                    iters=benchmark_iters,
                    eval_period=0,
                    checkpoint_period=1000000000,
                    prereq_ckpt=prereq_ckpt,
                    warmup_records=warmup_records,
                )
                record(results, run, output_root)
            if workers == 0 and run["status"] == "oom":
                worker0_ooms += 1
            if not is_success(run):
                if workers == 0 and pin_memory and worker0_ooms == 2:
                    raise RuntimeError(
                        f"No successful worker/pin benchmark for {stage.name} batch={batch}; "
                        "worker=0 pin false/true both OOM, treating batch as GPU-memory bound"
                    )
                continue
            if best is None or (run.get("throughput_images_per_sec") or 0) > (best.get("throughput_images_per_sec") or 0):
                best = run
    if best is None:
        raise RuntimeError(f"No successful worker/pin benchmark for {stage.name} batch={batch}")
    return best


def stabilize(
    *,
    stage: StageSpec,
    data_root: Path,
    output_root: Path,
    cuda_visible_devices: str,
    nproc_per_node: int,
    prereq_ckpt: Path | None,
    base_run: dict[str, Any],
    stability_iters: int,
    warmup_records: int,
    results: dict[str, Any],
) -> dict[str, Any]:
    batch = int(base_run["batch_per_gpu"])
    lr_scales = [1.0, 0.5, 0.25]
    while batch >= stage.min_batch:
        if stage.even_batch:
            batch = batch - (batch % 2)
        for lr_scale in lr_scales:
            ckpt_suffix = ""
            if base_run.get("checkpointing_full") is not None:
                ckpt_suffix = f"_ckptfull{int(bool(base_run['checkpointing_full']))}"
            suffix = (
                f"b{batch}_w{base_run['num_workers']}_pin{int(base_run['pin_memory'])}"
                f"_fp8{int(base_run['fp8'])}_compile{int(base_run['compile'])}{ckpt_suffix}_lr{lr_scale}"
            )
            run = find_recorded_run(
                results,
                stage=stage,
                phase="stability",
                suffix=suffix,
                batch_per_gpu=batch,
                num_workers=int(base_run["num_workers"]),
                pin_memory=bool(base_run["pin_memory"]),
                fp8=bool(base_run["fp8"]),
                compile_enabled=bool(base_run["compile"]),
                checkpointing_full=base_run.get("checkpointing_full"),
                lr_scale=lr_scale,
                iters=stability_iters,
            )
            if run is not None:
                run = reuse_recorded_run(run)
            else:
                run = run_train(
                    stage=stage,
                    data_root=data_root,
                    output_root=output_root,
                    phase="stability",
                    suffix=suffix,
                    cuda_visible_devices=cuda_visible_devices,
                    nproc_per_node=nproc_per_node,
                    batch_per_gpu=batch,
                    num_workers=int(base_run["num_workers"]),
                    pin_memory=bool(base_run["pin_memory"]),
                    fp8=bool(base_run["fp8"]),
                    compile_enabled=bool(base_run["compile"]),
                    checkpointing_full=base_run.get("checkpointing_full"),
                    lr_scale=lr_scale,
                    iters=stability_iters,
                    eval_period=0,
                    checkpoint_period=1000000000,
                    prereq_ckpt=prereq_ckpt,
                    warmup_records=warmup_records,
                )
                record(results, run, output_root)
            if is_success(run):
                return run
            if run["status"] == "nan":
                continue
            break
        batch = batch - (2 if stage.even_batch else 1)
    raise RuntimeError(f"No stable recipe found for {stage.name}")


def create_bootstrap_ckpt(
    *,
    recipe: Recipe,
    stage: StageSpec,
    data_root: Path,
    output_root: Path,
    cuda_visible_devices: str,
    nproc_per_node: int,
    prereq_ckpt: Path | None,
    results: dict[str, Any],
) -> Path:
    bootstrap_batch = min(recipe.batch_per_gpu, stage.default_batch)
    bootstrap_batch = normalize_batch(bootstrap_batch, even=stage.even_batch, min_batch=stage.min_batch)
    run = run_train(
        stage=stage,
        data_root=data_root,
        output_root=output_root,
        phase="bootstrap",
        suffix=f"{stage.name}_selected",
        cuda_visible_devices=cuda_visible_devices,
        nproc_per_node=nproc_per_node,
        batch_per_gpu=bootstrap_batch,
        num_workers=0,
        pin_memory=False,
        fp8=recipe.fp8,
        compile_enabled=False,
        checkpointing_full=recipe.checkpointing_full,
        lr_scale=recipe.lr_scale,
        iters=1,
        eval_period=1,
        checkpoint_period=1000000000,
        prereq_ckpt=prereq_ckpt,
        warmup_records=0,
    )
    record(results, run, output_root)
    ckpt = teacher_ckpt(Path(run["output_dir"]), 0)
    if not is_success(run) or not ckpt.exists():
        raise RuntimeError(f"Bootstrap checkpoint was not created for {stage.name}: {ckpt}")
    return ckpt


def benchmark_with_batch_backoff(
    *,
    stage: StageSpec,
    data_root: Path,
    output_root: Path,
    cuda_visible_devices: str,
    nproc_per_node: int,
    prereq_ckpt: Path | None,
    start_batch: int,
    fp8: bool,
    compile_enabled: bool,
    checkpointing_full: bool | None,
    max_workers: int,
    benchmark_iters: int,
    warmup_records: int,
    results: dict[str, Any],
) -> dict[str, Any]:
    step = 2 if stage.even_batch else 1
    start_batch = normalize_batch_down(start_batch, even=stage.even_batch, min_batch=stage.min_batch)

    def try_benchmark(batch: int) -> dict[str, Any] | None:
        try:
            return benchmark_io_and_mode(
                stage=stage,
                data_root=data_root,
                output_root=output_root,
                cuda_visible_devices=cuda_visible_devices,
                nproc_per_node=nproc_per_node,
                prereq_ckpt=prereq_ckpt,
                batch=batch,
                fp8=fp8,
                compile_enabled=compile_enabled,
                checkpointing_full=checkpointing_full,
                max_workers=max_workers,
                benchmark_iters=benchmark_iters,
                warmup_records=warmup_records,
                results=results,
            )
        except RuntimeError as exc:
            failure = {
                "stage": stage.name,
                "phase": "benchmark_backoff",
                "batch_per_gpu": batch,
                "fp8": fp8,
                "compile": compile_enabled,
                "checkpointing_full": checkpointing_full,
                "reason": str(exc),
            }
            results.setdefault("benchmark_backoffs", []).append(failure)
            write_json(results, output_root)
            return None

    start_result = try_benchmark(start_batch)
    if start_result is not None:
        return start_result

    high_failure = start_batch
    probe = normalize_batch_down(start_batch // 2, even=stage.even_batch, min_batch=stage.min_batch)
    low_success: tuple[int, dict[str, Any]] | None = None
    while probe >= stage.min_batch:
        result = try_benchmark(probe)
        if result is not None:
            low_success = (probe, result)
            break
        high_failure = probe
        if probe == stage.min_batch:
            break
        next_probe = normalize_batch_down(probe // 2, even=stage.even_batch, min_batch=stage.min_batch)
        if next_probe >= probe:
            next_probe = probe - step
        probe = max(stage.min_batch, next_probe)

    if low_success is None:
        raise RuntimeError(f"No successful worker/pin benchmark for {stage.name} at or below batch={start_batch}")

    low_batch, low_run = low_success
    while high_failure - low_batch > step:
        mid = (low_batch + high_failure) // 2
        mid = normalize_batch_down(mid, even=stage.even_batch, min_batch=stage.min_batch)
        if mid <= low_batch:
            mid = low_batch + step
        result = try_benchmark(mid)
        if result is not None:
            low_batch, low_run = mid, result
        else:
            high_failure = mid
    return low_run


def search_stage(
    *,
    stage: StageSpec,
    data_root: Path,
    output_root: Path,
    cuda_visible_devices: str,
    nproc_per_node: int,
    prereq_ckpt: Path | None,
    initial_workers: int,
    max_workers: int,
    fit_iters: int,
    benchmark_iters: int,
    stability_iters: int,
    warmup_records: int,
    max_batch_per_gpu: int,
    results: dict[str, Any],
) -> Recipe:
    precision_compile_modes = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]
    checkpointing_full_candidates: list[bool | None] = [False, True] if stage.name == "stage1" else [None]
    mode_failures: list[dict[str, Any]] = []
    for fp8, compile_enabled in precision_compile_modes:
        stable_recipes: list[Recipe] = []
        for checkpointing_full in checkpointing_full_candidates:
            batch, run = find_max_batch(
                stage=stage,
                data_root=data_root,
                output_root=output_root,
                cuda_visible_devices=cuda_visible_devices,
                nproc_per_node=nproc_per_node,
                prereq_ckpt=prereq_ckpt,
                num_workers=initial_workers,
                pin_memory=False,
                fp8=fp8,
                compile_enabled=compile_enabled,
                checkpointing_full=checkpointing_full,
                fit_iters=fit_iters,
                max_batch_per_gpu=max_batch_per_gpu,
                results=results,
            )
            if batch is None or run is None:
                mode_failures.append(
                    {
                        "stage": stage.name,
                        "fp8": fp8,
                        "compile": compile_enabled,
                        "checkpointing_full": checkpointing_full,
                        "reason": "no_fitting_batch",
                    }
                )
                continue
            try:
                benchmark = benchmark_with_batch_backoff(
                    stage=stage,
                    data_root=data_root,
                    output_root=output_root,
                    cuda_visible_devices=cuda_visible_devices,
                    nproc_per_node=nproc_per_node,
                    prereq_ckpt=prereq_ckpt,
                    start_batch=int(run["batch_per_gpu"]),
                    fp8=bool(run["fp8"]),
                    compile_enabled=bool(run["compile"]),
                    checkpointing_full=run.get("checkpointing_full"),
                    max_workers=max_workers,
                    benchmark_iters=benchmark_iters,
                    warmup_records=warmup_records,
                    results=results,
                )
                stable = stabilize(
                    stage=stage,
                    data_root=data_root,
                    output_root=output_root,
                    cuda_visible_devices=cuda_visible_devices,
                    nproc_per_node=nproc_per_node,
                    prereq_ckpt=prereq_ckpt,
                    base_run=benchmark,
                    stability_iters=stability_iters,
                    warmup_records=warmup_records,
                    results=results,
                )
            except RuntimeError as exc:
                failure = {
                    "stage": stage.name,
                    "fp8": fp8,
                    "compile": compile_enabled,
                    "checkpointing_full": checkpointing_full,
                    "reason": str(exc),
                }
                mode_failures.append(failure)
                results.setdefault("mode_failures", []).append(failure)
                write_json(results, output_root)
                continue
            stable_recipes.append(
                Recipe(
                    stage=stage.name,
                    batch_per_gpu=int(stable["batch_per_gpu"]),
                    num_workers=int(stable["num_workers"]),
                    pin_memory=bool(stable["pin_memory"]),
                    fp8=bool(stable["fp8"]),
                    compile=bool(stable["compile"]),
                    lr_scale=float(stable["lr_scale"]),
                    throughput_images_per_sec=stable.get("throughput_images_per_sec"),
                    iter_time_sec=stable.get("iter_time_sec"),
                    max_memory_mib=stable.get("max_memory_mib"),
                    mean_gpu_util=stable.get("mean_gpu_util"),
                    checkpointing_full=stable.get("checkpointing_full"),
                )
            )
        if stable_recipes:
            return max(stable_recipes, key=lambda recipe: recipe.throughput_images_per_sec or 0)
    raise RuntimeError(f"No stable recipe found for {stage.name}: {mode_failures}")


def write_recipe_env(recipes: dict[str, Recipe], path: Path) -> None:
    lines = [
        "# Generated by B200 ViT-7B recipe search.",
        "# Source this file through dinov3/launchers/b200/common.sh.",
    ]
    for stage in ("stage1", "stage2", "stage3"):
        if stage not in recipes:
            continue
        recipe = recipes[stage]
        prefix = stage.upper()
        lines.extend(
            [
                f"{prefix}_BATCH_PER_GPU={recipe.batch_per_gpu}",
                f"{prefix}_NUM_WORKERS={recipe.num_workers}",
                f"{prefix}_PIN_MEMORY={bool_str(recipe.pin_memory)}",
                f"{prefix}_FP8={bool_str(recipe.fp8)}",
                f"{prefix}_COMPILE={bool_str(recipe.compile)}",
            ]
        )
        if recipe.checkpointing_full is not None:
            lines.append(f"{prefix}_CHECKPOINTING_FULL={bool_str(recipe.checkpointing_full)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def shell_command_for_stage(stage: str) -> str:
    return f"MODEL=vit7b bash dinov3/launchers/b200/run_{ {'stage1': '1_pretraining', 'stage2': '2_gram_anchor', 'stage3': '3_high-res-adapt_content_state'}[stage] }.sh"


def write_markdown(results: dict[str, Any], recipes: dict[str, Recipe], doc_path: Path, recipe_env_path: Path) -> None:
    lines = [
        "# B200 ViT-7B CAG-Content-State Recipe Search",
        "",
        "Generated by `dinov3/launchers/b200/recipe_search/analysis/search_b200_vit7b_recipe.py`.",
        "",
        "## Environment",
        "",
        f"- Data root: `{results['preflight']['data_root']}`",
        f"- CUDA visible devices: `{results['preflight']['cuda_visible_devices']}`",
        f"- nproc per node: `{results['preflight']['nproc_per_node']}`",
        f"- PyTorch: `{results['preflight']['torch_check']}`",
        f"- Recipe env: `{recipe_env_path}`",
        "",
        "## Selected Recipes",
        "",
        "| Stage | Batch/GPU | Global Batch | Workers | Pin Memory | FP8 | Compile | Checkpointing Full | LR Scale | Images/s | Iter Time | Max Mem MiB | Mean GPU Util |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage in ("stage1", "stage2", "stage3"):
        recipe = recipes.get(stage)
        if recipe is None:
            lines.append(f"| {stage} | unresolved | unresolved | unresolved | unresolved | unresolved | unresolved | unresolved | unresolved | unresolved | unresolved | unresolved | unresolved |")
            continue
        global_batch = recipe.batch_per_gpu * int(results["preflight"]["nproc_per_node"])
        lines.append(
            "| {stage} | {batch} | {global_batch} | {workers} | {pin} | {fp8} | {compile} | {checkpointing_full} | {lr_scale:.3g} | {ips} | {iter_time} | {mem} | {util} |".format(
                stage=stage,
                batch=recipe.batch_per_gpu,
                global_batch=global_batch,
                workers=recipe.num_workers,
                pin=recipe.pin_memory,
                fp8=recipe.fp8,
                compile=recipe.compile,
                checkpointing_full=recipe.checkpointing_full if recipe.checkpointing_full is not None else "config",
                lr_scale=recipe.lr_scale,
                ips=f"{recipe.throughput_images_per_sec:.3f}" if recipe.throughput_images_per_sec else "n/a",
                iter_time=f"{recipe.iter_time_sec:.3f}" if recipe.iter_time_sec else "n/a",
                mem=recipe.max_memory_mib if recipe.max_memory_mib is not None else "n/a",
                util=f"{recipe.mean_gpu_util:.1f}" if recipe.mean_gpu_util is not None else "n/a",
            )
        )
    lines.extend(
        [
            "",
            "## Launch Commands",
            "",
            "```bash",
            "cd /home/gmail_asse/Workspace/dinov3_stack",
            "MODEL=vit7b bash dinov3/launchers/b200/run_1_pretraining.sh",
            "MODEL=vit7b bash dinov3/launchers/b200/run_2_gram_anchor.sh",
            "MODEL=vit7b bash dinov3/launchers/b200/run_3_high-res-adapt_content_state.sh",
            "```",
            "",
            "## Failed Runs",
            "",
        ]
    )
    failed = [run for run in results["runs"] if run["status"] != "success"]
    if not failed:
        lines.append("- None")
    else:
        for run in failed[-50:]:
            lines.append(
                "- `{stage}` `{phase}` `b{batch_per_gpu}` `w{num_workers}` "
                "`pin={pin_memory}` `fp8={fp8}` `compile={compile}` -> `{status}`".format(**run)
            )
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive B200 x8 ViT-7B CAG-Content-State recipe search.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--recipe-env-path", type=Path, default=DEFAULT_RECIPE_ENV_PATH)
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--stages", nargs="+", choices=("stage1", "stage2", "stage3"), default=("stage1", "stage2", "stage3"))
    parser.add_argument("--fit-iters", type=int, default=2)
    parser.add_argument("--benchmark-iters", type=int, default=20)
    parser.add_argument("--stability-iters", type=int, default=30)
    parser.add_argument("--warmup-records", type=int, default=1)
    parser.add_argument("--initial-workers", type=int, default=None)
    parser.add_argument("--max-workers-per-rank", type=int, default=None)
    parser.add_argument("--max-batch-per-gpu", type=int, default=512)
    parser.add_argument("--resume-results", action="store_true", help="Resume from an existing results.json in the output root.")
    args = parser.parse_args()

    args.data_root = args.data_root.expanduser().resolve()
    if not args.output_root.is_absolute():
        args.output_root = (REPO_ROOT / args.output_root).resolve()
    else:
        args.output_root = args.output_root.expanduser().resolve()
    if not args.doc_path.is_absolute():
        args.doc_path = (REPO_ROOT / args.doc_path).resolve()
    else:
        args.doc_path = args.doc_path.expanduser().resolve()
    if not args.recipe_env_path.is_absolute():
        args.recipe_env_path = (REPO_ROOT / args.recipe_env_path).resolve()
    else:
        args.recipe_env_path = args.recipe_env_path.expanduser().resolve()

    cpu_count = os.cpu_count() or 8
    inferred_max_workers = max(1, min(8, cpu_count // max(1, args.nproc_per_node)))
    max_workers = args.max_workers_per_rank if args.max_workers_per_rank is not None else inferred_max_workers
    initial_workers = args.initial_workers if args.initial_workers is not None else min(4, max_workers)

    results_path = args.output_root / "results.json"
    if args.resume_results and results_path.exists():
        results = json.loads(results_path.read_text())
        results.setdefault("runs", [])
        results.setdefault("recipes", {})
        results["resumed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        results = {
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runs": [],
            "recipes": {},
        }
    results["preflight"] = preflight(args.data_root, args.cuda_visible_devices, args.nproc_per_node)
    write_json(results, args.output_root)

    recipes: dict[str, Recipe] = {
        stage_name: Recipe(**recipe_data)
        for stage_name, recipe_data in results.get("recipes", {}).items()
    }
    bootstrap_ckpts: dict[str, Path] = {
        stage_name: Path(ckpt_path)
        for stage_name, ckpt_path in results.get("bootstrap_ckpts", {}).items()
    }
    for stage_name in args.stages:
        stage = STAGES[stage_name]
        prereq_ckpt = bootstrap_ckpts.get(stage.requires_ckpt) if stage.requires_ckpt else None
        if stage.requires_ckpt and prereq_ckpt is None:
            raise RuntimeError(f"{stage_name} requires bootstrap checkpoint from {stage.requires_ckpt}")
        if stage_name in recipes:
            recipe = recipes[stage_name]
        else:
            recipe = search_stage(
                stage=stage,
                data_root=args.data_root,
                output_root=args.output_root,
                cuda_visible_devices=args.cuda_visible_devices,
                nproc_per_node=args.nproc_per_node,
                prereq_ckpt=prereq_ckpt,
                initial_workers=initial_workers,
                max_workers=max_workers,
                fit_iters=args.fit_iters,
                benchmark_iters=args.benchmark_iters,
                stability_iters=args.stability_iters,
                warmup_records=args.warmup_records,
                max_batch_per_gpu=args.max_batch_per_gpu,
                results=results,
            )
            recipes[stage_name] = recipe
            results["recipes"][stage_name] = asdict(recipe)
            write_json(results, args.output_root)
        if stage_name != "stage3":
            existing_ckpt = bootstrap_ckpts.get(stage_name)
            if existing_ckpt is None or not existing_ckpt.exists():
                bootstrap_ckpts[stage_name] = create_bootstrap_ckpt(
                    recipe=recipe,
                    stage=stage,
                    data_root=args.data_root,
                    output_root=args.output_root,
                    cuda_visible_devices=args.cuda_visible_devices,
                    nproc_per_node=args.nproc_per_node,
                    prereq_ckpt=prereq_ckpt,
                    results=results,
                )
            results.setdefault("bootstrap_ckpts", {})[stage_name] = str(bootstrap_ckpts[stage_name])
            write_json(results, args.output_root)

    write_recipe_env(recipes, args.recipe_env_path)
    write_markdown(results, recipes, args.doc_path, args.recipe_env_path)
    write_json(results, args.output_root)
    print(f"Recipe search complete. Results: {args.output_root / 'results.json'}")
    print(f"Markdown summary: {args.doc_path}")
    print(f"Recipe env: {args.recipe_env_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
