#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset

_CACHE_SUBDIR = "cache/continuity_v1"


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def dirname(self) -> str:
        return self.value


def _read_lines(path: str) -> List[str]:
    lines: List[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                lines.append(line)
    return lines


def _parse_dicom_key(dicom_key: str) -> Dict[str, Any]:
    parts = dicom_key.split("/")
    if len(parts) != 4 or parts[2] != "XA" or not parts[3].endswith(".dcm"):
        raise RuntimeError(f"unexpected dicom key format: {dicom_key}")
    return {
        "patient_id": parts[0],
        "study_date": parts[1],
        "series_no": int(os.path.splitext(parts[3])[0]),
    }


class CAGContrastFMContinuityV1(ExtendedVisionDataset):
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "CAGContrastFMContinuityV1.Split",
        root: str,
        extra: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_decoder: Callable = ImageDataDecoder,
        target_decoder: Callable = TargetDecoder,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=image_decoder,
            target_decoder=target_decoder,
        )
        self._split = split
        self._cache_root = extra or os.path.join(root, _CACHE_SUBDIR)

        relpaths_path = os.path.join(self._cache_root, f"image_relpaths_{split.value}.txt")
        unique_dicom_keys_path = os.path.join(self._cache_root, f"unique_dicom_keys_{split.value}.txt")
        sample_to_dicom_idx_path = os.path.join(self._cache_root, f"sample_to_dicom_idx_{split.value}.npy")
        frame_idx_path = os.path.join(self._cache_root, f"frame_idx_{split.value}.npy")
        adj_offsets_path = os.path.join(self._cache_root, f"positive_adjacent_offsets_{split.value}.npy")
        adj_indices_path = os.path.join(self._cache_root, f"positive_adjacent_indices_{split.value}.npy")
        near_offsets_path = os.path.join(self._cache_root, f"positive_nearby_offsets_{split.value}.npy")
        near_indices_path = os.path.join(self._cache_root, f"positive_nearby_indices_{split.value}.npy")

        required_paths = [
            relpaths_path,
            unique_dicom_keys_path,
            sample_to_dicom_idx_path,
            frame_idx_path,
            adj_offsets_path,
            adj_indices_path,
            near_offsets_path,
            near_indices_path,
        ]
        missing = [path for path in required_paths if not os.path.isfile(path)]
        if missing:
            raise RuntimeError(
                "continuity cache is incomplete; missing files under "
                f'"{self._cache_root}": {missing[:5]}{"..." if len(missing) > 5 else ""}'
            )

        self._image_paths = _read_lines(relpaths_path)
        self._unique_dicom_keys = _read_lines(unique_dicom_keys_path)
        self._sample_to_dicom_idx = np.load(sample_to_dicom_idx_path, mmap_mode=None, allow_pickle=False)
        self._frame_idx = np.load(frame_idx_path, mmap_mode=None, allow_pickle=False)
        self._adj_offsets = np.load(adj_offsets_path, mmap_mode=None, allow_pickle=False)
        self._adj_indices = np.load(adj_indices_path, mmap_mode=None, allow_pickle=False)
        self._near_offsets = np.load(near_offsets_path, mmap_mode=None, allow_pickle=False)
        self._near_indices = np.load(near_indices_path, mmap_mode=None, allow_pickle=False)
        self._valid_anchor_indices = {
            "adjacent": np.flatnonzero(np.diff(self._adj_offsets) > 0).astype(np.int64, copy=False),
            "nearby": np.flatnonzero(np.diff(self._near_offsets) > 0).astype(np.int64, copy=False),
        }

        sample_count = len(self._image_paths)
        if len(self._sample_to_dicom_idx) != sample_count or len(self._frame_idx) != sample_count:
            raise RuntimeError("continuity cache sample arrays do not match image relpath count")
        if len(self._adj_offsets) != sample_count + 1 or len(self._near_offsets) != sample_count + 1:
            raise RuntimeError("continuity cache offset arrays do not match sample count")

    @property
    def split(self) -> "CAGContrastFMContinuityV1.Split":
        return self._split

    @property
    def cache_root(self) -> str:
        return self._cache_root

    def get_image_relpath(self, index: int) -> str:
        return self._image_paths[index]

    def get_image_data(self, index: int) -> bytes:
        image_relpath = self._image_paths[index]
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        return None

    def get_dicom_key(self, index: int) -> str:
        dicom_idx = int(self._sample_to_dicom_idx[index])
        return self._unique_dicom_keys[dicom_idx]

    def get_frame_idx(self, index: int) -> int:
        return int(self._frame_idx[index])

    def get_positive_indices(self, index: int, mode: str = "adjacent") -> List[int]:
        if mode in ("adjacent", "strict_adjacent"):
            offsets = self._adj_offsets
            indices = self._adj_indices
        elif mode == "nearby":
            offsets = self._near_offsets
            indices = self._near_indices
        else:
            raise ValueError(f"unsupported continuity mode: {mode}")

        start = int(offsets[index])
        end = int(offsets[index + 1])
        return indices[start:end].astype(np.int64, copy=False).tolist()

    def get_valid_anchor_indices(self, mode: str = "adjacent") -> List[int]:
        if mode in ("adjacent", "strict_adjacent"):
            key = "adjacent"
        elif mode == "nearby":
            key = "nearby"
        else:
            raise ValueError(f"unsupported continuity mode: {mode}")
        return self._valid_anchor_indices[key].tolist()

    def get_sample_meta(self, index: int) -> Dict[str, Any]:
        image_relpath = self.get_image_relpath(index)
        dicom_key = self.get_dicom_key(index)
        parsed = _parse_dicom_key(dicom_key)
        relpath_parts = image_relpath.split("/")
        frame_loc = relpath_parts[1] if len(relpath_parts) >= 3 else ""
        return {
            "sample_index": int(index),
            "split": self._split.value,
            "image_relpath": image_relpath,
            "patient_id": parsed["patient_id"],
            "study_date": parsed["study_date"],
            "series_no": parsed["series_no"],
            "frame_loc": frame_loc,
            "frame_idx": self.get_frame_idx(index),
            "dicom_key": dicom_key,
            "dicom_relpath": dicom_key,
        }

    def __len__(self) -> int:
        return len(self._image_paths)
