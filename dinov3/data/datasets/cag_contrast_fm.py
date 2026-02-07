#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import os
from enum import Enum
from typing import Any, Callable, List, Optional, Union

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov3")

_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
}
_CACHE_DIRNAME = "cache"


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def dirname(self) -> str:
        return self.value


def _is_image_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() in _IMAGE_EXTENSIONS


def _gather_image_relpaths(root: str, split_dirname: str) -> List[str]:
    split_root = os.path.join(root, split_dirname)
    if not os.path.isdir(split_root):
        raise RuntimeError(f'split directory not found: "{split_root}"')

    relpaths: List[str] = []
    for dirpath, _, filenames in os.walk(split_root):
        for name in filenames:
            if not _is_image_file(name):
                continue
            full_path = os.path.join(dirpath, name)
            relpaths.append(os.path.relpath(full_path, root))

    relpaths.sort()
    if not relpaths:
        raise RuntimeError(f'no image files found under "{split_root}"')

    return relpaths


def _cache_relpaths_path(root: str, split: _Split) -> str:
    cache_root = os.path.join(root, _CACHE_DIRNAME)
    return os.path.join(cache_root, f"cagcontrastfm3m-relpaths-{split.value.upper()}.txt")


def _load_cached_relpaths(cache_path: str) -> List[str]:
    relpaths: List[str] = []
    with open(cache_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            relpaths.append(line)
    if not relpaths:
        raise RuntimeError(f'cache file is empty: "{cache_path}"')
    return relpaths


def _write_cached_relpaths(cache_path: str, relpaths: List[str]) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "w") as f:
        for relpath in relpaths:
            f.write(relpath)
            f.write("\n")
    os.replace(tmp_path, cache_path)


class CAGContrastFM3M(ExtendedVisionDataset):
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "CAGContrastFM3M.Split",
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
        self._extra_root = extra

        cache_path = _cache_relpaths_path(root, split)
        if os.path.isfile(cache_path):
            logger.info(f'loading cached index: "{cache_path}"')
            self._image_paths = _load_cached_relpaths(cache_path)
        else:
            logger.info(f'building index for split "{split.value}" under "{root}"')
            self._image_paths = _gather_image_relpaths(root, split.dirname)
            logger.info(f'writing cached index: "{cache_path}"')
            _write_cached_relpaths(cache_path, self._image_paths)

    @property
    def split(self) -> "CAGContrastFM3M.Split":
        return self._split

    def get_image_relpath(self, index: int) -> str:
        return self._image_paths[index]

    def get_image_data(self, index: int) -> bytes:
        image_relpath = self._image_paths[index]
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Any:
        return None

    def __len__(self) -> int:
        return len(self._image_paths)

    def dump_extra(self, extra: Optional[str] = None) -> None:
        cache_path = _cache_relpaths_path(self.root, self._split)
        relpaths = _gather_image_relpaths(self.root, self._split.dirname)
        _write_cached_relpaths(cache_path, relpaths)
