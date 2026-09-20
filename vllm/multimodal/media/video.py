# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import json
import struct
import threading
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from vllm import envs
from vllm.logger import init_logger

from ..video import VIDEO_LOADER_REGISTRY
from .base import MediaIO
from .image import ImageMediaIO

logger = init_logger(__name__)

_VideoDecodeCacheKey = tuple[str, int, int, int, str, tuple[tuple[str, Any], ...]]
_VideoDecodeCacheValue = tuple[npt.NDArray, dict[str, Any]]
_LMMS_VIDEO_JPEG_MAGIC = b"LMMSVJPG1\n"


class _InflightVideoDecode:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: _VideoDecodeCacheValue | None = None
        self.error: BaseException | None = None


class _VideoDecodeCache:
    def __init__(self, max_size: int = envs.VLLM_VIDEO_DECODE_CACHE_SIZE) -> None:
        self.lock = threading.Lock()
        self.cache: OrderedDict[_VideoDecodeCacheKey, _VideoDecodeCacheValue] = (
            OrderedDict()
        )
        self.inflight: dict[_VideoDecodeCacheKey, _InflightVideoDecode] = {}
        self.max_size: int = max_size

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.inflight.clear()

    @classmethod
    def _freeze_value(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(
                sorted((str(k), cls._freeze_value(v)) for k, v in value.items())
            )
        if isinstance(value, (list, tuple)):
            return tuple(cls._freeze_value(v) for v in value)
        if isinstance(value, set):
            return tuple(sorted(cls._freeze_value(v) for v in value))
        try:
            hash(value)
        except TypeError:
            return repr(value)
        return value

    @staticmethod
    def _copy_value(value: _VideoDecodeCacheValue) -> _VideoDecodeCacheValue:
        frames, metadata = value
        return frames.copy(), deepcopy(metadata)

    def key_for_file(
        self,
        filepath: Path,
        num_frames: int,
        video_loader_backend: str,
        kwargs: dict[str, Any],
    ) -> _VideoDecodeCacheKey:
        stat = filepath.stat()
        kwargs_key = tuple(
            sorted((str(k), self._freeze_value(v)) for k, v in kwargs.items())
        )
        return (
            str(filepath.resolve()),
            stat.st_mtime_ns,
            stat.st_size,
            num_frames,
            video_loader_backend,
            kwargs_key,
        )

    def get_or_load(
        self,
        key: _VideoDecodeCacheKey,
        max_size: int,
        load: Callable[[], _VideoDecodeCacheValue],
    ) -> _VideoDecodeCacheValue:
        owner = False
        with self.lock:
            cached = self.cache.get(key)
            if cached is not None:
                self.cache.move_to_end(key)
                return self._copy_value(cached)

            inflight = self.inflight.get(key)
            if inflight is None:
                inflight = _InflightVideoDecode()
                self.inflight[key] = inflight
                owner = True

        if owner:
            try:
                result = load()
                cached_result = self._copy_value(result)
            except BaseException as exc:
                with self.lock:
                    inflight.error = exc
                    inflight.event.set()
                    self.inflight.pop(key, None)
                raise

            with self.lock:
                self.cache[key] = cached_result
                self.cache.move_to_end(key)
                while len(self.cache) > self.max_size:
                    self.cache.popitem(last=False)
                inflight.result = cached_result
                inflight.event.set()
                self.inflight.pop(key, None)
            return result

        inflight.event.wait()
        if inflight.error is not None:
            raise inflight.error
        assert inflight.result is not None
        return self._copy_value(inflight.result)


_VIDEO_DECODE_CACHE = _VideoDecodeCache()


class VideoMediaIO(MediaIO[tuple[npt.NDArray, dict[str, Any]]]):
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    @classmethod
    def merge_kwargs(
        cls,
        default_kwargs: dict[str, Any] | None,
        runtime_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged = super().merge_kwargs(default_kwargs, runtime_kwargs)
        # fps and num_frames interact with each other, so if either is
        # overridden at request time, wipe the other from defaults to
        # avoid unintuitive cross-field interactions.
        if runtime_kwargs:
            if "num_frames" in runtime_kwargs and "fps" not in runtime_kwargs:
                merged.pop("fps", None)
            elif "fps" in runtime_kwargs and "num_frames" not in runtime_kwargs:
                merged.pop("num_frames", None)
        return merged

    def __init__(
        self,
        image_io: ImageMediaIO,
        num_frames: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()

        self.image_io = image_io
        self.num_frames = num_frames
        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality, merged with
        # per-request runtime media_io_kwargs via merge_kwargs().
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.

        # Allow per-request override of video backend via kwargs.
        # This enables users to specify a different backend than the
        # global VLLM_VIDEO_LOADER_BACKEND env var, e.g.:
        #   --media-io-kwargs '{"video": {"video_backend": "torchcodec"}}'
        video_loader_backend = (
            kwargs.pop("video_backend", None) or envs.VLLM_VIDEO_LOADER_BACKEND
        )
        self.kwargs = kwargs
        self.video_loader = VIDEO_LOADER_REGISTRY.load(video_loader_backend)
        self.video_loader_backend = video_loader_backend

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, dict[str, Any]]:
        return self.video_loader.load_bytes(
            data, num_frames=self.num_frames, **self.kwargs
        )

    def load_base64(
        self, media_type: str, data: str
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        if media_type.lower() == "video/jpeg":
            load_frame = partial(
                self.image_io.load_base64,
                "image/jpeg",
            )
            frames = np.stack(
                [np.asarray(load_frame(frame_data)) for frame_data in data.split(",")]
            )
            num_frames = int(frames.shape[0])
            metadata = {
                "fps": 2.0,
                "duration": num_frames / 2.0,
                "total_num_frames": num_frames,
                "frames_indices": list(range(num_frames)),
                "video_backend": "video/jpeg",
                "do_sample_frames": False,
            }
            return frames, metadata

        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, dict[str, Any]]:
        cache_size = envs.VLLM_VIDEO_DECODE_CACHE_SIZE
        if cache_size <= 0:
            return self._load_file_uncached(filepath)

        return _VIDEO_DECODE_CACHE.get_or_load(
            _VIDEO_DECODE_CACHE.key_for_file(
                filepath,
                self.num_frames,
                self.video_loader_backend,
                self.kwargs,
            ),
            cache_size,
            lambda: self._load_file_uncached(filepath),
        )

    def _load_file_uncached(self, filepath: Path) -> tuple[npt.NDArray, dict[str, Any]]:
        if filepath.suffix == ".lmmsvjpg":
            return self._load_lmms_video_jpeg_file(filepath)
        with filepath.open("rb") as f:
            return self.load_bytes(f.read())
    def _load_lmms_video_jpeg_file(
        self,
        filepath: Path,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        with filepath.open("rb") as f:
            magic = f.read(len(_LMMS_VIDEO_JPEG_MAGIC))
            if magic != _LMMS_VIDEO_JPEG_MAGIC:
                raise ValueError(f"Invalid lmms video jpeg file magic: {filepath}")
            header_len = struct.unpack(">I", f.read(4))[0]
            metadata = json.loads(f.read(header_len).decode("utf-8"))
            frame_count = struct.unpack(">I", f.read(4))[0]
            frames = []
            for _ in range(frame_count):
                payload_len = struct.unpack(">I", f.read(4))[0]
                frame = self.image_io.load_bytes(f.read(payload_len))
                frames.append(np.asarray(frame))
        if not frames:
            raise ValueError(f"No frames found in lmms video jpeg file: {filepath}")
        return np.stack(frames), metadata

    def encode_base64(
        self,
        media: npt.NDArray,
        *,
        video_format: str = "JPEG",
    ) -> str:
        video = media

        if video_format == "JPEG":
            encode_frame = partial(
                self.image_io.encode_base64,
                image_format=video_format,
            )

            return ",".join(encode_frame(Image.fromarray(frame)) for frame in video)

        msg = "Only JPEG format is supported for now."
        raise NotImplementedError(msg)
