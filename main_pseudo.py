"""On-demand physics residual labels, optionally cached within a disk budget."""
import os
import tempfile
import time
import warnings
from zipfile import BadZipFile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

PSEUDO_VERSION = "edt_residual_float32_v1"


def pseudo_label(measured, known, depth, occupied, sigma=1.0, truncate=2.0, spacing=(1, 1, 1)):
    if not known.any():
        raise ValueError("Pseudo-label generation needs at least one measurement")
    # Do not compute or retain the unused float64 distance volume.
    indices = distance_transform_edt(~known, sampling=spacing, return_distances=False, return_indices=True)
    residual = np.zeros(depth.shape, dtype=np.float32)
    residual[known] = measured[known] - depth[known]
    filled = residual[tuple(indices)]
    del indices, residual
    if sigma > 0:
        gaussian_filter(filled, sigma=sigma, mode="reflect", truncate=truncate, output=filled)
    filled += depth
    np.clip(filled, 0, 1, out=filled)
    filled[known] = measured[known]
    filled[occupied] = 0
    return filled


@contextmanager
def _cache_lock(directory):
    lock = directory / ".write_lock"
    deadline = time.monotonic() + 30
    fd = None
    while fd is None:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() > deadline:
                # Cache is optional; stale locks must not prevent training.
                yield False
                return
            time.sleep(0.05)
    try:
        yield True
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)


class BoundedPseudoCache:
    """Stores float16 labels (not quantized uint8); no full dataset in RAM."""
    def __init__(self, directory, max_gib):
        self.directory = Path(directory)
        self.max_bytes = int(max_gib * 1024**3)

    def get(self, key, generate):
        if self.max_bytes <= 0:
            return generate()
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self.max_bytes = 0
            warnings.warn(f"Pseudo cache unavailable; computing labels without disk cache: {error}")
            return generate()
        path = self.directory / f"pseudo_{key}.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                label = archive["y2"].astype(np.float32)
            os.utime(path, None)
            return label
        except (FileNotFoundError, OSError, ValueError, KeyError, EOFError, BadZipFile):
            pass
        # Both hit and miss return the same float16-rounded target.
        label = generate().astype(np.float16)
        try:
            fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        except OSError as error:
            self.max_bytes = 0
            warnings.warn(f"Pseudo cache unavailable; computing labels without disk cache: {error}")
            return label.astype(np.float32)
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez_compressed(f, y2=label)
            size = os.path.getsize(tmp)
            with _cache_lock(self.directory) as locked:
                if locked and size <= self.max_bytes:
                    entries = []
                    for p in self.directory.glob("pseudo_*.npz"):
                        try:
                            st = p.stat()
                            entries.append((st.st_mtime, p, st.st_size))
                        except FileNotFoundError:
                            continue
                    total = sum(e[2] for e in entries)
                    old_size = path.stat().st_size if path.exists() else 0
                    for _, candidate, candidate_size in sorted(entries):
                        if total - old_size + size <= self.max_bytes:
                            break
                        if candidate != path:
                            candidate.unlink(missing_ok=True)
                            total -= candidate_size
                    os.replace(tmp, path)
        except OSError as error:
            self.max_bytes = 0
            warnings.warn(f"Pseudo cache write failed; subsequent labels will be computed on demand: {error}")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return label.astype(np.float32)
