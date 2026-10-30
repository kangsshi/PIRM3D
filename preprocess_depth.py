"""Generate/reuse raw physical depth for selected samples only, without OpenCV."""
import os
import tempfile

import numpy as np
from numba import njit

from main_protocol import building_path, depth_path, read_gray


@njit(cache=True)
def calculate_radio_depth(building, tx_col, tx_row, tx_z, rx_z, freq_mhz, dy=1.0, dx=1.0):
    ny, nx = building.shape
    result = np.zeros((ny, nx), dtype=np.float32)
    term_freq = 20.0 * np.log10(freq_mhz)
    for r in range(ny):
        for c in range(nx):
            dist_xy = np.sqrt(((r - tx_row) * dy)**2 + ((c - tx_col) * dx)**2)
            distance = np.sqrt(dist_xy**2 + (rx_z - tx_z)**2)
            path_loss = 20.0 * np.log10(distance) if distance >= 1.0 else 0.0
            steps = max(1, int(np.ceil(np.sqrt((r - tx_row)**2 + (c - tx_col)**2))))
            blocked = 0
            for k in range(steps + 1):
                t = k / steps
                rr = int(round(tx_row + (r - tx_row) * t))
                cc = int(round(tx_col + (c - tx_col) * t))
                zz = tx_z + (rx_z - tx_z) * t
                if zz < building[rr, cc] - 0.1:
                    blocked += 1
            transmittance = 1.0 - blocked / (steps + 1)
            result[r, c] = transmittance * max(0.0, 135.0 - term_freq - path_loss)
    return result


def ensure_depth(cfg, row):
    path = depth_path(cfg, row)
    shape = tuple(cfg["shape"])
    if path.is_file():
        existing = np.load(path, mmap_mode="r", allow_pickle=False)
        if existing.shape != shape or not np.issubdtype(existing.dtype, np.floating):
            raise ValueError(f"Invalid existing depth file (not overwritten): {path}")
        if not np.isfinite(existing).all() or (existing < 0).any():
            raise ValueError(f"Invalid existing depth values (not overwritten): {path}")
        return "reused"
    building = read_gray(building_path(cfg["data_root"], row["scene_id"]), shape[1:]).astype(np.float32)
    building *= cfg["max_building_height_m"] / 255.0
    _, sx, sy = row["filename"][:-4].split("_")
    tx_row, tx_col = shape[1] - 1 - int(sx[1:]), int(sy[1:])
    volume = np.empty(shape, dtype=np.float16)
    dz, dy, dx = cfg["spacing_m"]
    for z in range(shape[0]):
        volume[z] = calculate_radio_depth(building, tx_col, tx_row, 1.5, (z + 1) * dz, 5900.0, dy, dx)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, volume, allow_pickle=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return "generated"


if __name__ == "__main__":
    raise SystemExit("Use python prepare_main.py --config main_config.json --depth")
