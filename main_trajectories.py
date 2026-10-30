# PIRM3D: physics-inspired 3D radio map construction.
"""Obstacle-aware, ordered 6-connected flight paths; no mask dilation or point dropping.

Each route is a segment of a seeded depth-first coverage walk. New voxel budgets
are equal across routes; backtracking and shared endpoints remain in the route.
Consequently per-route coverage can exceed the nominal budget; exact statistics
are saved rather than claiming the overlapping routes have additive coverage.
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from numba import njit

from main_protocol import atomic_json, digest

ALGORITHM = "connected_dfs_coverage_v1"


@njit(cache=True)
def _coverage_walk(free, count, quota, seed):
    np.random.seed(seed)
    nz, ny, nx = free.shape
    total = nz * ny * nx
    allowed = free.ravel()
    seen = np.zeros(total, dtype=np.uint8)
    stack = np.empty(total, dtype=np.int32)
    # DFS traverses each tree edge at most twice, plus duplicated route endpoints.
    trace = np.empty((2 * total + count, 3), dtype=np.uint16)
    offsets = np.zeros(count + 1, dtype=np.int64)
    start = -1
    # Start in the highest flight layer (all voxels need not be free).
    start_col = np.random.randint(ny * nx)
    for j in range(ny * nx):
        candidate = (nz - 1) * ny * nx + (start_col + j) % (ny * nx)
        if allowed[candidate]:
            start = candidate
            break
    if start < 0:
        raise ValueError("No valid start in highest altitude layer")
    stack[0] = start
    seen[start] = 1
    top, used = 0, 0
    # z,y,x axis-aligned neighbours, in physical free space.
    dz = np.array([0, 0, 0, 0, -1, 1])
    dy = np.array([0, 0, -1, 1, 0, 0])
    dx = np.array([-1, 1, 0, 0, 0, 0])
    last_dir = -1
    for route in range(count):
        offsets[route] = used
        current = stack[top]
        trace[used, 0] = current // (ny * nx)
        trace[used, 1] = (current // nx) % ny
        trace[used, 2] = current % nx
        used += 1
        # First start is part of the global unique measurement budget.
        new_points = 1 if route == 0 else 0
        while new_points < quota:
            current = stack[top]
            z, y, x = current // (ny * nx), (current // nx) % ny, current % nx
            choice, available = -1, 0
            for direction in range(6):
                zz, yy, xx = z + dz[direction], y + dy[direction], x + dx[direction]
                if zz < 0 or zz >= nz or yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                    continue
                index = (zz * ny + yy) * nx + xx
                if allowed[index] and not seen[index]:
                    available += 1
                    if np.random.randint(available) == 0:
                        choice = direction
            # Prefer a straight segment when possible, retaining random turns.
            if last_dir >= 0 and np.random.random() < 0.65:
                zz, yy, xx = z + dz[last_dir], y + dy[last_dir], x + dx[last_dir]
                if 0 <= zz < nz and 0 <= yy < ny and 0 <= xx < nx:
                    index = (zz * ny + yy) * nx + xx
                    if allowed[index] and not seen[index]:
                        choice = last_dir
            if choice >= 0:
                zz, yy, xx = z + dz[choice], y + dy[choice], x + dx[choice]
                current = (zz * ny + yy) * nx + xx
                top += 1
                stack[top] = current
                seen[current] = 1
                new_points += 1
                last_dir = choice
            else:
                if top == 0:
                    raise ValueError("Connected free-space component cannot meet trajectory budget")
                top -= 1
                current = stack[top]
                zz, yy, xx = current // (ny * nx), (current // nx) % ny, current % nx
                last_dir = -1
            trace[used, 0], trace[used, 1], trace[used, 2] = zz, yy, xx
            used += 1
    offsets[count] = used
    return trace[:used].copy(), offsets


def trajectory_spec(cfg, scene_id, free):
    return {"algorithm": ALGORITHM, "scene_id": int(scene_id), "shape": list(free.shape),
            "free_hash": digest(np.packbits(free).tobytes().hex()),
            "seed": cfg["seed"] + int(scene_id), "count": cfg["trajectory_count"],
            "ratio": cfg["trajectory_ratio"], "clearance_m": cfg["trajectory_clearance_m"],
            "spacing_m": cfg["spacing_m"], "speed_mps": cfg["trajectory_speed_mps"]}


def generate_trajectories(cfg, scene_id, free):
    if max(free.shape) >= 65536:
        raise ValueError("Coordinates do not fit uint16")
    quota = int(round(free.size * cfg["trajectory_ratio"]))
    if quota < 2 or quota * cfg["trajectory_count"] > int(free.sum()):
        raise ValueError("Insufficient collision-free voxels for the nominal full-volume budget")
    coords, offsets = _coverage_walk(np.ascontiguousarray(free), cfg["trajectory_count"], quota,
                                    (cfg["seed"] + int(scene_id)) % (2**31))
    stats = []
    union = np.zeros(free.shape, dtype=bool)
    for i in range(cfg["trajectory_count"]):
        points = coords[offsets[i]:offsets[i + 1]]
        unique = np.unique(points, axis=0)
        union[tuple(unique.T)] = True
        delta = np.diff(points.astype(np.int32), axis=0)
        distance = float(np.linalg.norm(delta * np.asarray(cfg["spacing_m"]), axis=1).sum())
        stats.append({"id": i, "ordered_samples": len(points), "unique_voxels": len(unique),
                      "full_volume_ratio": len(unique) / free.size,
                      "length_m": distance, "duration_s": distance / cfg["trajectory_speed_mps"],
                      "altitude_min_m": float((int(points[:, 0].min()) + 1) * cfg["spacing_m"][0]),
                      "altitude_max_m": float((int(points[:, 0].max()) + 1) * cfg["spacing_m"][0])})
    metadata = {"spec": trajectory_spec(cfg, scene_id, free), "routes": stats,
                "union_voxels": int(union.sum()), "union_full_volume_ratio": float(union.mean()),
                "union_valid_air_ratio": float(union.sum() / max(free.sum(), 1)),
                "budget_new_voxels_per_route": quota,
                "motion_assumption": "Axis-aligned waypoint polyline; speed is nominal cruise speed. "
                                     "Turn acceleration, battery limits and takeoff/landing are not modelled.",
                "sample_interval_s_per_axis": [s / cfg["trajectory_speed_mps"] for s in cfg["spacing_m"]]}
    return coords, offsets, metadata


def get_trajectories(cfg, scene_id, free):
    spec = trajectory_spec(cfg, scene_id, free)
    key = digest(spec)
    # Packed ordered coordinates keep route identity via metadata offsets.
    directory = Path(cfg["trajectory_dir"]) / str(scene_id)
    stem = f"{cfg['trajectory_ratio'] * 100:g}".replace(".", "p")
    path = directory / f"{stem}_{key[:16]}.npy"
    meta_path = path.with_suffix(".json")
    if path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["spec"] != spec:
            raise ValueError("Trajectory cache metadata mismatch")
        coords = np.load(path, allow_pickle=False)
        offsets = np.asarray(meta["offsets"], dtype=np.int64)
        return coords, offsets, meta
    coords, offsets, meta = generate_trajectories(cfg, scene_id, free)
    meta["offsets"] = offsets.tolist()
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, coords, allow_pickle=False)
        os.replace(tmp, path)
        atomic_json(meta_path, meta)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return coords, offsets, meta


def route_mask(coords, offsets, index, shape):
    if not 0 <= index < len(offsets) - 1:
        raise IndexError("Trajectory index out of range")
    mask = np.zeros(shape, dtype=bool)
    points = coords[offsets[index]:offsets[index + 1]]
    mask[tuple(points.T)] = True
    return mask
