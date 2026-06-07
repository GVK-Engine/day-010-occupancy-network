# occupancy.py
# ray casting and 4-state occupancy classification
#
# takes a VoxelGrid with OCCUPIED voxels already marked
# fills FREE_CONFIRMED via GPU ray casting
# classifies FREE_ASSUMED and UNKNOWN_DANGER
# measures uncertainty per distance band
#
# optimizations vs v1:
#   classify uses sparse indexing not dense convolution
#   ray casting uses fewer samples with larger step
#   distance map precomputed once not per frame
#
# Nani — MS Robotics ASU

import numpy as np
import torch
import os
import time
from voxel_grid import (
    VoxelGrid, load_lidar,
    FREE_CONFIRMED, FREE_ASSUMED,
    OCCUPIED, UNKNOWN, UNKNOWN_DANGER,
    VOXEL_SIZE, X_MIN, Y_MIN, Z_MIN,
    NX, NY, NZ
)

RAY_STEP = VOXEL_SIZE * 0.8   # slightly larger step = faster

BANDS = [
    (0,  10,  "0-10m"),
    (10, 20,  "10-20m"),
    (20, 30,  "20-30m"),
    (30, 40,  "30-40m"),
    (40, 50,  "40-50m"),
]

# precompute distance map once at module load
# shape (NX, NY) — horizontal distance per voxel column
_ix = np.arange(NX)
_iy = np.arange(NY)
_cx = X_MIN + (_ix + 0.5) * VOXEL_SIZE
_cy = Y_MIN + (_iy + 0.5) * VOXEL_SIZE
_cx_g, _cy_g = np.meshgrid(_cx, _cy, indexing='ij')
DIST_MAP_2D = np.sqrt(_cx_g**2 + _cy_g**2).astype(np.float32)


def ray_cast_gpu(grid, points_gpu, max_range=50.0):
    """
    Mark FREE_CONFIRMED voxels along each LiDAR ray.

    samples positions along every ray simultaneously
    uses vectorized GPU operations throughout
    skips voxels already marked OCCUPIED
    """
    device = grid.device
    if len(points_gpu) == 0:
        return

    # filter to max range
    dists   = torch.norm(points_gpu, dim=1)
    mask    = dists < max_range
    pts     = points_gpu[mask]
    dst     = dists[mask].unsqueeze(1)

    if len(pts) == 0:
        return

    dirs = pts / dst    # unit vectors (M, 3)
    M    = len(pts)

    # step values along each ray
    t_vals  = torch.arange(
        RAY_STEP, max_range, RAY_STEP,
        device=device, dtype=torch.float32
    )
    n_steps = len(t_vals)

    # sample all points: (M, n_steps, 3)
    samples = dirs.unsqueeze(1) * \
              t_vals.view(1, n_steps, 1)

    # mask steps beyond actual hit point
    t_rep  = t_vals.view(1, n_steps).expand(M, -1)
    d_rep  = dst.expand(-1, n_steps)
    before = (t_rep < d_rep - RAY_STEP).reshape(-1)

    samples = samples.reshape(-1, 3)[before]

    if len(samples) == 0:
        return

    ix = ((samples[:, 0] - X_MIN) / VOXEL_SIZE).long()
    iy = ((samples[:, 1] - Y_MIN) / VOXEL_SIZE).long()
    iz = ((samples[:, 2] - Z_MIN) / VOXEL_SIZE).long()

    valid = (
        (ix >= 0) & (ix < NX) &
        (iy >= 0) & (iy < NY) &
        (iz >= 0) & (iz < NZ)
    )
    ix, iy, iz = ix[valid], iy[valid], iz[valid]

    not_occ = grid.states[ix, iy, iz] != OCCUPIED
    ix, iy, iz = ix[not_occ], iy[not_occ], iz[not_occ]

    if len(ix) == 0:
        return

    grid.states[ix, iy, iz]     = FREE_CONFIRMED
    grid.confidence[ix, iy, iz] = 0.90


def classify_unknown_voxels(grid):
    """
    Classify remaining UNKNOWN voxels using
    sparse neighborhood lookup.

    Faster than dense convolution:
      only processes UNKNOWN voxels
      not all 7.5 million voxels
    """
    device = grid.device

    # find unknown voxel indices
    unk_idx = (grid.states == UNKNOWN).nonzero(
        as_tuple=False
    )   # (K, 3)

    if len(unk_idx) == 0:
        return

    ix = unk_idx[:, 0]
    iy = unk_idx[:, 1]
    iz = unk_idx[:, 2]

    # check 6-face neighbors (not full 3x3x3)
    # faster and sufficient for classification
    offsets = torch.tensor([
        [ 1, 0, 0], [-1, 0, 0],
        [ 0, 1, 0], [ 0,-1, 0],
        [ 0, 0, 1], [ 0, 0,-1],
    ], device=device)   # (6, 3)

    # neighbor indices: (K, 6, 3)
    nix = (ix.unsqueeze(1) +
           offsets[:, 0].unsqueeze(0)).clamp(0, NX-1)
    niy = (iy.unsqueeze(1) +
           offsets[:, 1].unsqueeze(0)).clamp(0, NY-1)
    niz = (iz.unsqueeze(1) +
           offsets[:, 2].unsqueeze(0)).clamp(0, NZ-1)

    # get neighbor states: (K, 6)
    nbr_states = grid.states[nix, niy, niz]

    # any neighbor occupied?
    has_occ  = (nbr_states == OCCUPIED).any(dim=1)
    # any neighbor free confirmed?
    has_free = (nbr_states == FREE_CONFIRMED).any(dim=1)

    # UNKNOWN_DANGER: near an occupied voxel
    danger_mask = has_occ
    grid.states[ix[danger_mask],
                iy[danger_mask],
                iz[danger_mask]] = UNKNOWN_DANGER
    grid.confidence[ix[danger_mask],
                    iy[danger_mask],
                    iz[danger_mask]] = 0.15

    # FREE_ASSUMED: not near occupied but near free
    assumed_mask = (~has_occ) & has_free
    grid.states[ix[assumed_mask],
                iy[assumed_mask],
                iz[assumed_mask]] = FREE_ASSUMED
    grid.confidence[ix[assumed_mask],
                    iy[assumed_mask],
                    iz[assumed_mask]] = 0.55


def process_frame(lidar_path, device='cuda'):
    """
    Full occupancy pipeline for one LiDAR frame.
    Returns grid and timing dict.
    """
    timings = {}

    t0  = time.time()
    pts = load_lidar(lidar_path, device=device)
    timings['load'] = (time.time() - t0) * 1000

    grid = VoxelGrid(device=device)

    t0 = time.time()
    grid.mark_occupied(pts[:, :3])
    timings['voxelize'] = (time.time() - t0) * 1000

    t0 = time.time()
    ray_cast_gpu(grid, pts[:, :3])
    timings['raycast'] = (time.time() - t0) * 1000

    t0 = time.time()
    classify_unknown_voxels(grid)
    timings['classify'] = (time.time() - t0) * 1000

    timings['total'] = sum(timings.values())
    return grid, timings


def measure_uncertainty_by_distance(grid):
    """
    The novel finding:
    per-distance-band state distribution.
    Shows where FREE_ASSUMED catches FREE_CONFIRMED.
    """
    states_np = grid.states.cpu().numpy()

    # expand 2D distance map to 3D
    dist_3d = np.repeat(
        DIST_MAP_2D[:, :, np.newaxis], NZ, axis=2
    )

    results = {}
    for d_min, d_max, label in BANDS:
        band    = (dist_3d >= d_min) & (dist_3d < d_max)
        s       = states_np[band]
        total   = len(s)

        if total == 0:
            results[label] = None
            continue

        results[label] = {
            'total':          total,
            'occupied_pct':   (s == OCCUPIED).sum()        / total * 100,
            'free_conf_pct':  (s == FREE_CONFIRMED).sum()  / total * 100,
            'free_assum_pct': (s == FREE_ASSUMED).sum()    / total * 100,
            'danger_pct':     (s == UNKNOWN_DANGER).sum()  / total * 100,
            'unknown_pct':    (s == UNKNOWN).sum()         / total * 100,
        }

    return results


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    lidar_path = (
        r"C:\Users\vamsh\Downloads\kitti"
        r"\2011_09_26_drive_0001_sync"
        r"\2011_09_26"
        r"\2011_09_26_drive_0001_sync"
        r"\velodyne_points\data"
        r"\0000000005.bin"
    )

    # warmup run
    process_frame(lidar_path, device)

    # timed run
    print("processing frame...")
    grid, timings = process_frame(lidar_path, device)

    print(f"\ntimings:")
    for k, v in timings.items():
        print(f"  {k:<12}: {v:.1f}ms")

    print(f"\nvoxel states:")
    stats = grid.state_statistics()
    for name, info in stats.items():
        if info['pct'] > 0.001:
            print(f"  {name:<18}: {info['count']:>8,} "
                  f"({info['pct']:.3f}%)")

    print(f"\nuncertainty by distance:")
    unc = measure_uncertainty_by_distance(grid)
    print(f"  {'band':<10} {'free_conf%':>10} "
          f"{'free_assum%':>12} {'ratio':>7}")
    print(f"  {'-'*43}")
    for label, r in unc.items():
        if r is None:
            continue
        fc  = r['free_conf_pct']
        fa  = r['free_assum_pct']
        rat = fc / fa if fa > 0 else 999
        print(f"  {label:<10} {fc:>10.1f} "
              f"{fa:>12.1f} {rat:>7.1f}x")