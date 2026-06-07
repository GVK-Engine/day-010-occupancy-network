# voxel_grid.py
# GPU-accelerated 3D voxel grid for occupancy estimation
#
# converts raw LiDAR point clouds into a 3D grid
# where each cell (voxel) tracks occupancy state
# and confidence score
#
# voxel states:
#   0 = UNKNOWN          no information yet
#   1 = FREE_CONFIRMED   laser ray passed through
#   2 = FREE_ASSUMED     inferred free from neighbors
#   3 = OCCUPIED         laser ray hit this voxel
#   4 = UNKNOWN_DANGER   near occupied, no ray data
#
# all heavy operations run on GPU via PyTorch CUDA
# 121,000 points processed in < 5ms on RTX 4050
#
# Nani — MS Robotics ASU

import numpy as np
import torch
import os

# voxel grid parameters
# these define the 3D space around the sensor
VOXEL_SIZE   = 0.2      # meters per voxel edge
X_MIN, X_MAX = -50.0, 50.0   # forward/backward range
Y_MIN, Y_MAX = -50.0, 50.0   # left/right range
Z_MIN, Z_MAX =  -2.0,  4.0   # height range

# derived grid dimensions
NX = int((X_MAX - X_MIN) / VOXEL_SIZE)  # 500
NY = int((Y_MAX - Y_MIN) / VOXEL_SIZE)  # 500
NZ = int((Z_MAX - Z_MIN) / VOXEL_SIZE)  # 30

# voxel state codes
UNKNOWN          = 0
FREE_CONFIRMED   = 1
FREE_ASSUMED     = 2
OCCUPIED         = 3
UNKNOWN_DANGER   = 4

# RGB colors per state for visualization
STATE_COLORS = {
    UNKNOWN:        (0.15, 0.15, 0.15),   # dark grey
    FREE_CONFIRMED: (0.0,  0.8,  0.2),    # green
    FREE_ASSUMED:   (0.9,  0.8,  0.0),    # yellow
    OCCUPIED:       (0.9,  0.1,  0.1),    # red
    UNKNOWN_DANGER: (0.5,  0.0,  0.5),    # purple
}


class VoxelGrid:
    """
    3D voxel grid living on GPU.

    The grid is a tensor of shape (NX, NY, NZ)
    storing integer state codes.

    A second tensor stores float confidence scores
    in the same shape.

    The sensor origin is always at the center of
    the X-Y plane and at Z_MIN in height.
    """

    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() \
                     else 'cpu'
        self.device = device
        self.voxel_size = VOXEL_SIZE
        self.nx, self.ny, self.nz = NX, NY, NZ

        # state grid: integer tensor on GPU
        self.states = torch.zeros(
            (NX, NY, NZ),
            dtype=torch.uint8,
            device=device
        )

        # confidence grid: float tensor on GPU
        self.confidence = torch.zeros(
            (NX, NY, NZ),
            dtype=torch.float32,
            device=device
        )

        # hit count: how many times each voxel was hit
        self.hit_count = torch.zeros(
            (NX, NY, NZ),
            dtype=torch.int16,
            device=device
        )

    def reset(self):
        # clear all tensors for new frame
        self.states.zero_()
        self.confidence.zero_()
        self.hit_count.zero_()

    def points_to_voxel_indices(self, points):
        """
        Convert 3D point coordinates to voxel grid indices.

        points: torch tensor (N, 3) — x, y, z in meters
        returns: tuple of three tensors (ix, iy, iz)
                 each shape (N,) with integer indices
                 points outside grid are masked out

        The math:
          voxel index = floor((coordinate - min) / voxel_size)
          e.g. x=5.3m → ix = floor((5.3 - (-50)) / 0.2)
                       → ix = floor(277.5) = 277
        """
        ix = ((points[:, 0] - X_MIN) / VOXEL_SIZE).long()
        iy = ((points[:, 1] - Y_MIN) / VOXEL_SIZE).long()
        iz = ((points[:, 2] - Z_MIN) / VOXEL_SIZE).long()

        # mask: keep only points inside grid bounds
        valid = (
            (ix >= 0) & (ix < NX) &
            (iy >= 0) & (iy < NY) &
            (iz >= 0) & (iz < NZ)
        )

        return ix[valid], iy[valid], iz[valid], valid

    def mark_occupied(self, points_gpu):
        """
        Mark voxels hit by LiDAR returns as OCCUPIED.

        points_gpu: torch tensor (N, 3) on GPU
        Each point is a confirmed LiDAR hit.
        The voxel containing it is OCCUPIED.
        Confidence set to 0.95 (high — direct measurement).
        """
        ix, iy, iz, _ = self.points_to_voxel_indices(
            points_gpu
        )

        if len(ix) == 0:
            return 0

        # mark state and confidence
        self.states[ix, iy, iz]     = OCCUPIED
        self.confidence[ix, iy, iz] = 0.95
        self.hit_count[ix, iy, iz] += 1

        return len(ix)

    def get_occupied_mask(self):
        # returns boolean tensor of occupied voxels
        return self.states == OCCUPIED

    def get_free_confirmed_mask(self):
        return self.states == FREE_CONFIRMED

    def get_unknown_danger_mask(self):
        return self.states == UNKNOWN_DANGER

    def voxel_center(self, ix, iy, iz):
        """
        Convert voxel indices back to 3D coordinates.
        Returns the center point of the voxel.
        """
        x = X_MIN + (ix + 0.5) * VOXEL_SIZE
        y = Y_MIN + (iy + 0.5) * VOXEL_SIZE
        z = Z_MIN + (iz + 0.5) * VOXEL_SIZE
        return x, y, z

    def distance_to_voxel(self, ix, iy, iz):
        """
        Distance from sensor origin (0,0,0) to voxel center.
        Used to compute per-distance-band statistics.
        """
        x, y, z = self.voxel_center(ix, iy, iz)
        return np.sqrt(x**2 + y**2 + z**2)

    def state_statistics(self):
        """
        Count voxels per state.
        Returns dict with counts and percentages.
        """
        total   = self.nx * self.ny * self.nz
        counts  = {}
        names   = {
            UNKNOWN:        'unknown',
            FREE_CONFIRMED: 'free_confirmed',
            FREE_ASSUMED:   'free_assumed',
            OCCUPIED:       'occupied',
            UNKNOWN_DANGER: 'unknown_danger',
        }
        for code, name in names.items():
            n = (self.states == code).sum().item()
            counts[name] = {
                'count': n,
                'pct':   n / total * 100
            }
        return counts

    def to_numpy(self):
        # move grid to CPU for visualization
        return {
            'states':     self.states.cpu().numpy(),
            'confidence': self.confidence.cpu().numpy(),
        }

    def memory_usage_mb(self):
        # total GPU memory used by this grid
        s = self.states.element_size() * self.states.nelement()
        c = self.confidence.element_size() * \
            self.confidence.nelement()
        h = self.hit_count.element_size() * \
            self.hit_count.nelement()
        return (s + c + h) / 1024 / 1024


def load_lidar(filepath, device='cuda'):
    """
    Load KITTI binary LiDAR file and return
    as GPU tensor (N, 4): x, y, z, intensity.

    KITTI format: float32, 4 values per point.
    """
    pts = np.fromfile(filepath, dtype=np.float32)
    pts = pts.reshape(-1, 4)

    # forward-facing points only (x > 0)
    pts = pts[pts[:, 0] > 0]

    return torch.tensor(
        pts, dtype=torch.float32, device=device
    )


if __name__ == "__main__":
    print("voxel grid test")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    # build grid
    grid = VoxelGrid(device=device)
    print(f"grid shape  : {grid.nx} x {grid.ny} x {grid.nz}")
    print(f"voxel size  : {grid.voxel_size}m")
    print(f"memory usage: {grid.memory_usage_mb():.1f} MB")

    # test with one KITTI frame
    lidar_path = (
        r"C:\Users\vamsh\Downloads\kitti"
        r"\2011_09_26_drive_0001_sync"
        r"\2011_09_26"
        r"\2011_09_26_drive_0001_sync"
        r"\velodyne_points\data"
        r"\0000000000.bin"
    )

    if os.path.exists(lidar_path):
        pts = load_lidar(lidar_path, device=device)
        print(f"points loaded: {len(pts)}")

        import time
        t0 = time.time()
        n  = grid.mark_occupied(pts[:, :3])
        ms = (time.time() - t0) * 1000
        print(f"voxelization: {ms:.2f}ms")
        print(f"occupied voxels: {n}")

        stats = grid.state_statistics()
        occ = stats['occupied']
        print(f"occupied: {occ['count']} "
              f"({occ['pct']:.3f}% of grid)")
    else:
        print("lidar file not found, path check needed")

NEIGHBORHOOD = 5