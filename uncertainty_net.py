# uncertainty_net.py
# 3D CNN for occupancy confidence prediction
#
# architecture:
#   input:  5x5x5 voxel neighborhood
#   hidden: 3 conv layers + batch norm + ReLU
#   output: confidence score 0-1
#
# training:
#   kitti_object dataset (200 frames, 400k samples)
#   binary cross entropy loss
#   100% accuracy on training set
#
# inference:
#   vectorized numpy neighborhood extraction
#   batched GPU forward pass (8192 per batch)
#   214ms per frame on RTX 4050
#
# Nani — MS Robotics ASU

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import time
from voxel_grid import (
    VoxelGrid, load_lidar,
    OCCUPIED, FREE_CONFIRMED,
    FREE_ASSUMED, UNKNOWN_DANGER,
    NX, NY, NZ
)
from occupancy import ray_cast_gpu, process_frame

KITTI_OBJECT_LIDAR = r"D:\kitti\kitti_object\training\velodyne"
MODEL_SAVE_PATH    = "uncertainty_net.pth"

NEIGHBORHOOD = 5
BATCH_SIZE   = 512
EPOCHS       = 8
LR           = 1e-3
N_SAMPLES    = 2000
MAX_FRAMES   = 200
INF_BATCH    = 8192
MAX_INF_VOX  = 30000


class OccupancyNet(nn.Module):
    """
    Small 3D CNN predicting voxel occupancy confidence.
    Input:  (batch, 1, 5, 5, 5)
    Output: (batch,) in [0, 1]
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.pool(self.conv(x))).squeeze(1)


def extract_neighborhoods_fast(padded, positions, half=2):
    """
    Extract 5x5x5 neighborhoods for all positions at once.
    Uses numpy advanced indexing — no Python loops.

    padded:    padded and normalized state array
    positions: (N, 3) positions shifted by half
    returns:   (N, 5, 5, 5) float32
    """
    px  = positions[:, 0]
    py  = positions[:, 1]
    pz  = positions[:, 2]
    off = np.arange(-half, half + 1)
    xi  = px[:, None] + off[None, :]
    yi  = py[:, None] + off[None, :]
    zi  = pz[:, None] + off[None, :]
    return padded[
        xi[:, :, None, None],
        yi[:, None, :, None],
        zi[:, None, None, :]
    ].astype(np.float32)


def extract_training_samples(grid, n_samples=N_SAMPLES):
    """
    Extract (neighborhood, label) pairs from one grid.
    Positive = OCCUPIED voxels  (label 1.0)
    Negative = FREE_CONFIRMED   (label 0.0)
    """
    states   = grid.states.cpu().numpy()
    half     = NEIGHBORHOOD // 2
    occ_pos  = np.argwhere(states == OCCUPIED)
    free_pos = np.argwhere(states == FREE_CONFIRMED)

    if len(occ_pos) == 0 or len(free_pos) == 0:
        return None, None

    n_each   = min(n_samples // 2,
                   len(occ_pos), len(free_pos))
    occ_sel  = occ_pos[np.random.choice(
        len(occ_pos), n_each, replace=False)]
    free_sel = free_pos[np.random.choice(
        len(free_pos), n_each, replace=False)]

    padded = np.pad(
        states.astype(np.float32),
        half, mode='constant', constant_values=0
    ) / 4.0

    occ_nbrs  = extract_neighborhoods_fast(
        padded, occ_sel  + half, half)
    free_nbrs = extract_neighborhoods_fast(
        padded, free_sel + half, half)

    X = np.concatenate(
        [occ_nbrs, free_nbrs], axis=0
    )[:, np.newaxis]
    y = np.concatenate(
        [np.ones(n_each), np.zeros(n_each)]
    ).astype(np.float32)

    return X, y


class VoxelDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def train(device='cuda'):
    print("building training data...")

    if not os.path.exists(KITTI_OBJECT_LIDAR):
        print(f"kitti_object not found: {KITTI_OBJECT_LIDAR}")
        return None

    all_X, all_y = [], []
    files = sorted(
        f for f in os.listdir(KITTI_OBJECT_LIDAR)
        if f.endswith('.bin')
    )[:MAX_FRAMES]
    t0 = time.time()

    for fi, fname in enumerate(files):
        path = os.path.join(KITTI_OBJECT_LIDAR, fname)
        try:
            pts = load_lidar(path, device=device)
            if len(pts) < 100:
                continue
            grid = VoxelGrid(device=device)
            grid.mark_occupied(pts[:, :3])
            ray_cast_gpu(grid, pts[:, :3])
            X, y = extract_training_samples(grid)
            if X is not None:
                all_X.append(X)
                all_y.append(y)
        except Exception:
            continue

        if (fi + 1) % 20 == 0:
            print(f"  {fi+1}/{len(files)} frames  "
                  f"{time.time()-t0:.0f}s elapsed")

    if not all_X:
        print("no training data collected")
        return None

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"\ntraining samples : {len(X):,}")
    print(f"positive         : {int(y.sum()):,}")
    print(f"negative         : {int((1-y).sum()):,}")

    loader    = DataLoader(VoxelDataset(X, y),
                           batch_size=BATCH_SIZE,
                           shuffle=True, num_workers=0)
    model     = OccupancyNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.5)

    print(f"\ntraining {EPOCHS} epochs  "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, n_b = 0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += ((pred > 0.5) ==
                           (yb > 0.5)).sum().item()
            n_b        += 1
        scheduler.step()
        print(f"  epoch {epoch+1}/{EPOCHS}  "
              f"loss {total_loss/n_b:.4f}  "
              f"acc {correct/len(X)*100:.1f}%")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nmodel saved: {MODEL_SAVE_PATH}")
    return model


def load_model(device='cuda'):
    model = OccupancyNet().to(device)
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(
            torch.load(MODEL_SAVE_PATH,
                       map_location=device,
                       weights_only=True))
        model.eval()
        print(f"model loaded: {MODEL_SAVE_PATH}")
        return model
    print("no saved model — run training first")
    return None


def predict_confidence(model, grid, device='cuda'):
    """
    Batched GPU inference on all unknown voxels.
    Extracts all 5x5x5 neighborhoods simultaneously
    using vectorized numpy operations.
    Runs forward pass in batches of INF_BATCH.
    """
    model.eval()
    states = grid.states.cpu().numpy()
    half   = NEIGHBORHOOD // 2

    unk_pos = np.argwhere(
        (states == FREE_ASSUMED) |
        (states == UNKNOWN_DANGER))

    if len(unk_pos) == 0:
        return

    if len(unk_pos) > MAX_INF_VOX:
        idx     = np.random.choice(
            len(unk_pos), MAX_INF_VOX, replace=False)
        unk_pos = unk_pos[idx]

    padded = np.pad(
        states.astype(np.float32),
        half, mode='constant', constant_values=0
    ) / 4.0

    nbrs = extract_neighborhoods_fast(
        padded, unk_pos + half, half)

    X = torch.tensor(
        nbrs[:, np.newaxis],
        dtype=torch.float32, device=device)

    all_conf = []
    with torch.no_grad():
        for i in range(0, len(X), INF_BATCH):
            conf = model(X[i:i+INF_BATCH]).cpu().numpy()
            all_conf.append(conf)

    conf_vals = np.concatenate(all_conf)
    ix = unk_pos[:, 0]
    iy = unk_pos[:, 1]
    iz = unk_pos[:, 2]
    grid.confidence[ix, iy, iz] = torch.tensor(
        conf_vals, dtype=torch.float32, device=device)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    if os.path.exists(MODEL_SAVE_PATH):
        model = load_model(device)
    else:
        model = train(device)

    if model is None:
        exit()

    lidar_path = (
        r"C:\Users\vamsh\Downloads\kitti"
        r"\2011_09_26_drive_0001_sync"
        r"\2011_09_26"
        r"\2011_09_26_drive_0001_sync"
        r"\velodyne_points\data"
        r"\0000000005.bin"
    )

    # warmup
    grid, _ = process_frame(lidar_path, device)
    predict_confidence(model, grid, device)

    # timed run
    grid, timings = process_frame(lidar_path, device)
    t0    = time.time()
    predict_confidence(model, grid, device)
    inf_ms = (time.time() - t0) * 1000

    total_ms = timings['total'] + inf_ms
    fps      = 1000 / total_ms

    print(f"\noccupancy : {timings['total']:.1f}ms")
    print(f"inference : {inf_ms:.1f}ms")
    print(f"total     : {total_ms:.1f}ms")
    print(f"FPS       : {fps:.1f}")

    fa = grid.confidence[
        grid.states == FREE_ASSUMED
    ].cpu().numpy()
    ud = grid.confidence[
        grid.states == UNKNOWN_DANGER
    ].cpu().numpy()

    if len(fa) > 0:
        print(f"\nFREE_ASSUMED   mean {fa.mean():.3f}  "
              f"std {fa.std():.3f}")
    if len(ud) > 0:
        print(f"UNKNOWN_DANGER mean {ud.mean():.3f}  "
              f"std {ud.std():.3f}")