# visualize.py
# all visual outputs for Day 10 occupancy network
#
# coordinate system:
#   KITTI: X=forward, Y=left, Z=up
#   BEV display: ahead=top, right=right (matches camera)
#   bev_orient() applies flipud + fliplr everywhere
#
# Nani — MS Robotics ASU

import numpy as np
import torch
import os
import cv2
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio

from voxel_grid import (
    VoxelGrid, load_lidar,
    FREE_CONFIRMED, FREE_ASSUMED,
    OCCUPIED, UNKNOWN, UNKNOWN_DANGER,
    VOXEL_SIZE, X_MIN, Y_MIN, Z_MIN,
    NX, NY, NZ, NEIGHBORHOOD
)
from occupancy import (
    process_frame, measure_uncertainty_by_distance,
    BANDS
)
from uncertainty_net import (
    OccupancyNet, load_model, predict_confidence
)

KITTI_DIR = (
    r"C:\Users\vamsh\Downloads\kitti"
    r"\2011_09_26_drive_0001_sync"
    r"\2011_09_26"
    r"\2011_09_26_drive_0001_sync"
    r"\velodyne_points\data"
)
KITTI_IMG_DIR = (
    r"C:\Users\vamsh\Downloads\kitti"
    r"\2011_09_26_drive_0001_sync"
    r"\2011_09_26"
    r"\2011_09_26_drive_0001_sync"
    r"\image_02\data"
)
RESULTS_DIR = "results"
GIF_3D_SIZE = (800, 600)

BEV_X0, BEV_X1 = 250, 490
BEV_Y0, BEV_Y1 = 130, 370

STATE_RGB = {
    UNKNOWN:        (20,  20,  20),
    FREE_CONFIRMED: (0,  210,  70),
    FREE_ASSUMED:   (230, 200,  0),
    OCCUPIED:       (220,  30,  30),
    UNKNOWN_DANGER: (160,   0, 200),
}

STATE_NAMES = {
    UNKNOWN:        "Unknown",
    FREE_CONFIRMED: "Free Confirmed",
    FREE_ASSUMED:   "Free Assumed",
    OCCUPIED:       "Occupied",
    UNKNOWN_DANGER: "Unknown Danger",
}


def bev_orient(arr):
    # flipud: ahead at top, fliplr: match camera left/right
    return np.fliplr(np.flipud(arr))


def make_bev_frame(grid):
    states_np = grid.states.cpu().numpy()
    pmap      = {UNKNOWN: 0, FREE_CONFIRMED: 1,
                 FREE_ASSUMED: 2, UNKNOWN_DANGER: 3,
                 OCCUPIED: 4}
    priority  = np.zeros_like(
        states_np[:, :, 0], dtype=np.uint8)
    for z in range(NZ):
        layer = states_np[:, :, z]
        p     = np.vectorize(pmap.get)(layer)
        old_p = np.vectorize(pmap.get)(priority)
        priority[p > old_p] = layer[p > old_p]
    img = np.zeros((NX, NY, 3), dtype=np.uint8)
    for state, rgb in STATE_RGB.items():
        img[priority == state] = rgb
    return bev_orient(img[BEV_X0:BEV_X1,
                           BEV_Y0:BEV_Y1])


def make_conf_heatmap(grid, W, H):
    conf_np   = grid.confidence.cpu().numpy()
    states_np = grid.states.cpu().numpy()
    conf_2d   = conf_np.max(axis=2)
    crop      = bev_orient(conf_2d[BEV_X0:BEV_X1,
                                    BEV_Y0:BEV_Y1])
    norm_inv  = 255 - (np.clip(crop, 0, 1) * 255
                       ).astype(np.uint8)
    colored   = cv2.applyColorMap(norm_inv,
                                   cv2.COLORMAP_SUMMER)
    known_2d  = (states_np != UNKNOWN).any(axis=2)
    known_c   = bev_orient(known_2d[BEV_X0:BEV_X1,
                                     BEV_Y0:BEV_Y1])
    colored[~known_c] = 20
    return cv2.resize(colored, (W, H))


def plot_3d_occupancy_frame(grid, ax, angle,
                             max_pts=4000):
    states_np = grid.states.cpu().numpy()
    ax.cla()
    ax.set_facecolor('#0a0a0a')
    for state, rgb in STATE_RGB.items():
        if state == UNKNOWN:
            continue
        pos = np.argwhere(states_np == state)
        if len(pos) == 0:
            continue
        if len(pos) > max_pts:
            pos = pos[np.random.choice(
                len(pos), max_pts, replace=False)]
        x = X_MIN + (pos[:, 0] + 0.5) * VOXEL_SIZE
        y = Y_MIN + (pos[:, 1] + 0.5) * VOXEL_SIZE
        z = Z_MIN + (pos[:, 2] + 0.5) * VOXEL_SIZE
        c = tuple(v/255 for v in rgb)
        s = 3 if state == OCCUPIED else 0.8
        ax.scatter(x, y, z, c=[c], s=s, alpha=0.7)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_zlim(-2,   4)
    ax.view_init(elev=25, azim=angle)
    ax.set_xlabel('X (m)', color='white', fontsize=7)
    ax.set_ylabel('Y (m)', color='white', fontsize=7)
    ax.set_zlabel('Z (m)', color='white', fontsize=7)
    ax.tick_params(colors='#555', labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


def create_3d_gif(grid, save_path, n_angles=36):
    print(f"  building 3D GIF ({n_angles} angles)...")
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor('#0a0a0a')
    ax  = fig.add_subplot(111, projection='3d')
    handles = []
    for state in [OCCUPIED, FREE_CONFIRMED,
                  FREE_ASSUMED, UNKNOWN_DANGER]:
        c = tuple(v/255 for v in STATE_RGB[state])
        handles.append(mpatches.Patch(
            color=c, label=STATE_NAMES[state]))
    fig.legend(handles=handles, loc='lower center',
               ncol=2, facecolor='#111',
               labelcolor='white', fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    frames = []
    angles = np.linspace(0, 360, n_angles,
                         endpoint=False)
    tmp    = os.path.join(RESULTS_DIR, "_tmp_3d.png")
    for angle in angles:
        plot_3d_occupancy_frame(grid, ax, angle)
        fig.suptitle(
            "Neural Occupancy Network  4-State Uncertainty\n"
            "Vamshikrishna Gadde  |  MS Robotics ASU",
            color='white', fontsize=9)
        plt.tight_layout(rect=[0, 0.09, 1, 0.95])
        plt.savefig(tmp, dpi=80, bbox_inches='tight',
                    facecolor='#0a0a0a')
        raw   = imageio.imread(tmp)
        frame = cv2.resize(raw, GIF_3D_SIZE,
                           interpolation=cv2.INTER_AREA)
        frames.append(frame)
    imageio.mimsave(save_path, frames,
                    duration=0.1, loop=0)
    plt.close()
    if os.path.exists(tmp):
        os.remove(tmp)
    print(f"  saved {save_path}")


def plot_confidence_decay(n_frames=20,
                           device='cuda',
                           save_path=None):
    files       = sorted(os.listdir(KITTI_DIR))[:n_frames]
    band_labels = [b[2] for b in BANDS]
    fc_per_band = {l: [] for l in band_labels}
    fa_per_band = {l: [] for l in band_labels}
    ud_per_band = {l: [] for l in band_labels}
    print(f"  measuring uncertainty over {n_frames} frames...")
    for fname in files:
        path = os.path.join(KITTI_DIR, fname)
        grid, _ = process_frame(path, device)
        unc     = measure_uncertainty_by_distance(grid)
        for label in band_labels:
            r = unc.get(label)
            if r:
                fc_per_band[label].append(
                    r['free_conf_pct'])
                fa_per_band[label].append(
                    r['free_assum_pct'])
                ud_per_band[label].append(
                    r['danger_pct'])
    fc_means = [np.mean(fc_per_band[l])
                for l in band_labels]
    fa_means = [np.mean(fa_per_band[l])
                for l in band_labels]
    ud_means = [np.mean(ud_per_band[l])
                for l in band_labels]
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    x = np.arange(len(band_labels))
    w = 0.28
    ax.bar(x - w, fc_means, w, color='#00C864',
           label='FREE_CONFIRMED', linewidth=0)
    ax.bar(x,     fa_means, w, color='#E6C800',
           label='FREE_ASSUMED',   linewidth=0)
    ax.bar(x + w, ud_means, w, color='#8C00B4',
           label='UNKNOWN_DANGER', linewidth=0)
    crossing = None
    for i, label in enumerate(band_labels):
        if fa_means[i] >= fc_means[i]:
            crossing = label
            break
    if crossing:
        cross_x = band_labels.index(crossing)
        ax.axvline(cross_x - 0.5,
                   color='#FF4444', linewidth=2,
                   linestyle='--', alpha=0.9,
                   label='Unsafe boundary')
        ax.text(cross_x - 0.45,
                max(fc_means) * 0.88,
                'UNSAFE\nBOUNDARY',
                color='#FF4444', fontsize=10,
                fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, color='white',
                       fontsize=11)
    ax.set_xlabel('Distance from Sensor',
                  color='white', fontsize=12)
    ax.set_ylabel('Percentage of Voxels (%)',
                  color='white', fontsize=12)
    ax.set_title(
        'Occupancy Confidence Decay vs Distance\n'
        'Vamshikrishna Gadde  |  MS Robotics ASU  '
        '|  Day 10',
        color='white', fontsize=13)
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1a1a',
              labelcolor='white', fontsize=11)
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')
    plt.tight_layout()
    path = save_path or os.path.join(
        RESULTS_DIR, 'confidence_decay.png')
    plt.savefig(path, dpi=130, bbox_inches='tight',
                facecolor='#1a1a1a')
    plt.close()
    print(f"  saved {path}")
    print(f"\n  NOVEL FINDING:")
    for i, label in enumerate(band_labels):
        ratio = (fc_means[i] / fa_means[i]
                 if fa_means[i] > 0 else 999)
        safe  = ("SAFE"    if ratio > 2   else
                 "CAUTION" if ratio > 1.2 else "UNSAFE")
        print(f"    {label:<10} fc:{fc_means[i]:.1f}%  "
              f"fa:{fa_means[i]:.1f}%  "
              f"ratio:{ratio:.1f}x  {safe}")


def plot_resolution_tradeoff(device='cuda'):
    from voxel_grid import X_MAX, Y_MAX, Z_MAX

    resolutions = [0.1, 0.2, 0.5]
    lidar_path  = os.path.join(
        KITTI_DIR, sorted(os.listdir(KITTI_DIR))[5])
    pts     = load_lidar(lidar_path, device=device)
    results = {}
    print("  measuring resolution tradeoff...")

    for res in resolutions:
        nx = int((X_MAX - X_MIN) / res)
        ny = int((Y_MAX - Y_MIN) / res)
        nz = int((Z_MAX - Z_MIN) / res)
        states = torch.zeros(
            (nx, ny, nz), dtype=torch.uint8,
            device=device)

        # CUDA event timing for accurate GPU measurement
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()

        ix = ((pts[:, 0] - X_MIN) / res).long()
        iy = ((pts[:, 1] - Y_MIN) / res).long()
        iz = ((pts[:, 2] - Z_MIN) / res).long()
        valid = (
            (ix >= 0) & (ix < nx) &
            (iy >= 0) & (iy < ny) &
            (iz >= 0) & (iz < nz))
        ix, iy, iz = ix[valid], iy[valid], iz[valid]
        states[ix, iy, iz] = OCCUPIED

        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)

        mem_mb  = (states.element_size() *
                   states.nelement() / 1024 / 1024)
        n_occ   = (states == OCCUPIED).sum().item()
        cov_pct = n_occ / (nx * ny * nz) * 100
        results[res] = {
            'time_ms':   ms,
            'memory_mb': mem_mb,
            'coverage':  cov_pct,
        }
        del states
        torch.cuda.empty_cache()
        print(f"    {res}m: {ms:.2f}ms  "
              f"{mem_mb:.1f}MB  "
              f"{cov_pct:.3f}% coverage")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle(
        'Voxel Resolution Tradeoff Analysis\n'
        'Vamshikrishna Gadde  |  MS Robotics ASU',
        color='white', fontsize=12)
    labels  = [f"{r}m" for r in resolutions]
    colors  = ['#FF6B35', '#00C8FF', '#00FF88']
    metrics = [
        ('time_ms',   'Processing Time (ms)', 'Speed'),
        ('memory_mb', 'GPU Memory (MB)',       'Memory'),
        ('coverage',  'Occupied Coverage (%)', 'Coverage'),
    ]
    for ax, (key, ylabel, title) in zip(axes, metrics):
        ax.set_facecolor('#1a1a1a')
        vals = [results[r][key] for r in resolutions]
        bars = ax.bar(labels, vals, color=colors,
                      linewidth=0, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.02,
                    f"{v:.2f}", ha='center',
                    color='white', fontsize=10,
                    fontweight='bold')
        ax.set_title(title, color='white', fontsize=11)
        ax.set_ylabel(ylabel, color='white', fontsize=9)
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR,
                        'resolution_tradeoff.png')
    plt.savefig(path, dpi=130, bbox_inches='tight',
                facecolor='#1a1a1a')
    plt.close()
    print(f"  saved {path}")


def plot_uncertainty_heatmap(grid):
    conf_np   = grid.confidence.cpu().numpy()
    states_np = grid.states.cpu().numpy()
    conf_2d   = conf_np.max(axis=2)
    known     = (states_np != UNKNOWN).any(axis=2)

    crop_c = bev_orient(conf_2d[BEV_X0:BEV_X1,
                                  BEV_Y0:BEV_Y1])
    crop_k = bev_orient(known[BEV_X0:BEV_X1,
                                BEV_Y0:BEV_Y1])

    pmap       = {UNKNOWN: 0, FREE_CONFIRMED: 1,
                  FREE_ASSUMED: 2, UNKNOWN_DANGER: 3,
                  OCCUPIED: 4}
    state_proj = np.zeros((NX, NY), dtype=np.uint8)
    for z in range(NZ):
        layer = states_np[:, :, z]
        for s in [FREE_CONFIRMED, FREE_ASSUMED,
                  UNKNOWN_DANGER, OCCUPIED]:
            mask   = layer == s
            old_p  = np.vectorize(pmap.get)(state_proj)
            update = mask & (pmap[s] > old_p)
            state_proj[update] = s

    crop_s  = bev_orient(state_proj[BEV_X0:BEV_X1,
                                     BEV_Y0:BEV_Y1])
    rgb_map = np.zeros(
        (crop_s.shape[0], crop_s.shape[1], 3),
        dtype=np.uint8)
    for state, rgb in STATE_RGB.items():
        rgb_map[crop_s == state] = rgb

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle(
        'Spatial Uncertainty Heatmap  Forward 50m Region\n'
        'Vamshikrishna Gadde  |  MS Robotics ASU',
        color='white', fontsize=12)
    extent = [0, 50, -12, 12]

    ax = axes[0]
    ax.set_facecolor('#0d0d0d')
    masked = np.where(crop_k, crop_c, np.nan)
    im = ax.imshow(masked, origin='upper',
                   cmap='RdYlGn', vmin=0, vmax=1,
                   extent=extent, aspect='auto')
    plt.colorbar(im, ax=ax, label='Confidence Score')
    ax.set_title('Confidence Score Map',
                 color='white', fontsize=11)
    ax.set_xlabel('Forward Distance (m)', color='white')
    ax.set_ylabel('Lateral Distance (m)', color='white')
    ax.tick_params(colors='white')

    ax = axes[1]
    ax.set_facecolor('#0d0d0d')
    ax.imshow(rgb_map, origin='upper',
              extent=extent, aspect='auto')
    ax.set_title('4-State Classification Map',
                 color='white', fontsize=11)
    ax.set_xlabel('Forward Distance (m)', color='white')
    ax.set_ylabel('Lateral Distance (m)', color='white')
    ax.tick_params(colors='white')

    handles = []
    for state in [OCCUPIED, FREE_CONFIRMED,
                  FREE_ASSUMED, UNKNOWN_DANGER]:
        c = tuple(v/255 for v in STATE_RGB[state])
        handles.append(mpatches.Patch(
            color=c, label=STATE_NAMES[state]))
    axes[1].legend(handles=handles,
                   facecolor='#1a1a1a',
                   labelcolor='white', fontsize=9,
                   loc='upper right')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR,
                        'uncertainty_heatmap.png')
    plt.savefig(path, dpi=130, bbox_inches='tight',
                facecolor='#1a1a1a')
    plt.close()
    print(f"  saved {path}")


def add_label(img, text, x, y, scale=0.65,
              color=(255, 255, 255), thickness=2):
    font      = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, scale,
                                  thickness)
    cv2.rectangle(img, (x-2, y-h-4),
                  (x+w+2, y+4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale,
                color, thickness, cv2.LINE_AA)


def create_demo_video(model, n_frames=108,
                       device='cuda'):
    lidar_files = sorted(os.listdir(KITTI_DIR))
    img_files   = sorted(os.listdir(KITTI_IMG_DIR))
    n           = min(n_frames, len(lidar_files),
                      len(img_files))
    print(f"  building demo video ({n} frames)...")

    video_path = os.path.join(RESULTS_DIR, 'demo_video.mp4')
    gif_path   = os.path.join(RESULTS_DIR, 'demo_video.gif')
    tmp_chart  = os.path.join(RESULTS_DIR, "_tmp_chart.png")

    gif_frames = []
    writer     = None
    frame_size = None
    white      = (255, 255, 255)
    cyan       = (0, 220, 255)
    grey       = (160, 160, 160)

    for fi in range(n):
        lidar_path = os.path.join(KITTI_DIR,
                                   lidar_files[fi])
        img_path   = os.path.join(KITTI_IMG_DIR,
                                   img_files[fi])
        grid, timings = process_frame(lidar_path, device)
        predict_confidence(model, grid, device)

        cam_img = cv2.imread(img_path)
        if cam_img is None:
            continue
        H, W = cam_img.shape[:2]

        bev     = make_bev_frame(grid)
        bev_bgr = cv2.cvtColor(
            cv2.resize(bev, (W, H)),
            cv2.COLOR_RGB2BGR)

        heatmap = make_conf_heatmap(grid, W, H)

        stats    = grid.state_statistics()
        fig_s, ax_s = plt.subplots(
            figsize=(W/100, H/100), dpi=100)
        fig_s.patch.set_facecolor('#111111')
        ax_s.set_facecolor('#111111')
        s_names  = ['free_confirmed', 'free_assumed',
                    'occupied', 'unknown_danger']
        s_labels = ['Free\nConfirmed', 'Free\nAssumed',
                    'Occupied', 'Unknown\nDanger']
        counts   = [stats[sn]['pct'] for sn in s_names]
        cols     = ['#00C864', '#E6C800',
                    '#DC1E1E', '#9900CC']
        bars     = ax_s.bar(s_labels, counts,
                            color=cols, linewidth=0,
                            width=0.6)
        for bar, v in zip(bars, counts):
            ax_s.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.2,
                f"{v:.1f}%", ha='center',
                color='white', fontsize=8,
                fontweight='bold')
        ax_s.set_title(
            f"Frame {fi+1}/{n}  |  "
            f"{timings['total']:.0f}ms per frame",
            color='white', fontsize=9)
        ax_s.tick_params(colors='white', labelsize=9)
        ax_s.set_ylabel('% of voxel grid',
                        color='white', fontsize=8)
        for sp in ax_s.spines.values():
            sp.set_edgecolor('#333')
        plt.tight_layout(pad=0.5)
        plt.savefig(tmp_chart, dpi=100,
                    bbox_inches='tight',
                    facecolor='#111111')
        plt.close()

        chart = cv2.resize(cv2.imread(tmp_chart),
                           (W, H))
        gap   = 4
        pw    = W * 2 + gap
        ph    = H * 2 + gap + 30
        panel = np.full((ph, pw, 3), 15,
                        dtype=np.uint8)
        panel[30:30+H,     0:W]      = cam_img
        panel[30:30+H,     W+gap:pw] = bev_bgr
        panel[30+H+gap:ph, 0:W]      = heatmap
        panel[30+H+gap:ph, W+gap:pw] = chart

        add_label(panel, "Camera Image",
                  8, 52, 0.65, white)
        add_label(panel, "BEV Occupancy Map",
                  W+gap+8, 52, 0.65, cyan)
        add_label(panel, "Confidence Heatmap",
                  8, 30+H+gap+22, 0.65, cyan)
        add_label(panel, "Voxel State Distribution",
                  W+gap+8, 30+H+gap+22, 0.65, white)
        cv2.putText(
            panel,
            "Neural Occupancy Network  |  "
            "4-State Uncertainty  |  "
            "Vamshikrishna Gadde  |  MS Robotics ASU",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, grey, 1, cv2.LINE_AA)

        if writer is None:
            frame_size = (panel.shape[1], panel.shape[0])
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                8, frame_size)
        writer.write(panel)

        if fi < 20:
            rgb_f = cv2.cvtColor(panel,
                                  cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb_f, (
                frame_size[0]//2,
                frame_size[1]//2))
            gif_frames.append(small)

        if (fi + 1) % 20 == 0:
            print(f"    {fi+1}/{n} frames")

    if writer:
        writer.release()
        print(f"  saved {video_path}")
    if gif_frames:
        imageio.mimsave(gif_path, gif_frames,
                        duration=0.15, loop=0)
        print(f"  saved {gif_path}")
    if os.path.exists(tmp_chart):
        os.remove(tmp_chart)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() \
             else 'cpu'
    print(f"device: {device}")

    model = load_model(device)
    if model is None:
        print("run uncertainty_net.py first")
        exit()
    model.eval()

    lidar_files = sorted(os.listdir(KITTI_DIR))
    frame_path  = os.path.join(KITTI_DIR,
                                lidar_files[5])

    print("\n[1/6] building occupancy grid...")
    grid, timings = process_frame(frame_path, device)
    predict_confidence(model, grid, device)
    print(f"  total: {timings['total']:.1f}ms")

    print("\n[2/6] creating 3D rotating GIF...")
    create_3d_gif(
        grid,
        os.path.join(RESULTS_DIR, 'occupancy_3d.gif'),
        n_angles=36)

    print("\n[3/6] plotting confidence decay...")
    plot_confidence_decay(n_frames=20, device=device)

    print("\n[4/6] plotting resolution tradeoff...")
    plot_resolution_tradeoff(device=device)

    print("\n[5/6] creating uncertainty heatmap...")
    plot_uncertainty_heatmap(grid)

    print("\n[6/6] building demo video...")
    create_demo_video(model, n_frames=108,
                      device=device)

    print("\nall outputs saved to results/")
    print("  occupancy_3d.gif")
    print("  confidence_decay.png")
    print("  resolution_tradeoff.png")
    print("  uncertainty_heatmap.png")
    print("  demo_video.mp4")
    print("  demo_video.gif")