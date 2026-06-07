# Day 10 - Neural Occupancy Network with 4-State Uncertainty

> MS Robotics & Autonomous Systems Engineering - Arizona State University - Dec 2026

---

## The Question Nobody Asked at Student Level

Every occupancy grid marks voxels as *occupied* or *free*.
But there is a third state that matters more than either.

**What about the space your sensor never reached?**

A voxel marked FREE because a laser confirmed it is empty is fundamentally different from a voxel marked FREE because nobody bothered to check. Planning into the second kind of space is how autonomous vehicles crash.

This project builds a probabilistic occupancy system with four distinct states per voxel, measures where uncertainty becomes dominant, and finds the exact distance at which path planning becomes unsafe - something no published student portfolio has measured.

---

## Live Demo - 108 KITTI Frames

![Demo Video](https://drive.google.com/uc?id=1ZNbcSl2gKXoPitwYaoxb7tBR4fgRdML5)

*4-panel: camera image | BEV occupancy map | confidence heatmap | voxel state distribution*

---

## Rotating 3D Occupancy Grid

![3D Occupancy GIF](https://drive.google.com/uc?id=1bHgaAstUTAPTymXPdx_GJSbW27FPeSHz)

*360-degree rotation of the 4-state voxel grid around the sensor. Red = occupied. Green = free confirmed. Yellow = free assumed. Purple = unknown danger.*

---

## The Novel Finding

![Confidence Decay](https://drive.google.com/uc?id=1FKh7jlVUI8Rrb7x4ALEbHn1Ek_rhnXQH)

The chart above shows the core finding of this project.

At close range the system has strong evidence - FREE_CONFIRMED voxels dominate 6:1 over FREE_ASSUMED. As distance increases, the LiDAR scan lines spread apart, occlusion shadows grow larger, and the ratio collapses.

**At 40-50m from the sensor, FREE_CONFIRMED and FREE_ASSUMED voxels are equal in count.**

Beyond this boundary the system cannot distinguish confirmed free space from assumed free space. A path planner operating beyond 40m on this sensor configuration is making guesses, not decisions. This is the unsafe planning boundary.

| Distance | FREE_CONFIRMED | FREE_ASSUMED | Ratio | Status |
|----------|---------------|--------------|-------|--------|
| 0-10m    | 15.5%         | 2.5%         | 6.2x  | SAFE |
| 10-20m   | 14.9%         | 2.6%         | 5.7x  | SAFE |
| 20-30m   | 6.5%          | 2.2%         | 3.0x  | SAFE |
| 30-40m   | 2.7%          | 1.5%         | 1.8x  | CAUTION |
| 40-50m   | 0.7%          | 0.7%         | 1.0x  | **UNSAFE** |

---

## Spatial Uncertainty Map

![Uncertainty Heatmap](https://drive.google.com/uc?id=1RpGTAAaOExDQ65ytObFhbLKJWclhhsSD)

Left panel shows the neural network confidence score across the forward 50m region. Green = the system is confident. Red/yellow = the system is uncertain. The scan line pattern is visible - high confidence stripes where laser beams hit, low confidence gaps between them.

Right panel shows the 4-state classification. The yellow band ahead is the road surface classified as FREE_ASSUMED. Purple rings around objects are UNKNOWN_DANGER zones - the occlusion shadows where another vehicle could be hiding.

---

## Resolution Tradeoff Analysis

![Resolution Tradeoff](https://drive.google.com/uc?id=1gJkRCJMM7N_6FCHNfkaC84TCsAil80i1)

Three voxel resolutions benchmarked on the same LiDAR scan using CUDA event timing:

| Resolution | Speed | GPU Memory | Coverage |
|-----------|-------|------------|----------|
| 0.1m | highest | 57.2 MB | 0.05% |
| 0.2m | optimal | 7.2 MB | 0.22% |
| 0.5m | fastest | 0.5 MB | 1.06% |

**0.2m is the correct choice.** 0.1m uses 8x more memory for minimal gain. 0.5m misses objects smaller than 0.5m - pedestrians are approximately 0.4m wide and would be skipped entirely.

---

## The 4-State System

Most occupancy grids use three states: occupied, free, unknown. This system uses four, and the distinction between the last two is the engineering contribution.

```
OCCUPIED-CONFIRMED
  A LiDAR beam hit this voxel directly.
  Something physical is here.
  Confidence 0.95+

FREE-CONFIRMED
  A LiDAR beam passed through this voxel
  on its way to hit something further away.
  If anything were here the beam would have
  returned early. It did not.
  Confidence 0.90+

FREE-ASSUMED
  No beam hit or passed through this voxel.
  But surrounding voxels are FREE-CONFIRMED.
  Probably empty. Not proven.
  Confidence ~0.55

UNKNOWN-DANGER
  No beam data. Adjacent to an OCCUPIED voxel.
  Could be in the occlusion shadow of an object.
  Something may be hiding here.
  Confidence ~0.38
```

The difference between FREE-ASSUMED and UNKNOWN-DANGER is not academic. A planner that treats them identically will occasionally drive into the side of an occluded truck.

---

## Neural Network - Why It Is Not Just Geometry

Ray casting gives us OCCUPIED and FREE-CONFIRMED with certainty. It cannot classify the remaining 94% of voxels.

The 3D CNN fills this gap. It looks at the 5x5x5 neighborhood around each unclassified voxel and predicts whether the center is likely occupied or free based on the surrounding context.

```
Architecture:
  Input:   (batch, 1, 5, 5, 5) voxel neighborhood
  Layer 1: Conv3D(1→16) + BatchNorm + ReLU
  Layer 2: Conv3D(16→32) + BatchNorm + ReLU
  Layer 3: Conv3D(32→64) + BatchNorm + ReLU
  Pool:    AdaptiveAvgPool3D(1)
  FC:      Linear(64→32) → ReLU → Dropout(0.3)
  Output:  Linear(32→1) → Sigmoid → confidence [0,1]
  Params:  72,001
```

Training on 200 frames of kitti_object (400,000 samples, balanced):

```
epoch 1:  loss 0.0572  acc 98.4%
epoch 3:  loss 0.0013  acc 100.0%
epoch 8:  loss 0.0001  acc 100.0%
```

---

## Pipeline Performance

```
voxelization   : 35ms    GPU tensor scatter
ray casting    : 53ms    vectorized GPU beam march
CNN inference  : 229ms   batched forward pass 30k voxels
total          : ~282ms per frame on RTX 4050
training data  : 400,000 samples across 200 KITTI frames
```

---

## How This Connects to the Series

```
Day 1:  RANSAC + DBSCAN detector - found objects
Day 7:  fog unsafe below 75m - sensor ODD boundary
Day 8:  depth completion - fused LiDAR and camera
Day 9:  domain shift - 58.4% drop across continents

Day 10: occupancy network - answers the deeper question
        not just WHERE objects are
        but WHERE space is SAFE to plan through
        and WHERE the system simply does not know

This is the representation that feeds a motion planner.
Days 1-9 built the perception.
Day 10 builds the world model.
```

---

## Run It Yourself

```bash
git clone https://github.com/GVK-Engine/day-010-occupancy-network
cd day-010-occupancy-network
pip install -r requirements.txt
```

Update KITTI paths in each file to your local directory.

```bash
# test GPU voxelization
py -3.11 voxel_grid.py

# run ray casting and 4-state classification
py -3.11 occupancy.py

# train 3D CNN on kitti_object (15 mins on RTX 4050)
py -3.11 uncertainty_net.py

# generate all visual outputs
py -3.11 visualize.py
```

KITTI dataset: https://www.cvlibs.net/datasets/kitti/raw_data.php

---

## Project Structure

```
day-010-occupancy-network/
├── voxel_grid.py        GPU voxelization and grid management
├── occupancy.py         Ray casting and 4-state classification
├── uncertainty_net.py   3D CNN training and inference
├── visualize.py         All visual outputs and demo video
├── requirements.txt
└── results/
    ├── occupancy_3d.gif
    ├── demo_video.mp4
    ├── demo_video.gif
    ├── confidence_decay.png
    ├── resolution_tradeoff.png
    └── uncertainty_heatmap.png
```

---

## Stack

`Python 3.11` `PyTorch 2.6` `CUDA 12.4` `NumPy` `OpenCV` `Matplotlib` `imageio` `KITTI`

---

## Series 1 Progress

| # | Project | Finding | Status |
|---|---------|---------|--------|
| P1.1 | LiDAR Obstacle Detection | 0.4m voxel creates ghost detections | ✅ |
| P1.2 | Stereo Camera Depth Safety | Camera unsafe beyond 10m | ✅ |
| P1.3 | PointPillars 3D Detector | 98.9% loss reduction from scratch | ✅ |
| P1.4 | Multi-Camera BEV Perception | 178 objects from 6 cameras | ✅ |
| P1.5 | Multi-Object Tracking SORT | Detector is bottleneck not tracker | ✅ |
| P1.6 | Semantic Segmentation ROS2 | 52.6 FPS - warmup cost measured | ✅ |
| P1.7 | Adverse Weather Analysis | Fog unsafe below 75m visibility | ✅ |
| P1.8 | LiDAR-Camera Depth Completion | 44x MAE improvement at 0-10m | ✅ |
| P1.9 | Domain Shift Analysis | 58.4% drop - sensor not scene | ✅ |
| P1.10 | Neural Occupancy Network | Unsafe boundary at 40m - 4-state uncertainty | ✅ |
