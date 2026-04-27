# Manus To L20 Retargeting Workspace

This repository aggregates the local workspace used to retarget Manus glove hand pose data to the LinkerHand L20 right hand with two approaches:

1. `GeoRT`: the training-based pipeline, including L20 configuration, URDF assets, recorded datasets, checkpoints, analysis scripts, and generated visualizations.
2. `dex-retargeting`: the optimization-based pipeline used here as the no-training replacement for real-time and offline retargeting.

The repository is organized as a clean upload target for GitHub. Build outputs and oversized proprietary SDK binaries are intentionally excluded so the project can be versioned and cloned reliably.

## Repository Layout

```text
.
├── GeoRT/
│   ├── README_L20_GEORT.md
│   ├── README_L20_DEX_RETARGETING.md
│   ├── dex_retargeting/
│   ├── data/
│   ├── analysis/
│   ├── assets/linkerhand_l20_right/
│   └── checkpoint/
├── dex-retargeting/
└── src/ROS2/
    ├── manus_ros2/
    ├── manus_ros2_msgs/
    └── ManusSDK/
```

## What Is Included

- `GeoRT/`
  The main experimental workspace, including:
  local L20 hand configuration,
  Manus ROS2 bridge scripts,
  recorded datasets such as `manus_data.npy`, `OKandFist.npy`, and `TouchEveryFinger.npy`,
  trained checkpoints,
  analysis outputs,
  and the dex-retargeting replacement scripts and docs.

- `dex-retargeting/`
  A local copy of the upstream dex-retargeting project used as the optimizer backend and reference implementation.

- `src/ROS2/`
  The Manus ROS2 publisher packages and the Manus SDK headers needed to build and run the publisher side of the pipeline.

## What Is Excluded

- ROS workspace build artifacts:
  `build/`, `install/`, `log/`

- Local editor or tool metadata:
  `.git/`, `.claude/`, `__pycache__/`

- Oversized proprietary Manus SDK shared libraries:
  `src/ROS2/ManusSDK/lib/libManusSDK.so`
  `src/ROS2/ManusSDK/lib/libManusSDK_Integrated.so`

These two binaries exceed GitHub's normal file size limit and are intentionally omitted from version control. See `src/ROS2/ManusSDK/lib/README.md` for restore instructions.

## Main Entry Points

### GeoRT L20 pipeline

- `GeoRT/README_L20_GEORT.md`
  Original training-based Manus -> L20 instructions.

### dex-retargeting replacement

- `GeoRT/README_L20_DEX_RETARGETING.md`
  Final no-training Manus -> L20 instructions using dex-retargeting.

- `GeoRT/dex_retargeting/manus_l20_dex_retarget.py`
  Offline `.npy` replay and real-time ROS2 retargeting script.

- `GeoRT/dex_retargeting/visualize_manus_l20.py`
  Offline visualization script for human hand vs. L20 retargeting trajectories.

## Generated Results Already Included

- Offline dex-retargeting result:
  `GeoRT/analysis/manus_l20_dex_retargeting.npz`

- Full-sequence visualizations:
  `GeoRT/analysis/manus_l20_visualization.gif`
  `GeoRT/analysis/OKandFist_full.gif`
  `GeoRT/analysis/TouchEveryFinger_full.gif`

## Recommended Usage

### 1. Read the dex-retargeting instructions

Start from:

- `GeoRT/README_L20_DEX_RETARGETING.md`

### 2. Verify local datasets

Key datasets are under:

- `GeoRT/data/manus_data.npy`
- `GeoRT/data/OKandFist.npy`
- `GeoRT/data/TouchEveryFinger.npy`

### 3. Run offline retargeting

From `GeoRT/`:

```bash
/usr/bin/python3 dex_retargeting/manus_l20_dex_retarget.py \
  --input npy \
  --npy-path data/manus_data.npy \
  --output-path analysis/manus_l20_dex_retargeting.npz \
  --joint-order geort \
  --scaling 1.8
```

### 4. Run offline visualization

```bash
/usr/bin/python3 dex_retargeting/visualize_manus_l20.py \
  --input-path data/manus_data.npy \
  --output-path analysis/manus_l20_visualization.gif \
  --frame-step 1 \
  --fps 30
```

### 5. Run real-time ROS2 retargeting

```bash
source /opt/ros/humble/setup.bash
source /path/to/your/ros2_ws/install/setup.bash
cd GeoRT

/usr/bin/python3 dex_retargeting/manus_l20_dex_retarget.py \
  --input ros \
  --glove-id 0 \
  --publish-topic /l20_dex_retarget/joint_states \
  --joint-order geort \
  --scaling 1.8
```

## Notes For GitHub Upload

- This repository is prepared as a new aggregate project rather than pushing directly to the original `facebookresearch/GeoRT` or `dexsuite/dex-retargeting` remotes.
- Before pushing, set the remote to your own GitHub repository.
- If the Manus SDK `.so` files are required on another machine, restore them manually into:
  `src/ROS2/ManusSDK/lib/`

## Local Provenance

Prepared from the local ROS2 workspace at:

`/home/user/ros2_ws`

GitHub identity intended for upload:

- username: `flf041106`
- email: `fanlinfeng23@gmail.com`
