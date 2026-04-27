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

## L20 Content Status

The L20-specific content used in this project is already included in the repository:

- `GeoRT/assets/linkerhand_l20_right/`
- `GeoRT/geort/config/linkerhand_l20_right.json`
- `GeoRT/dex_retargeting/linkerhand_l20_right_vector.yml`
- `GeoRT/data/*.npy`
- `GeoRT/checkpoint/`
- `GeoRT/analysis/`

In practice, collaborators do **not** need to separately fetch the L20 URDF, meshes, retargeting configs, datasets, or generated visualizations.

The main content that still needs to be restored manually is the MANUS SDK shared library pair under:

```text
src/ROS2/ManusSDK/lib/
```

## Quick Start For Collaborators

If you are cloning this repository on a new machine, use this order:

1. clone the repository,
2. restore the Manus SDK shared libraries into `src/ROS2/ManusSDK/lib/`,
3. build the ROS2 publisher workspace,
4. run either the GeoRT or dex-retargeting workflow.

The fastest path for new collaborators is to start from:

- `GeoRT/README_L20_DEX_RETARGETING.md`

because it avoids retraining and directly supports offline replay and real-time retargeting.

## Prerequisites

### Python

The dex-retargeting replacement workflow in `GeoRT/` is expected to run with:

```bash
/usr/bin/python3
```

### ROS2

For Manus ROS2 publisher and real-time retargeting:

```bash
source /opt/ros/humble/setup.bash
```

### Manus SDK

The repository does not include the large Manus SDK shared libraries. Collaborators must restore:

- `src/ROS2/ManusSDK/lib/libManusSDK.so`
- `src/ROS2/ManusSDK/lib/libManusSDK_Integrated.so`

See:

- `src/ROS2/ManusSDK/lib/README.md`

for placement details.

## Restoring Excluded Content

### 1. Manus SDK shared libraries

These are not tracked in git.

According to the official MANUS documentation:

- the C++ SDK and ROS2 package are distributed together as part of the `MANUS Core 3 SDK (including ROS2 Package)` download,
- the download is available from the MANUS Download Center after creating a free account or logging in,
- the SDK zip contains Linux and Windows examples plus a `ROS2` folder,
- on Linux, `integrated` mode works directly on Linux,
- on Linux, `remote` mode still requires a separate Windows machine running MANUS Core on the network because MANUS Core itself runs on Windows.

Official references:

- MANUS SDK getting started:
  `https://docs.manus-meta.com/3.1.0/Plugins/SDK/getting%20started/`
- Linux SDK guide:
  `https://docs.manus-meta.com/3.1.0/Plugins/SDK/Linux/`
- ROS2 getting started:
  `https://docs.manus-meta.com/3.1.0/Plugins/SDK/ROS2/getting%20started/`
- Latest MANUS downloads page:
  `https://docs.manus-meta.com/latest/Resources/`

### 1.1 Download Procedure From MANUS

Follow this sequence:

1. Open the MANUS SDK getting-started page listed above.
2. Go to the MANUS Download Center from that page.
3. Create a free MANUS account or log in to your existing account.
4. Open the latest downloads page.
5. Under `MANUS Core 3.1`, download:

```text
MANUS Core 3 SDK (including ROS2 Package)
```

At the time reflected by the documentation page, the listed SDK version is:

```text
3.1.1
```

### 1.2 Expected ZIP Contents

The MANUS SDK getting-started guide says the extracted archive contains at least:

- `SDKClient_Linux`
- `SDKClient_Windows`
- `SDKMinimalClient_Linux`
- `SDKMinimalClient_Windows`
- `ROS2`

For this repository, the critical directory to recover is the SDK library folder containing:

- `libManusSDK.so`
- `libManusSDK_Integrated.so`

These are expected under the SDK package's `ManusSDK/lib/` directory.

### 1.3 Restore Into This Repository

After downloading and extracting the official MANUS SDK package, copy the two Linux shared libraries into:

```text
src/ROS2/ManusSDK/lib/
```

Expected contents after restore:

```text
src/ROS2/ManusSDK/lib/
├── libManusSDK.so
└── libManusSDK_Integrated.so
```

If your extracted SDK lives at `/path/to/MANUS_SDK`, the copy command is:

```bash
cp /path/to/MANUS_SDK/ManusSDK/lib/libManusSDK.so \
   src/ROS2/ManusSDK/lib/

cp /path/to/MANUS_SDK/ManusSDK/lib/libManusSDK_Integrated.so \
   src/ROS2/ManusSDK/lib/
```

### 1.4 Which Library To Use

From the MANUS Linux SDK guide:

- `libManusSDK.so`
  standard library name used by the remote workflow,
- `libManusSDK_Integrated.so`
  library used when running Linux in `integrated mode` without the remote stack.

The Linux guide explicitly notes that if you are using `integrated mode` only, you may point your build to `libManusSDK_Integrated.so`, or rename it to `libManusSDK.so` and replace the original library in the package directory.

### 1.5 License Requirements

The official MANUS documentation states:

- Linux SDK functionality requires either a `MANUS Bodypack` or a `MANUS license key` with the `SDK` feature enabled.
- The `SDK Integrated` documentation further states that the integrated functionality requires a `Feature` style license with the `SDK` feature enabled, and advises contacting `support@manus-meta.com` if you do not have that license.

### 1.6 Linux Prerequisites From MANUS Docs

The MANUS Linux guide lists these supported Ubuntu versions:

- 20.04
- 22.04
- 24.04

For `integrated mode`, the documented package install command is:

```bash
sudo apt-get update && sudo apt-get install -y \
  build-essential \
  libusb-1.0-0-dev \
  zlib1g-dev \
  libudev-dev \
  gdb \
  libncurses5-dev && sudo apt-get clean
```

For `remote mode`, the MANUS docs additionally list packages such as:

- `git`
- `libtool`
- `libzmq3-dev`
- GRPC / Protobuf requirements

### 1.7 Linux Device Rules

The MANUS Linux guide also documents udev rules for hardware access. Create:

```text
/etc/udev/rules.d/70-manus-hid.rules
```

with:

```text
# HIDAPI/libusb
SUBSYSTEMS=="usb", ATTRS{idVendor}=="3325", MODE:="0666"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="83fd", MODE:="0666"

# HIDAPI/hidraw
KERNEL=="hidraw*", ATTRS{idVendor}=="3325", MODE:="0666"
```

The MANUS documentation recommends a full reboot after placing the rules file.

### 2. ROS2 build artifacts

The following are intentionally excluded and must be rebuilt locally:

- `build/`
- `install/`
- `log/`

From the repository root, rebuild with:

```bash
source /opt/ros/humble/setup.bash
colcon build --base-paths src/ROS2
```

After build:

```bash
source install/setup.bash
```

## Build And Run

### Build Manus ROS2 packages

From repository root:

```bash
source /opt/ros/humble/setup.bash
colcon build --base-paths src/ROS2
source install/setup.bash
```

This matches the MANUS ROS2 getting-started guide at a high level:

1. install ROS2,
2. obtain the MANUS C++ SDK,
3. copy the `ROS2` package contents into your workspace `src`,
4. build with `colcon`,
5. source `install/setup.bash`,
6. run `ros2 run manus_ros2 manus_data_publisher`.

### Start Manus publisher

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run manus_ros2 manus_data_publisher
```

### Check Manus topics

```bash
ros2 topic list | grep manus
ros2 topic echo /manus_glove_0 --once
```

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

These datasets are already included and do not require regeneration for offline verification.

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

### 6. Run offline visualization

For a full-sequence visualization without frame subsampling:

```bash
cd GeoRT
/usr/bin/python3 dex_retargeting/visualize_manus_l20.py \
  --input-path data/manus_data.npy \
  --output-path analysis/manus_l20_visualization.gif \
  --frame-step 1 \
  --fps 30
```

## Notes For GitHub Upload

- This repository is prepared as a new aggregate project rather than pushing directly to the original `facebookresearch/GeoRT` or `dexsuite/dex-retargeting` remotes.
- Before pushing, set the remote to your own GitHub repository.
- If the Manus SDK `.so` files are required on another machine, restore them manually into:
  `src/ROS2/ManusSDK/lib/`

## Included Outputs

The repository already includes useful ready-made outputs for review and demonstration:

- `GeoRT/analysis/manus_l20_dex_retargeting.npz`
- `GeoRT/analysis/manus_l20_visualization.gif`
- `GeoRT/analysis/OKandFist_full.gif`
- `GeoRT/analysis/TouchEveryFinger_full.gif`

This allows collaborators to inspect the results immediately before rebuilding or rerunning anything locally.

## Local Provenance

Prepared from the local ROS2 workspace at:

`/home/user/ros2_ws`

GitHub identity intended for upload:

- username: `flf041106`
- email: `fanlinfeng23@gmail.com`
