# GeoRT × LinkerHand L20 全流程教程

> 基于 [GeoRT](https://github.com/facebookresearch/GeoRT) 官方框架，将 Manus 动捕手套数据
> retargeting 到灵心巧手 L20 右手的完整流程。

---

## 目录

1. [依赖与环境](#1-依赖与环境)
2. [获取 L20 URDF](#2-获取-l20-urdf)
3. [配置文件说明](#3-配置文件说明)
4. [硬件连接与 Manus ROS2](#4-硬件连接与-manus-ros2)
5. [启动 Manus → ZMQ 桥接节点](#5-启动-manus--zmq-桥接节点)
6. [录制训练数据](#6-录制训练数据)
7. [训练 GeoRT IK 模型](#7-训练-geort-ik-模型)
8. [实时 Retargeting 可视化](#8-实时-retargeting-可视化)
9. [离线回放验证](#9-离线回放验证)
10. [常见问题](#10-常见问题)
11. [文件结构总览](#11-文件结构总览)

---

## 1. 依赖与环境

### Python 依赖

```bash
cd /home/user/ros2_ws/GeoRT
pip install -r requirements.txt
# 若 requirements.txt 中缺少，手动补充：
pip install pyzmq tqdm scipy
```

### ROS2 环境

```bash
source /home/user/ros2_ws/install/setup.bash
```

### GPU

训练需要 CUDA GPU。确认可用：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 2. 获取 L20 URDF

官方 GeoRT 仓库不含 L20 URDF，需从灵巧手仿真仓库获取：

```bash
git clone https://github.com/linker-bot/linkerhand-sim.git /tmp/linkerhand-sim

# 将 L20 right URDF 及 mesh 文件复制到 GeoRT assets 目录
cd /home/user/ros2_ws/GeoRT
mkdir -p assets/linkerhand_l20_right
cp -r /tmp/linkerhand-sim/linker_hand_isaac_gym_urdf/l20/right/* assets/linkerhand_l20_right/
```

复制完成后，确认以下文件存在：

```
assets/linkerhand_l20_right/
├── linkerhand_l20_right.urdf
└── meshes/          ← STL/DAE 网格文件
```

### 验证 URDF 关节名称

GeoRT 配置中使用的关节名必须与 URDF 完全一致。运行下面命令查看 URDF 中的关节名：

```bash
grep '<joint ' assets/linkerhand_l20_right/linkerhand_l20_right.urdf | grep 'name='
```

期望看到（顺序不限）：
`thumb_joint0`–`thumb_joint4`，`index_joint0`–`index_joint3`，
`middle_joint0`–`middle_joint3`，`ring_joint0`–`ring_joint3`，
`little_joint0`–`little_joint3`

如果实际 URDF 关节名不同（例如 `Thumb_Joint_0` 等），**需要同步修改**
`geort/config/linkerhand_l20_right.json` 中的 `joint_order` 和 `fingertip_link[*].joint`。

---

## 3. 配置文件说明

配置文件位于 `geort/config/linkerhand_l20_right.json`，内容如下：

```json
{
    "name": "linkerhand_l20_right",
    "urdf_path": "./assets/linkerhand_l20_right/linkerhand_l20_right.urdf",
    "base_link": "base_link",
    "joint_order": [
        "thumb_joint0", "thumb_joint1", "thumb_joint2", "thumb_joint3", "thumb_joint4",
        "index_joint0",  "index_joint1",  "index_joint2",  "index_joint3",
        "middle_joint0", "middle_joint1", "middle_joint2", "middle_joint3",
        "ring_joint0",   "ring_joint1",   "ring_joint2",   "ring_joint3",
        "little_joint0", "little_joint1", "little_joint2", "little_joint3"
    ],
    "fingertip_link": [
        {"name":"thumb",  "link":"thumb_link5",  "joint":["thumb_joint0",...,"thumb_joint4"], "center_offset":[0,0,0], "human_hand_id":4},
        {"name":"index",  "link":"index_link4",  "joint":["index_joint0",...,"index_joint3"], "center_offset":[0,0,0], "human_hand_id":8},
        ...
    ]
}
```

**关键字段说明：**

| 字段 | 说明 |
|------|------|
| `joint_order` | 训练时关节的排列顺序，决定 IK 输出向量的维度顺序 |
| `fingertip_link` | 每根手指的末端 link 名，用于前向运动学计算指尖位置 |
| `human_hand_id` | 对应 MediaPipe 21 关键点中的指尖 ID（拇4、食8、中12、环16、小20）|
| `base_link` | 机械手的根坐标系 link 名 |

---

## 4. 硬件连接与 Manus ROS2

### 连接步骤

1. 将 Manus USB dongle 插入 Linux 主机
2. 戴上 Manus 手套并确认配对
3. 启动 manus_ros2 节点（发布 `/manus_glove_0`）：

```bash
source /home/user/ros2_ws/install/setup.bash
ros2 run manus_ros2 manus_data_publisher
```

4. 验证话题正在发布：

```bash
ros2 topic hz /manus_glove_0
# 期望：约 30–120 Hz
```

5. 验证消息结构（有数据后）：

```bash
ros2 topic echo /manus_glove_0 --once
# 查看 raw_nodes 中是否有 21 个节点，node_id 从 0 到 20
```

---

## 5. 启动 Manus → ZMQ 桥接节点

> **说明**：官方 GeoRT 的 `manus_mocap_core.py` 要求 Windows Manus Core Server
> 通过特定 ROS 话题发送旋转数据，与本套 `manus_ros2` SDK 不兼容。
> `manus_ros2_bridge.py` 是专门为本设置编写的桥接层，功能等效。

```bash
# 终端 1：确保 manus_data_publisher 已运行
# 终端 2：
cd /home/user/ros2_ws/GeoRT
export PYTHONPATH="$(pwd):${PYTHONPATH}"
source /home/user/ros2_ws/install/setup.bash
python geort/mocap/manus_ros2_bridge.py --glove_id 0 --port 8765
```

桥接节点将：
1. 订阅 `/manus_glove_0`
2. 从 `raw_nodes` 提取 21 个节点的世界坐标
3. 应用 `hand_to_canonical()` 转换到腕关节本地正则坐标系
4. 通过 ZMQ PUB（端口 8765）发布 `float32 (21×3)` 字节流

### 坐标系约定（GeoRT canonical frame）

| 轴 | 含义 |
|----|------|
| X  | 手掌法线方向（手背→手心） |
| Y  | 腕→拇指根方向 |
| Z  | 腕→中指根方向 |
| 原点 | 腕关节（node_id=0）|

---

## 6. 录制训练数据

桥接节点运行期间，录制动捕数据：

```bash
# 终端 3（GeoRT 根目录）：
cd /home/user/ros2_ws/GeoRT
export PYTHONPATH="$(pwd):${PYTHONPATH}"

python record_manus.py \
    --name manus_data \
    --duration 120 \
    --fps 30
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--name` | 输出文件名（保存到 `data/<name>.npy`）| `manus_data` |
| `--duration` | 录制时长（秒）| `60` |
| `--fps` | 采样帧率 | `30` |

录制期间，请做各种手势：
- 张开/握拳
- 各手指独立弯曲
- 拇指与各手指对捏
- 手腕旋转（正则变换会消除手腕姿态影响，但多样动作有助于训练）

录制完成后文件保存为：

```
data/manus_data.npy     # shape: (N_frames, 21, 3)  dtype: float32
```

---

## 7. 训练 GeoRT IK 模型

### 方法一：直接运行脚本（推荐）

```bash
cd /home/user/ros2_ws/GeoRT
bash train_l20.sh
```

可通过环境变量调整超参数：

```bash
HUMAN_DATA=manus_data \
TAG=manus_l20_v1 \
W_CHAMFER=80 \
W_PINCH=1.0 \
bash train_l20.sh
```

### 方法二：手动命令

```bash
cd /home/user/ros2_ws/GeoRT
export PYTHONPATH="$(pwd):${PYTHONPATH}"

python geort/trainer.py \
    -hand linkerhand_l20_right \
    -human_data manus_data \
    -ckpt_tag manus_l20 \
    --w_chamfer 80.0 \
    --w_curvature 0.1 \
    --w_collision 0.0 \
    --w_pinch 1.0
```

### 训练过程说明

训练分两个阶段自动完成：

**阶段 1 — 生成机器人运动学数据集**（首次运行）

系统自动采样 100,000 个随机关节角，通过 SAPIEN 运动学计算对应指尖位置，保存到：
```
data/linkerhand_l20_right.npz
```

**阶段 2 — 训练神经 FK 模型**（首次运行）

训练一个小型 MLP 近似机器人正向运动学，保存到：
```
checkpoint/fk_model_linkerhand_l20_right.pth
```

**阶段 3 — 训练 IK 模型**（主要训练）

使用录制的 Manus 数据训练 IK 映射网络，每个 epoch 保存 checkpoint：
```
checkpoint/linkerhand_l20_right_<timestamp>_manus_l20/
├── config.json       ← 关节限位等配置（推理时必需）
├── epoch_0.pth
├── epoch_1.pth
├── ...
└── last.pth          ← 最新 epoch

checkpoint/linkerhand_l20_right_last/
└── last.pth          ← 方便直接引用的最新副本
```

### 损失函数说明

| 损失项 | 权重 | 作用 |
|--------|------|------|
| Direction Loss | 1.0 | 保持人手运动方向与机器人响应方向一致（核心） |
| Chamfer Loss | 80.0 | 将机器人工作空间对齐人手工作空间 |
| Curvature Loss | 0.1 | 使映射平滑（避免局部抖动） |
| Pinch Loss | 1.0 | 保持拇指-手指对捏关系 |

---

## 8. 实时 Retargeting 可视化

> **前提**：桥接节点（步骤 5）正在运行，训练已完成。

```bash
cd /home/user/ros2_ws/GeoRT
bash evaluate_l20.sh
```

或等价的手动命令（必须包含 Vulkan ICD 设置）：

```bash
cd /home/user/ros2_ws/GeoRT
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"
source /home/user/ros2_ws/install/setup.bash

python geort/mocap/manus_evaluation.py \
    -hand linkerhand_l20_right \
    -ckpt_tag manus_l20
```

> **注意**：
> - 不要在可视化终端中 `source /home/user/ros2_ws/install/setup.bash`。
>   ROS2 会向 `LD_LIBRARY_PATH` 注入版本冲突的共享库，导致 SAPIEN 渲染器段错误崩溃。
>   评估脚本通过 ZMQ 接收数据，不依赖 ROS2 运行时。
>   `bash evaluate_l20.sh` 已正确处理，无需手动设置。

程序将：
1. 加载 `checkpoint/linkerhand_l20_right_*_manus_l20/last.pth` 中的 IK 模型
2. 从 ZMQ（端口 8765）实时接收正则化手部关键点
3. 推理得到 L20 关节角度
4. 在 SAPIEN 仿真窗口中实时驱动 L20 模型

**预期效果**：SAPIEN 窗口中的 L20 灵巧手跟随你的实际手部动作实时运动。

### 调整视角

SAPIEN Viewer 键鼠操作：
- 鼠标右键拖动：旋转视角
- 鼠标中键拖动：平移
- 滚轮：缩放

---

## 9. 离线回放验证

如果不方便实时运行，可以用录制的数据做回放验证：

```bash
cd /home/user/ros2_ws/GeoRT
bash evaluate_l20.sh --replay manus_data
```

或手动命令：

```bash
cd /home/user/ros2_ws/GeoRT
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"

python geort/mocap/replay_evaluation.py \
    -hand linkerhand_l20_right \
    -ckpt_tag manus_l20 \
    -data manus_data
```

程序循环播放 `data/manus_data.npy` 中的每一帧，直观验证 retargeting 质量。

---

## 10. 常见问题

### Q: 运行 trainer.py 时报 `ModuleNotFoundError: No module named 'geort'`

必须在 GeoRT 根目录下设置 PYTHONPATH：

```bash
cd /home/user/ros2_ws/GeoRT
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

或直接使用 `bash train_l20.sh`（脚本内已自动处理）。

### Q: SAPIEN 报 `AttributeError: PhysxSceneConfig has no attribute ...`

已在 `geort/env/hand.py` 中修复（兼容 SAPIEN 2.x 和 3.x）。
确认你使用的是本仓库中已修改的版本。

### Q: `raw_nodes` 数量不是 21，或 node_id 不是 0–20

不同版本的 Manus SDK 可能有不同的 node_id 编号方案。
运行以下命令查看实际 node_id：

```bash
ros2 topic echo /manus_glove_0 --once 2>/dev/null | grep node_id
```

如果 node_id 不是 0–20，需在 `manus_ros2_bridge.py` 的 `_glove_cb` 中添加映射表：

```python
# 示例：如果 SDK 用 1-21 而非 0-20
NODE_ID_REMAP = {i+1: i for i in range(21)}
keypoints = np.array([positions[NODE_ID_REMAP[i]] for i in range(21)])
```

### Q: 训练 loss 不收敛 / retargeting 效果差

1. **数据量不足**：增加录制时长（建议 ≥ 120 秒，覆盖多种手势）
2. **Chamfer 权重过高**：尝试 `W_CHAMFER=40`
3. **训练轮数不够**：默认 200 epoch，可增加到 500
4. **URDF 与配置不匹配**：检查关节名、base_link 名

### Q: 如何检查 Checkpoint 是否正常加载

```bash
cd /home/user/ros2_ws/GeoRT
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python -c "
from geort import load_model
m = load_model(tag='manus_l20')
print('Model loaded OK. Human IDs:', m.human_ids)
"
```

### Q: SAPIEN 窗口打开后手指不停乱动（即使没有手套数据）

原因：SAPIEN PD 控制器默认目标角度为 0，若未收到 ZMQ 数据则持续把所有关节驱向 0。
已在 `geort/env/hand.py` 中修复：初始化时将 drive target 设为关节中间值。

### Q: SAPIEN 渲染器段错误（`段错误 (核心已转储)`）

原因：在运行评估脚本的终端中执行了 `source /home/user/ros2_ws/install/setup.bash`，
导致 ROS2 路径（`LD_LIBRARY_PATH`）注入与 SAPIEN 依赖冲突的共享库版本，SAPIEN 崩溃。

修复：**评估脚本不需要 ROS2 环境**。在干净的终端（未 source ROS2）中运行：

```bash
# 在未 source ROS2 的新终端中（只激活 conda geort 环境）：
conda activate geort
cd /home/user/ros2_ws/GeoRT
bash evaluate_l20.sh
```

`bash evaluate_l20.sh` 本身已正确处理所有环境变量，无需额外操作。

### Q: SAPIEN Viewer 窗口没有出现

确认系统有 display（X11 或 Wayland）：

```bash
echo $DISPLAY
# 本地桌面：应有值如 :0
# SSH 远程：需要 X11 转发（ssh -X）或 VNC
```

---

## 11. 文件结构总览

```
GeoRT/
├── geort/
│   ├── config/
│   │   └── linkerhand_l20_right.json      ← L20 配置（已创建）
│   ├── env/
│   │   └── hand.py                        ← SAPIEN 3.x 兼容修复（已修改）
│   ├── mocap/
│   │   ├── manus_mocap.py                 ← 官方 ZMQ 订阅器
│   │   ├── manus_mocap_core.py            ← 官方 Windows-Manus 桥（参考用）
│   │   ├── manus_ros2_bridge.py           ← 本机 manus_ros2 → ZMQ 桥（已创建）
│   │   ├── manus_evaluation.py            ← 官方实时可视化
│   │   └── replay_evaluation.py           ← 官方离线回放可视化
│   ├── trainer.py                         ← GeoRT 训练主程序
│   └── export.py                          ← 模型加载接口
├── assets/
│   └── linkerhand_l20_right/              ← 需手动从 linkerhand-sim 复制
│       ├── linkerhand_l20_right.urdf
│       └── meshes/
├── data/
│   └── manus_data.npy                     ← 录制后生成 (N, 21, 3)
├── checkpoint/
│   └── linkerhand_l20_right_*_manus_l20/  ← 训练后生成
│       ├── config.json
│       └── last.pth
├── record_manus.py                        ← 数据录制脚本（已创建）
├── train_l20.sh                           ← 训练启动脚本（已创建）
└── README_L20_GEORT.md                    ← 本文档
```

---

## 完整流程速查

```bash
# ── 一次性准备 ────────────────────────────────────────────────
# 1. 获取 URDF
git clone https://github.com/linker-bot/linkerhand-sim.git /tmp/linkerhand-sim
mkdir -p /home/user/ros2_ws/GeoRT/assets/linkerhand_l20_right
cp -r /tmp/linkerhand-sim/linker_hand_isaac_gym_urdf/l20/right/* \
      /home/user/ros2_ws/GeoRT/assets/linkerhand_l20_right/

# ── 每次使用前 ───────────────────────────────────────────────
source /home/user/ros2_ws/install/setup.bash
cd /home/user/ros2_ws/GeoRT
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# 2. 启动 Manus ROS2 节点（终端 A）
ros2 run manus_ros2 manus_data_publisher

# 3. 启动 ZMQ 桥（终端 B）
python geort/mocap/manus_ros2_bridge.py --glove_id 0

# 4. 录制训练数据（终端 C，120 秒，做各种手势）
python record_manus.py --name manus_data --duration 120

# 5. 训练（终端 C，约 30–60 分钟）
bash train_l20.sh

# 6. 实时可视化（终端 C，桥接节点继续运行）
bash evaluate_l20.sh
```
