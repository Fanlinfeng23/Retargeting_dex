# Manus -> 灵心巧手 L20：`dex-retargeting` 最终使用说明

本文档是 `GeoRT/README_L20_GEORT.md` 的 `dex-retargeting` 替代版，目标是把 Manus 手套提取到的人手位姿，直接重定向到灵心巧手 L20 右手，不再经过 GeoRT 的训练式 IK 网络。

这个方案已经在当前工作区完成了以下落地：

1. 复用现有 `GeoRT/assets/linkerhand_l20_right` 下的 L20 URDF 和 mesh。
2. 使用 `dex-retargeting` 的 `vector` 接口做无训练实时优化。
3. 兼容 Manus ROS2 话题 `/manus_glove_0`。
4. 自动处理 L20 URDF 中的 mimic joints。
5. 支持离线 `.npy` 回放测试和实时 ROS2 发布 `JointState`。

---

## 1. 方案概览

### 1.1 方案和 GeoRT 的区别

原 GeoRT 方案链路是：

```text
Manus -> 21x3 人手关键点 -> 录数据 -> 训练 FK/IK 模型 -> 实时推理 -> L20 qpos
```

本方案链路是：

```text
Manus -> 21x3 人手关键点 -> canonical 手坐标 -> dex-retargeting 在线优化 -> L20 qpos
```

也就是说：

- 不需要重新训练模型。
- 不依赖 GeoRT 的 checkpoint 推理。
- 更适合直接替代“实时 retargeting”这一段。

### 1.2 为什么选 `vector` 接口

`dex-retargeting` 有 `position` / `vector` / `dexpilot` 三类接口。这里选 `vector`，原因是：

1. Manus 数据天然就是 `21 x 3` 的人手关键点。
2. 你当前的 GeoRT-L20 配置，本质上也是基于“手腕到各关键点”的几何约束。
3. `vector` 方式最容易复用 `README_L20_GEORT.md` 里的关键点定义。
4. 对 L20 这种带 mimic joints 的手，`dex-retargeting` 的处理逻辑比较直接稳定。

---

## 2. 相关文件

本方案的核心文件如下：

```text
/home/user/ros2_ws/GeoRT/
├── README_L20_DEX_RETARGETING.md
└── dex_retargeting/
    ├── linkerhand_l20_right_vector.yml
    └── manus_l20_dex_retarget.py
```

文件作用：

- [linkerhand_l20_right_vector.yml](/home/user/ros2_ws/GeoRT/dex_retargeting/linkerhand_l20_right_vector.yml)
  `dex-retargeting` 的 L20 配置，定义 URDF、优化变量、目标 link、关键点映射、缩放系数等。

- [manus_l20_dex_retarget.py](/home/user/ros2_ws/GeoRT/dex_retargeting/manus_l20_dex_retarget.py)
  运行脚本，支持两种模式：
  `--input npy` 离线回放。
  `--input ros` 实时订阅 Manus ROS2 数据。

---

## 3. 环境要求

### 3.1 Python 解释器

这里必须用系统 Python：

```bash
/usr/bin/python3 --version
```

期望是：

```text
Python 3.10.x
```

不要用：

- 默认 conda `python 3.13`
- `miniconda3/envs/geort` 里的 `python 3.8`

原因：

1. ROS2 Humble 的 `rclpy` 绑定是给 Python 3.10 编译的。
2. 当前机器上的 `nlopt` / `pinocchio` / `pytransform3d` / `zmq` 也都在系统 Python 3.10 侧可用。

### 3.2 ROS2 环境

实时模式前需要 source：

```bash
source /opt/ros/humble/setup.bash
source /home/user/ros2_ws/install/setup.bash
```

### 3.3 L20 URDF 资源

需要确认下面路径已经存在：

```text
/home/user/ros2_ws/GeoRT/assets/linkerhand_l20_right/
├── linkerhand_l20_right.urdf
└── meshes/
```

如果这些资源缺失，请参考 [README_L20_GEORT.md](/home/user/ros2_ws/GeoRT/README_L20_GEORT.md) 中“获取 L20 URDF”的步骤恢复。

---

## 4. 当前实现的输入输出

### 4.1 输入

脚本接受两种输入源：

1. 离线 `.npy`
   文件形状必须是 `(N, 21, 3)`。
   每帧是右手 21 个关键点。

2. 实时 ROS2 Manus 话题
   默认订阅：
   `/manus_glove_0`

### 4.2 内部坐标处理

脚本会把原始人手关键点转换成 GeoRT 同款 canonical 坐标：

- 原点：腕部 `wrist`
- `+Z`：腕 -> 中指 MCP 方向
- `+Y`：掌面横向
- `+X`：掌法线方向

这样做的目的，是让实时数据和你之前 GeoRT 录制/训练使用的几何定义保持一致。

### 4.3 输出

实时模式发布：

```text
/l20_dex_retarget/joint_states
```

消息类型：

```text
sensor_msgs/msg/JointState
```

默认输出关节顺序：

```text
thumb_joint0
thumb_joint1
thumb_joint2
thumb_joint3
thumb_joint4
index_joint0
index_joint1
index_joint2
index_joint3
middle_joint0
middle_joint1
middle_joint2
middle_joint3
ring_joint0
ring_joint1
ring_joint2
ring_joint3
little_joint0
little_joint1
little_joint2
little_joint3
```

这个顺序和 `README_L20_GEORT.md` / `geort/config/linkerhand_l20_right.json` 保持一致，方便直接替换旧链路。

---

## 5. L20 mimic joints 说明

L20 URDF 中有以下 mimic joints：

- `thumb_joint4 -> thumb_joint3`
- `index_joint3 -> index_joint2`
- `middle_joint3 -> middle_joint2`
- `ring_joint3 -> ring_joint2`
- `little_joint3 -> little_joint2`

这意味着：

1. 优化器实际只求解 16 个独立关节。
2. 输出仍然会补全为 21 维关节角。
3. 你下游如果只认 21 个关节，不需要额外处理 mimic 逻辑。

---

## 6. 启动前检查

建议每次运行前按这个顺序检查。

### 6.1 检查 Manus 话题

```bash
source /opt/ros/humble/setup.bash
source /home/user/ros2_ws/install/setup.bash
ros2 topic list | grep manus
```

期望看到：

```text
/manus_glove_0
/manus_glove_0/vibration_cmd
```

再检查是否真的有数据：

```bash
ros2 topic echo /manus_glove_0 --once
```

### 6.2 检查脚本帮助

```bash
cd /home/user/ros2_ws/GeoRT
/usr/bin/python3 dex_retargeting/manus_l20_dex_retarget.py --help
```

### 6.3 检查离线数据

如果要跑离线测试，确认：

```bash
ls -lh /home/user/ros2_ws/GeoRT/data/manus_data.npy
```

---

## 7. 最小测试流程

建议先做离线，再做实时。

### 7.1 离线 smoke test

先只跑前 100 帧：

```bash
cd /home/user/ros2_ws/GeoRT
/usr/bin/python3 dex_retargeting/manus_l20_dex_retarget.py \
  --input npy \
  --npy-path data/manus_data.npy \
  --max-frames 100 \
  --joint-order geort
```

期望现象：

1. 命令正常结束。
2. 控制台打印 `valid_frames`、`avg_solve_ms`、`qpos range`。
3. 没有 `No valid frames were retargeted` 之类错误。

### 7.2 离线完整回放并保存结果

```bash
cd /home/user/ros2_ws/GeoRT
/usr/bin/python3 dex_retargeting/manus_l20_dex_retarget.py \
  --input npy \
  --npy-path data/manus_data.npy \
  --output-path analysis/manus_l20_dex_retargeting.npz \
  --joint-order geort \
  --print-every 200
```

输出文件：

[manus_l20_dex_retargeting.npz](/home/user/ros2_ws/GeoRT/analysis/manus_l20_dex_retargeting.npz)

### 7.3 检查离线结果内容

```bash
/usr/bin/python3 - <<'PY'
import numpy as np
p = "/home/user/ros2_ws/GeoRT/analysis/manus_l20_dex_retargeting.npz"
d = np.load(p, allow_pickle=True)
print(d.files)
print(d["qpos"].shape)
print(d["joint_names"])
print(d["scaling"])
PY
```

期望：

1. `qpos` 形状是 `(N, 21)`
2. `joint_names` 是 L20 关节名
3. `scaling` 有值

---

## 8. 实时启动流程

### 8.1 终端 1：启动 Manus 发布节点

如果你的 Manus 节点还没运行：

```bash
source /opt/ros/humble/setup.bash
source /home/user/ros2_ws/install/setup.bash
ros2 run manus_ros2 manus_data_publisher
```

### 8.2 终端 2：启动 dex-retargeting 实时节点

```bash
source /opt/ros/humble/setup.bash
source /home/user/ros2_ws/install/setup.bash
cd /home/user/ros2_ws/GeoRT

/usr/bin/python3 dex_retargeting/manus_l20_dex_retarget.py \
  --input ros \
  --glove-id 0 \
  --publish-topic /l20_dex_retarget/joint_states \
  --joint-order geort \
  --print-every 120
```

### 8.3 终端 3：查看输出

```bash
source /opt/ros/humble/setup.bash
source /home/user/ros2_ws/install/setup.bash
ros2 topic echo /l20_dex_retarget/joint_states --once
```

也可以看频率：

```bash
ros2 topic hz /l20_dex_retarget/joint_states
```

---

## 9. 如何把输出接到下游控制链路

这个脚本当前只负责发布 `JointState`，不直接驱动真实机械手。

下游接法通常有两种：

### 9.1 仿真/可视化侧

订阅：

```text
/l20_dex_retarget/joint_states
```

然后按 `msg.name` 对应的关节名，把 `msg.position` 映射到你的仿真器或可视化器。

### 9.2 真机控制侧

如果你的控制节点已经能吃 `sensor_msgs/JointState`，直接复用即可。

如果你的控制节点要求自定义消息，就做一个桥接节点：

1. 订阅 `/l20_dex_retarget/joint_states`
2. 按 `name -> position` 建映射
3. 填充你的驱动消息
4. 发布到机械手驱动接口

建议不要假设位置数组顺序永远固定，优先按关节名做映射。

---

## 10. 关键参数怎么调

### 10.1 `--joint-order`

可选：

- `geort`
- `pin`

建议默认使用：

```bash
--joint-order geort
```

原因：

1. 跟你现有 L20 GeoRT 配置一致。
2. 方便无缝替换旧方案。
3. 下游更容易复用之前的关节顺序假设。

### 10.2 `--scaling`

默认配置里是：

```text
1.8
```

这个值反映“人手尺寸”和“L20 机械手尺寸”的比例。

什么时候需要调：

1. 你发现手指总是张不开或过度蜷缩。
2. 你更换了手套佩戴方式或采集对象。
3. 你接入了新的 L20 URDF 版本。

常用试法：

```bash
--scaling 1.6
--scaling 1.8
--scaling 2.0
```

### 10.3 `--auto-scale`

如果你不确定缩放值，可以先用：

```bash
--auto-scale
```

脚本会用第一帧有效数据估计缩放比例。

建议：

- 初次调试时可以开
- 稳定运行后，如果你已经知道一个好用的固定值，优先改成显式 `--scaling`

### 10.4 `--print-every`

控制日志打印频率。

例如：

```bash
--print-every 1
```

每帧都打印。

```bash
--print-every 120
```

每 120 帧打印一次，更适合长时间运行。

---

## 11. 输出数据格式

### 11.1 离线 `.npz`

保存的字段包括：

- `qpos`
  形状 `(N, 21)`，L20 关节轨迹

- `joint_names`
  输出关节名顺序

- `source_joint_names`
  `dex-retargeting / pinocchio` 内部关节顺序

- `scaling`
  当前使用的缩放系数

- `input_path`
  输入数据文件路径

- `config_path`
  配置文件路径

### 11.2 实时 `JointState`

消息字段：

- `header.stamp`
- `header.frame_id`
- `name`
- `position`

默认 `frame_id`：

```text
base_link
```

如果你下游有特殊要求，可以改：

```bash
--frame-id your_frame
```

---

## 12. 当前已经验证过的内容

在当前工作区里，已经完成过这些验证：

### 12.1 离线验证

使用：

```text
/home/user/ros2_ws/GeoRT/data/manus_data.npy
```

对前 1000 帧做了回放，结果是：

1. 成功输出 `(1000, 21)` 的 `qpos`
2. 所有关节角都是有限值
3. 平均求解耗时约 `1.694 ms/frame`
4. 输出范围约 `[-0.2980, 1.4010]`
5. 结果已保存到 `analysis/manus_l20_dex_retargeting.npz`

### 12.2 实时启动验证

已经验证：

1. 脚本可在系统 Python 3.10 下启动
2. `rclpy`、`manus_ros2_msgs`、`sensor_msgs`、`nlopt`、`pinocchio` 导入正常
3. 节点可正常初始化并准备订阅 `/manus_glove_0`

说明：

在当前受限沙箱里，DDS/UDP socket 权限受限，所以无法在这里完整跑通真实 ROS2 通信；这不是脚本逻辑错误，而是当前执行环境限制。你在正常宿主机终端运行上述命令即可做完整实时验证。

---

## 13. 常用命令汇总

### 13.1 查看帮助

```bash
cd /home/user/ros2_ws/GeoRT
/usr/bin/python3 dex_retargeting/manus_l20_dex_retarget.py --help
```

### 13.2 离线快速测试

```bash
cd /home/user/ros2_ws/GeoRT
/usr/bin/python3 dex_retargeting/manus_l20_dex_retarget.py \
  --input npy \
  --npy-path data/manus_data.npy \
  --max-frames 100
```

### 13.3 离线完整导出

```bash
cd /home/user/ros2_ws/GeoRT
/usr/bin/python3 dex_retargeting/manus_l20_dex_retarget.py \
  --input npy \
  --npy-path data/manus_data.npy \
  --output-path analysis/manus_l20_dex_retargeting.npz \
  --joint-order geort \
  --scaling 1.8
```

### 13.4 实时运行

```bash
source /opt/ros/humble/setup.bash
source /home/user/ros2_ws/install/setup.bash
cd /home/user/ros2_ws/GeoRT

/usr/bin/python3 dex_retargeting/manus_l20_dex_retarget.py \
  --input ros \
  --glove-id 0 \
  --publish-topic /l20_dex_retarget/joint_states \
  --joint-order geort \
  --scaling 1.8
```

### 13.5 实时运行并自动估计缩放

```bash
source /opt/ros/humble/setup.bash
source /home/user/ros2_ws/install/setup.bash
cd /home/user/ros2_ws/GeoRT

/usr/bin/python3 dex_retargeting/manus_l20_dex_retarget.py \
  --input ros \
  --glove-id 0 \
  --publish-topic /l20_dex_retarget/joint_states \
  --joint-order geort \
  --auto-scale
```

### 13.6 观察输出

```bash
source /opt/ros/humble/setup.bash
source /home/user/ros2_ws/install/setup.bash
ros2 topic echo /l20_dex_retarget/joint_states --once
```

---

## 14. 故障排查

### 14.1 报错：`No valid frames were retargeted`

原因通常是：

1. 输入 `.npy` 形状不对，不是 `(N, 21, 3)`
2. 关键点里有 NaN / Inf
3. 手部关键点退化，腕和手指几乎重合

检查方式：

```bash
/usr/bin/python3 - <<'PY'
import numpy as np
x = np.load('/home/user/ros2_ws/GeoRT/data/manus_data.npy')
print(x.shape, np.isfinite(x).all())
PY
```

### 14.2 报错：`rclpy` 导入失败

基本都是因为用错了解释器。

正确做法：

```bash
/usr/bin/python3 ...
```

不是：

```bash
python ...
```

### 14.3 报错：找不到 `/manus_glove_0`

先确认 Manus 节点是否运行：

```bash
ros2 topic list | grep manus
```

如果没有，就先启动：

```bash
ros2 run manus_ros2 manus_data_publisher
```

### 14.4 能启动，但 `/l20_dex_retarget/joint_states` 没数据

通常按这个顺序检查：

1. `/manus_glove_0` 是否真的在发布
2. `ros2 topic echo /manus_glove_0 --once` 是否能看到 `raw_nodes`
3. 手套是否断连或姿态数据退化
4. 是否因为 `--glove-id` 选错了，实际发布的话题可能是 `/manus_glove_1`

### 14.5 关节动作太小或太大

优先调整：

```bash
--scaling 1.6
--scaling 1.8
--scaling 2.0
```

如果你不确定，从：

```bash
--auto-scale
```

开始。

### 14.6 输出顺序和下游对不上

优先检查你下游到底是按“数组顺序”还是按“关节名”读数据。

如果下游历史上沿用了 GeoRT 的 L20 顺序，请确保使用：

```bash
--joint-order geort
```

---

## 15. 推荐使用顺序

实际使用时，建议按下面流程执行：

1. 先确认 L20 URDF 资源存在
2. 再确认 `/manus_glove_0` 有数据
3. 先做离线 `--input npy` 测试
4. 确认 `analysis/manus_l20_dex_retargeting.npz` 输出正常
5. 再启动实时 `--input ros`
6. 最后把 `/l20_dex_retarget/joint_states` 对接到你的仿真或真机控制节点

---

## 16. 一条最推荐的启动命令

如果你现在要直接跑实时版，优先用这条：

```bash
source /opt/ros/humble/setup.bash
source /home/user/ros2_ws/install/setup.bash
cd /home/user/ros2_ws/GeoRT

/usr/bin/python3 dex_retargeting/manus_l20_dex_retarget.py \
  --input ros \
  --glove-id 0 \
  --publish-topic /l20_dex_retarget/joint_states \
  --joint-order geort \
  --scaling 1.8 \
  --print-every 120
```

如果你想先保守验证，优先用这条离线命令：

```bash
cd /home/user/ros2_ws/GeoRT
/usr/bin/python3 dex_retargeting/manus_l20_dex_retarget.py \
  --input npy \
  --npy-path data/manus_data.npy \
  --output-path analysis/manus_l20_dex_retargeting.npz \
  --joint-order geort \
  --scaling 1.8
```
