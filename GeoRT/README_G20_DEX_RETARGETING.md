# Manus -> 灵心巧手 G20：`dex-retargeting` 最终使用说明

本文档说明如何在当前工作区中，把 Manus 右手关键点通过 `dex-retargeting` 重定向为灵心巧手 `G20` 右手的 20 维控制参数。

如果你是从 GitHub 克隆本仓库开始使用，下面所有命令都以仓库根目录记为：

```text
<REPO_ROOT>
```

其中 G20 相关代码位于：

```text
<REPO_ROOT>/GeoRT
```

当前实现路径为：

```text
Manus 右手 -> MediaPipe顺序21x3关键点 -> dex-retargeting官方MANO/right-hand坐标
-> G20 URDF坐标对齐 -> dex-retargeting官方IK -> G20 右手20维range参数
```

## 1. 当前实现结论

截至 2026-04-29，当前 G20 版本已经满足这四点：

1. 不再把 Manus 右手映射到 G20 左手。
2. `dex-retargeting` 的求解模型已经切换到 G20 右手专属 URDF。
3. 实际 IK 仍然使用 `dex-retargeting` 官方优化求解器，而不是自写近似 IK。
4. 删除了旧的伪 `*_tip` link/vector配置；G20 retargeting只引用当前G20 URDF真实存在的link。

需要同时说明的一点是：

- 官方 `linkerhand-urdf` 仓库公开了 `lhg20/left/linkerhand_lhg20_left.urdf`
- 截至 2026-04-28，没有公开 `lhg20/right/linkerhand_lhg20_right.urdf`

因此当前右手 URDF 的来源是：

1. 以官方 `lhg20/left` 为基准模型
2. 按右手镜像规则生成本地 `G20 right` 运动学 URDF
3. 再用这个 `G20 right` URDF 做 `dex-retargeting`

这和“把右手数据映射到左手机器手”是两回事。现在的链路是“右手数据 -> 右手机器手模型”。

## 2. 相关文件

```text
<REPO_ROOT>/GeoRT/
├── README_G20_DEX_RETARGETING.md
├── assets/
│   └── linkerhand_g20_right/
│       └── linkerhand_g20_right.urdf
└── dex_retargeting/
    ├── linkerhand_g20_right_vector.yml
    └── manus_g20_dex_retarget.py
```

- [manus_g20_dex_retarget.py](dex_retargeting/manus_g20_dex_retarget.py)
  G20 专属运行脚本。
- [linkerhand_g20_right_vector.yml](dex_retargeting/linkerhand_g20_right_vector.yml)
  G20 右手 `dex-retargeting` 配置。
- [linkerhand_g20_right.urdf](assets/linkerhand_g20_right/linkerhand_g20_right.urdf)
  基于官方 `lhg20/left` 镜像得到的本地 G20 右手运动学 URDF。

## 3. 官方 IK 接口

当前 G20 方案使用的是 `dex-retargeting` 官方 `vector` 接口。

脚本内部实际调用链是：

1. Manus `raw_nodes` 按语义重排为 MediaPipe 21点顺序。
2. 执行官方 `SingleHandDetector` 同款流程：先减去 wrist，再估计 wrist frame，再乘 `OPERATOR2MANO_RIGHT`。
3. 执行固定的 G20 URDF坐标对齐：`--frame-alignment g20`，默认开启。
4. 通过 `RetargetingConfig.load_from_file(...).build()` 构造官方优化器。
5. 对 `target_link_human_indices` 计算 `joint_pos[task] - joint_pos[origin]`。
6. 调用 `retargeting.retarget(ref_value)`。

也就是说：

- IK 优化器来自 `dex-retargeting` 官方实现
- URDF 是 G20 右手专属模型
- 人手约束是 Manus 21 点 canonical 几何向量
- G20 坐标对齐只是把官方MANO/right-hand点表达到LinkerHand G20 URDF坐标系，不替换优化器

不是训练模型，也不是手写逆运动学闭式解。

## 3.1 G20 坐标对齐

`dex-retargeting` 官方示例默认机器人URDF已经和MANO/right-hand坐标对齐；LinkerHand G20官方URDF的四指屈曲方向在产品URDF坐标中是 `+X`，而官方MANO/right-hand输入中屈曲方向表现为 `-X`。因此当前脚本默认执行：

```text
G20_DEX_TO_URDF_FRAME = diag([-1, 1, 1])
```

可以用 `--frame-alignment none` 关闭，但实时控制G20时应保持默认 `g20`。

## 4. G20 参数语义

`G20` 官方 SDK 公开的控制语义是 20 维：

1. `thumb_base`
2. `index_base`
3. `middle_base`
4. `ring_base`
5. `little_base`
6. `thumb_abduction`
7. `index_abduction`
8. `middle_abduction`
9. `ring_abduction`
10. `little_abduction`
11. `thumb_roll`
12. `reserved_11`
13. `reserved_12`
14. `reserved_13`
15. `reserved_14`
16. `thumb_tip`
17. `index_tip`
18. `middle_tip`
19. `ring_tip`
20. `little_tip`

当前 G20 右手 URDF 中参与求解的独立自由度是 16 个：

- 四指：`mcp_roll + mcp_pitch + pip`
- 拇指：`cmc_pitch + cmc_roll + cmc_yaw + mcp`

映射关系如下：

```text
thumb_base        <- thumb_cmc_pitch
index_base        <- index_mcp_pitch
middle_base       <- middle_mcp_pithch
ring_base         <- ring_mcp_pitch
little_base       <- pinky_mcp_pitch
thumb_abduction   <- thumb_cmc_roll
index_abduction   <- index_mcp_roll
middle_abduction  <- middle_mcp_roll
ring_abduction    <- ring_mcp_roll
little_abduction  <- pinky_mcp_roll
thumb_roll        <- thumb_cmc_yaw
reserved_11~14    <- 固定值
thumb_tip         <- thumb_mcp
index_tip         <- index_pip
middle_tip        <- middle_pip
ring_tip          <- ring_pip
little_tip        <- pinky_pip
```

## 5. 输出模式

### 5.1 `raw`

直接发布到 SDK 控制话题：

```text
/cb_right_hand_control_cmd
```

消息类型：

```text
sensor_msgs/msg/JointState
```

`position` 长度为 20，范围为 `[0, 255]`。

### 5.2 `arc`

发布到分析用弧度话题：

```text
/g20_dex_retarget/joint_states_arc
```

消息类型同样是：

```text
sensor_msgs/msg/JointState
```

`position` 为 20 维弧度值。

注意：
官方 ROS2 SDK 当前公开代码里，G20 真机控制默认订阅的是 `/cb_right_hand_control_cmd` 范围值话题。因此实时对真机，优先使用 `raw`。

## 6. 环境要求

```bash
source /opt/ros/humble/setup.bash
source /home/user/ros2_ws/install/setup.bash
```

Python 解释器请使用系统 Python：

```bash
/usr/bin/python3 --version
```

不要用 conda 里的 Python。

## 7. 离线测试

### 7.1 最小测试

```bash
cd <REPO_ROOT>/GeoRT
/usr/bin/python3 dex_retargeting/manus_g20_dex_retarget.py \
  --input npy \
  --npy-path data/manus_data.npy \
  --frame-alignment g20 \
  --max-frames 100 \
  --auto-scale \
  --print-every 20
```

### 7.2 全量离线验证

```bash
cd <REPO_ROOT>/GeoRT
/usr/bin/python3 dex_retargeting/manus_g20_dex_retarget.py \
  --input npy \
  --npy-path data/manus_data.npy \
  --frame-alignment g20 \
  --auto-scale \
  --output-path analysis/manus_g20_dex_retargeting.npz \
  --print-every 120
```

输出文件：

- [analysis/manus_g20_dex_retargeting.npz](analysis/manus_g20_dex_retargeting.npz)

文件中包含：

- `qpos_g20`
  G20 右手 URDF 求解关节角
- `qpos_joint_names`
  G20 URDF 内部关节顺序
- `g20_arc`
  20 维弧度参数
- `g20_raw`
  20 维 `[0, 255]` 参数
- `g20_command_names`
  G20 参数名

## 8. 实时 ROS2 使用

### 8.1 发布 G20 原始参数

```bash
cd <REPO_ROOT>/GeoRT
source /opt/ros/humble/setup.bash
source /home/user/ros2_ws/install/setup.bash
/usr/bin/python3 dex_retargeting/manus_g20_dex_retarget.py \
  --input ros \
  --output-mode raw \
  --publish-topic /cb_right_hand_control_cmd \
  --frame-alignment g20 \
  --auto-scale \
  --print-every 120
```

检查是否有数据：

```bash
ros2 topic echo /cb_right_hand_control_cmd --once
```

### 8.2 发布 G20 弧度参数

```bash
cd <REPO_ROOT>/GeoRT
source /opt/ros/humble/setup.bash
source /home/user/ros2_ws/install/setup.bash
/usr/bin/python3 dex_retargeting/manus_g20_dex_retarget.py \
  --input ros \
  --output-mode arc \
  --publish-topic /g20_dex_retarget/joint_states_arc \
  --frame-alignment g20 \
  --auto-scale \
  --print-every 120
```

检查：

```bash
ros2 topic echo /g20_dex_retarget/joint_states_arc --once
```

## 9. 验证结果

截至 2026-04-29，当前离线验证结果为：

1. `GeoRT/data/manus_data.npy` 全量 `3591` 帧均可跑通
2. `g20_arc.shape == (3591, 20)`
3. `g20_raw.shape == (3591, 20)`
4. `g20_raw` 范围为 `[0, 255]`
5. ROS2 实时节点可正常启动，并默认发布到 `/cb_right_hand_control_cmd`

这说明当前链路已经满足：

- 使用 G20 右手模型
- 使用 `dex-retargeting` 官方 IK
- 可以离线和实时稳定运行

## 10. 关于“准确性”的工程结论

当前实现相对于之前版本，准确性提升点是明确的：

1. 不再使用 L20 几何
2. 不再把右手数据喂给左手机器手
3. 改为 G20 右手专属 URDF
4. 关节限位改为 G20 结构范围
5. 20 维输出语义与 G20 SDK 保持一致

但是仍然要实事求是地说明边界：

1. 官方截至 2026-04-28 未公开 `lhg20/right` 原始 URDF，只公开了 `lhg20/left`
2. 当前右手 URDF 是基于官方左手模型按右手镜像规则生成的
3. 因此“足够准确”可以对离线 retargeting 和 SDK 参数语义负责
4. 真机绝对精度仍建议用几组标准手势做联调确认，尤其是环指、小指和拇指侧摆

如果你后面提供 G20 真机对照视频、标定姿态或官方右手 URDF，我可以继续把这一步再压实。
