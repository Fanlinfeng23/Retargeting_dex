# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

def get_finger_fk(n_joint=4, n_points=1, hidden=128):
    return nn.Sequential(
        nn.Linear(n_joint, hidden), 
        nn.LeakyReLU(), 
        nn.BatchNorm1d(hidden),
        nn.Linear(hidden, hidden), 
        nn.LeakyReLU(), 
        nn.BatchNorm1d(hidden),
        nn.Linear(hidden, 3 * n_points)
    ) 

def get_finger_ik(n_joint=4, n_points=1, hidden=128):
    return nn.Sequential(
        nn.Linear(3 * n_points, hidden), 
        nn.LeakyReLU(), 
        nn.BatchNorm1d(hidden),
        nn.Linear(hidden, hidden), 
        nn.LeakyReLU(), 
        nn.BatchNorm1d(hidden),
        nn.Linear(hidden, n_joint),
        nn.Tanh()   # Normalize.
    ) 

class FKModel(nn.Module):
    def __init__(self, keypoint_joints, keypoint_group_sizes=None):
        # keypoint_joints: a list of list.
        # keypoint_group_sizes: number of observed points used for each finger group.

        super().__init__()
        num_fingers = len(keypoint_joints)
        if keypoint_group_sizes is None:
            keypoint_group_sizes = [1 for _ in range(num_fingers)]
        
        self.nets = []
        self.n_total_joint = 0

        for joint, n_points in zip(keypoint_joints, keypoint_group_sizes):
            net = get_finger_fk(n_joint=len(joint), n_points=n_points)
            self.nets.append(net)
            self.n_total_joint += len(joint)

        self.nets = nn.ModuleList(self.nets)

        self.keypoint_joints = keypoint_joints
        self.keypoint_group_sizes = keypoint_group_sizes

    def forward(self, joint):
        # x: [B, DOF], joint values. normalized to [-1, 1]. 
        # out:   [B, N, 3], sequence of keypoint.
        keypoints = []
        for i, net in enumerate(self.nets):
            joint_ids = self.keypoint_joints[i]
            n_points = self.keypoint_group_sizes[i]
            keypoint = net(joint[:, joint_ids]).reshape(joint.size(0), n_points, 3)
            keypoints.append(keypoint)

        return torch.cat(keypoints, dim=1)

    
class IKModel(nn.Module):
    def __init__(self, keypoint_joints, keypoint_group_sizes=None):
        # keypoint_joints: a list of list.
        # keypoint_group_sizes: number of observed points used for each finger group.

        super().__init__()
        self.n_total_joint = 0
        self.nets = []
        if keypoint_group_sizes is None:
            keypoint_group_sizes = [1 for _ in range(len(keypoint_joints))]

        for joint, n_points in zip(keypoint_joints, keypoint_group_sizes):
            net = get_finger_ik(n_joint=len(joint), n_points=n_points)
            self.nets.append(net)
            self.n_total_joint += len(joint)

        self.nets = nn.ModuleList(self.nets)
        self.keypoint_joints = keypoint_joints
        self.keypoint_group_sizes = keypoint_group_sizes

    def forward(self, x):
        # x:   [B, N, 3], sequence of keypoint.
        # out: [B, DOF], joint values. normalized to [-1, 1]. 
        batch_size = x.size(0)
        out = torch.zeros((batch_size, self.n_total_joint)).to(x.device)
        start_idx = 0
        for i, net in enumerate(self.nets):
            n_points = self.keypoint_group_sizes[i]
            group_input = x[:, start_idx:start_idx + n_points].reshape(batch_size, n_points * 3)
            joint = net(group_input)
            out[:, self.keypoint_joints[i]] = joint
            start_idx += n_points
        return out 
