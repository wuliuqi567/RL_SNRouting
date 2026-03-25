
from argparse import Namespace
from copy import deepcopy
from torch.nn import Module
import torch
import torch.nn as nn
import numpy as np
import os
import re

from Utils.utilsfunction import *
from Utils.statefunction import *
from .base_agent import BaseAgent
from Algorithm.GNNmodel.MAGNA_model import MAGNA
from Algorithm.learner.mhgnnddqn_learner import MHGNNDDQNLearner
from dgl import DGLGraph
import dgl, random
import argparse
from .base_agent import get_configs, save_configs
import system_configure



class MHGNNAgent(BaseAgent):
    model_name = 'mhgnn_model.pth'

    def __init__(self):
        configs_dict = get_configs("../algo_config/gnn_pd.yaml")
        config = argparse.Namespace(**configs_dict)
        
        super(MHGNNAgent, self).__init__(config)
        # Additional initialization for MHGNNAgent can be added here
        self.n_order_adj = config.n_order_adj
        self.actionSize = config.action_size
        self.actions = ('U', 'D', 'R', 'L')

        self.policy = self._build_policy()  # build policy
        self.learner = self._build_learner()  # build learner
        # self.memory = self._build_memory()  # build memory
        self.train_TA_model = config.train_TA_model
        self.use_student_network = config.use_student_network
        if not self.train_TA_model:
            self.load_model()
        else:
            save_configs(configs_dict, self.model_dir_save)

        self.nTrain = config.nTrain
        self.train_epoch = hasattr(config, 'train_epoch') and config.train_epoch or 1
        self.step = 0
        self.updateF_count = 0
        self.updateF = config.updateF
        self.epsilon = []

        self.w1 = hasattr(config, 'w1') and config.w1 or 50        # rewards the getting to empty queues
        self.w2 = hasattr(config, 'w2') and config.w2 or 20        # rewards getting closes phisycally
        self.w4 = hasattr(config, 'w4') and config.w4 or 5         # Normalization for the distance reward, for the traveled distance factor

    def _build_policy(self) -> Module:
        
        return MHGNNNetwork(config=self.config)

    def _build_learner(self, *args):
        return MHGNNDDQNLearner(self.config, self.policy)

    def _get_q_values_for_action(self, new_state):
        if self.use_student_network:
            self.policy.sNetwork.eval()
            self.policy.sNetwork.g = new_state
            return self.policy.sNetwork(new_state.ndata['1st_order_feat'])

        self.policy.qNetwork.eval()
        self.policy.qNetwork.g = new_state
        return self.policy.qNetwork(new_state.ndata['feat'])

    def _load_snetwork_state(self, sNet_model_path):
        try:
            self.policy.sNetwork.load_state_dict(torch.load(sNet_model_path, map_location=self.device, weights_only=True))
        except RuntimeError as e:
            print(f"Warning: skip loading incompatible sNetwork checkpoint: {e}")

    def _calculate_reward_v2(self, block, sat, prevSat, is_terminal=False, is_failure=False):
        w = 5        # rewards the getting to empty queues
        ArriveRewardC1 = 5
        PenaltyC2 = -5
        
        satDest = block.destination.linkedSat[1]
        reward = getLyapunovReward(prevSat, sat, satDest, w)

        if is_failure:
            return reward - PenaltyC2
        if is_terminal:
            return  reward + ArriveRewardC1
        return reward



class StudentMLP(Module):
    """轻量学生网络：拼接中心节点特征 + 一阶邻居均值池化特征后送入多层 MLP。

    实际 MLP 输入维度 = input_dim * 2（center_feat ∥ mean_pool(1st_order_feat)）。
    """

    def __init__(self, input_dim: int, action_size: int, hidden_dims=None, dropout: float = 0.0):
        super(StudentMLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        # 中心节点(input_dim) + 一阶邻居均值池化(input_dim)
        mlp_input_dim = input_dim * 2

        dims = [mlp_input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], action_size))
        self.model = nn.Sequential(*layers)

    def forward(self, node_features: torch.Tensor):
        """
        前向传播：从全图节点特征中提取中心节点 + 一阶邻居信息，拼接后送入 MLP。

        【关于输入 node_features 的说明】
        调用方传入的特征有两种形式，但形状完全相同 (N_total, D=input_dim)：
          - 推理阶段：newState.ndata['1st_order_feat']
          - 蒸馏训练阶段：pd_graph.ndata['pdfeat']
        两者都是由 feat 乘以 (is_center | is_first_order) 掩码得到的，即：
          node_features[i] = feat[i]  如果节点 i 是中心节点或一阶邻居
          node_features[i] = 0        否则（二阶及更远的节点特征已被置零）
        因此 node_features 包含 **所有** N_total 个节点，但只有中心 + 一阶邻居行有有效值。

        本 forward 通过布尔掩码显式提取有效节点，不会使用那些零向量行，
        所以输入中"其他节点被置零"这一事实不影响结果正确性。

        Args:
            node_features: (N_total, D) 全图节点特征张量。
                           - 推理时 N_total = 单张子图节点数（如 n_order=4 时为 41）
                           - 蒸馏时 N_total = batch_size 张子图节点数之和
                           - D = input_dim = 10

        Returns:
            (B, action_size) 每个子图的 Q 值向量。
                           - 推理时 B=1, action_size=4
                           - 蒸馏时 B=batch_size, action_size=4
        """
        # ── 0. 前置检查：确保调用方已设置 self.g（DGL 图上下文） ──
        if not hasattr(self, 'g') or self.g is None:
            raise AttributeError("StudentMLP requires graph context via 'self.g' before forward.")

        g = self.g
        device = node_features.device

        # ── 1. 获取节点角色掩码 ──
        # is_center:      (N_total,) bool，每个子图恰好有 1 个 True（决策卫星）
        # is_first_order: (N_total,) bool，每个子图有 ≤4 个 True（上下左右邻居）
        center_mask = g.ndata['is_center']            # (N_total,)
        first_order_mask = g.ndata['is_first_order']  # (N_total,)

        # ── 2. 提取中心节点特征 ──
        # 用布尔索引从 node_features 中取出所有 is_center==True 的行。
        # 推理时得到 (1, D)，蒸馏训练时得到 (batch_size, D)。
        # 这些特征在 1st_order_feat/pdfeat 中是非零的（中心节点属于掩码范围）。
        center_features = node_features[center_mask]  # (B, D)
        B, D = center_features.shape                  # B = 子图数量, D = input_dim = 10
        if B == 0:
            raise RuntimeError("No center node found in graph for StudentMLP forward.")

        # ── 3. 构建 "节点→所属子图" 映射 ──
        # DGL 的 batched graph 将多张子图的节点拼成一维：
        #   子图0: node 0..n0-1, 子图1: node n0..n0+n1-1, ...
        # batch_num_nodes() 返回每张子图的节点数，如 [41, 41, ..., 41]（共 B 个）
        # repeat_interleave 将子图编号按节点数展开，得到每个节点对应的子图 ID：
        #   [0,0,...0, 1,1,...1, ..., B-1,B-1,...B-1]   形状 (N_total,)
        batch_num_nodes = g.batch_num_nodes()          # (B,)  e.g. [41, 41, ...]
        node_graph_id = torch.arange(len(batch_num_nodes), device=device) \
                             .repeat_interleave(batch_num_nodes)  # (N_total,)

        # ── 4. 提取一阶邻居特征及其所属子图 ID ──
        # fo_features:  所有子图中 is_first_order==True 的节点特征
        #               形状 (num_fo_total, D)，num_fo_total ≈ B*4（每个子图最多 4 个邻居）
        #               这些特征在 1st_order_feat/pdfeat 中同样是非零的。
        # fo_graph_ids: 每个一阶邻居节点属于哪张子图，形状 (num_fo_total,)
        fo_features = node_features[first_order_mask]  # (num_fo_total, D)
        fo_graph_ids = node_graph_id[first_order_mask]  # (num_fo_total,)

        # ── 5. 按子图进行均值池化（mean pooling） ──
        # 目标：将同一子图内的一阶邻居特征求平均，得到每个子图一个 (D,) 向量。
        #
        # fo_sum:   (B, D) 先累加每个子图的一阶邻居特征
        # fo_count: (B, 1) 统计每个子图有几个一阶邻居
        #
        # scatter_add_ 按 fo_graph_ids 把 fo_features 的各行加到对应子图行上：
        #   fo_sum[graph_id] += fo_features[i]  对每个一阶邻居 i
        # fo_count 同理累加 1.0，最后 fo_sum / fo_count 即均值。
        # clamp(min=1) 防止某子图没有一阶邻居时除零（理论上不会发生）。
        fo_sum = torch.zeros(B, D, device=device)
        fo_count = torch.zeros(B, 1, device=device)
        fo_sum.scatter_add_(0, fo_graph_ids.unsqueeze(1).expand_as(fo_features), fo_features)
        fo_count.scatter_add_(0, fo_graph_ids.unsqueeze(1),
                              torch.ones(fo_graph_ids.size(0), 1, device=device))
        fo_mean = fo_sum / fo_count.clamp(min=1)       # (B, D)

        # ── 6. 拼接并送入 MLP ──
        # combined = [center_features ∥ fo_mean]，形状 (B, 2*D) = (B, 20)
        # 前 D 维：中心卫星自身的位置 + 队列信息（8 维有效，2 维恒为 0）
        # 后 D 维：4 个一阶邻居的位置 + 队列信息的平均值
        # 经过 MLP: Linear(20→64) → ReLU → Linear(64→32) → ReLU → Linear(32→4)
        combined = torch.cat([center_features, fo_mean], dim=1)  # (B, 2*D)
        return self.model(combined)                               # (B, action_size)

class MHGNNNetwork(Module):
    def __init__(self, config: Namespace):
        super(MHGNNNetwork, self).__init__()
        # Initialize the MHGNN network components here based on config
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.n_order_adj = config.n_order_adj
        self.tau = config.tau
        self.input_dim = config.input_dim
        self.actionSize = config.action_size
        self.student_model = getattr(config, 'student_model', 'magna').lower()


        heads = [config.num_heads] * config.num_layers
        self.qNetwork = MAGNA(
            num_layers=config.num_layers,
            input_dim=config.input_dim,
            project_dim=config.project_dim,
            hidden_dim=config.num_hidden,
            action_dim=self.actionSize,
            n_order_adj=self.n_order_adj,
            heads=heads,
            feat_drop=config.in_drop,
            attn_drop=config.attn_drop,
            alpha=config.alpha,
            hop_num=config.hop_num,
            edge_drop=config.edge_drop,
            layer_norm=config.layer_norm,
            feed_forward=config.feed_forward,
            self_loop_number=1,
            self_loop=(config.self_loop == 1),
            head_tail_shared=(config.head_tail_shared == 1),
            negative_slope=config.negative_slope
        ).to(self.device)

        self.qTarget = deepcopy(self.qNetwork).to(self.device)
        if self.student_model == 'magna':
            self.sNetwork = deepcopy(self.qNetwork).to(self.device)
        elif self.student_model == 'mlp':
            student_hidden_dims = getattr(config, 'student_hidden_dims', [64, 32])
            student_dropout = getattr(config, 'student_dropout', 0.0)
            self.sNetwork = StudentMLP(
                input_dim=self.input_dim,
                action_size=self.actionSize,
                hidden_dims=student_hidden_dims,
                dropout=student_dropout
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported student_model: {self.student_model}. Use 'magna' or 'mlp'.")

    def hard_update_target(self):
        """硬更新：直接复制Q网络权重到目标网络"""
        self.qTarget.load_state_dict(self.qNetwork.state_dict())

    def update_student(self):
        """按学生网络类型同步权重。"""
        if self.student_model == 'magna':
            self.sNetwork.load_state_dict(self.qNetwork.state_dict())
        return
