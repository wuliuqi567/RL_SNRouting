from argparse import Namespace
from copy import deepcopy
from torch.nn import Module
import torch
import numpy as np
import os, re

from Algorithm.common.experienceReplay import ExperienceReplay
from Utils.utilsfunction import *
from Utils.statefunction import *
from .base_agent import BaseAgent

from Algorithm.learner.ddqn_learner import DDQN_learner
from dgl import DGLGraph
import dgl, random
import argparse
from .base_agent import get_configs, save_configs
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class DDQNAgent(BaseAgent):
    model_name = 'ddqn_model.pth'

    def __init__(self, config: Namespace | None = None):
        configs_dict = get_configs("../algo_config/ddqn.yaml")
        config = argparse.Namespace(**configs_dict)
        # save_configs(configs_dict, os.path.join(config.outputPath, "configs_used.yaml"))
        super(DDQNAgent, self).__init__(config)
        self.n_order_adj = config.n_order_adj
        self.actionSize = config.action_size
        self.actions = ('U', 'D', 'R', 'L')
        self.train_TA_model = config.train_TA_model

        # 构建策略网络
        self.policy = self._build_policy()

        # 构建学习器
        self.learner = self._build_learner()

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


    def _build_policy(self) -> Module:
        return DDQNNetwork(self.config)

    def _build_learner(self, *args):
        return DDQN_learner(self.config, self.policy)

class DDQNNetwork(Module):
    def __init__(self, config: Namespace):
        super(DDQNNetwork, self).__init__()
        # Initialize the MHGNN network components here based on config
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # self.n_order_adj = config.n_order_adj
        self.tau = config.tau

        self.qNetwork = DDQNModel(config.input_dim, config.hidden_feats, config.mlp_hidden_feats, config.action_size, config.n_order_adj).to(self.device)

        self.qTarget = deepcopy(self.qNetwork).to(self.device)
        self.sNetwork = deepcopy(self.qNetwork).to(self.device)

    def hard_update_target(self):
        """硬更新：直接复制Q网络权重到目标网络"""
        self.qTarget.load_state_dict(self.qNetwork.state_dict())

    def update_student(self):
        """将教师网络的权重复制到学生网络"""
        self.sNetwork.load_state_dict(self.qNetwork.state_dict())


class DDQNModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, mlp_hidden_feats, out_classes, n_order_adj):
        super(DDQNModel, self).__init__()
        
        # 1. 用于特征提取
        self.linear = nn.Linear(in_feats, 8)  # 假设我们使用5个节点的特征拼接作为输入
        # 2. 计算 MLP 的输入维度
        self.num_nodes = 2 * n_order_adj * (n_order_adj + 1) + 1  # 计算子图中的节点数
        mlp_in_dim = hidden_feats * self.num_nodes  # 假设我们使用5个节点的特征拼接作为输入
        
        # 3. MLP 层：用于决策
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, mlp_hidden_feats),
            nn.ReLU(),
            nn.Dropout(0.5), # 防止过拟合
            nn.Linear(mlp_hidden_feats, out_classes)
        )

    def forward(self, h, g=None):
        """Forward pass.

        The rest of this repo follows the pattern:
          model.g = dgl_graph
          q_values = model(node_features)

        So we support both (h) and (h, g).
        """
        if g is None:
            g = getattr(self, 'g', None)
        if g is None:
            raise TypeError("DDQNModel.forward requires a DGLGraph via argument 'g' or by setting self.g")

        # --- 第一步：特征提取 ---
        # h 的形状: (N, in_feats)
        # linear_out 的形状: (N, hidden_feats)
        linear_out = self.linear(h)

        # --- 第二步：处理多头维度 (关键步骤) ---
        total_nodes = g.num_nodes()
        batch_size = getattr(g, 'batch_size', None)
        if batch_size is None:
            if total_nodes % self.num_nodes != 0:
                # Fallback or error if graph size doesn't match expected node count
                batch_size = total_nodes // self.num_nodes
            else:
                batch_size = total_nodes // self.num_nodes
        
        feature_embedding = linear_out.reshape(batch_size, -1)
        
        # 可以在这里加一个激活函数，比如 ELU (GAT 论文常用)
        feature_embedding = F.elu(feature_embedding)

        # --- 第三步：MLP 决策 ---
        # logits 的形状: (N, out_classes)
        logits = self.mlp(feature_embedding)
        
        return logits