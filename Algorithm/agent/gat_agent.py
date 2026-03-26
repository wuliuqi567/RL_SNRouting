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

from Algorithm.learner.gatddqn_learner import GATLearner
from dgl import DGLGraph
import dgl, random
import argparse
from .base_agent import get_configs, save_configs
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class GATAgent(BaseAgent):
    model_name = 'mhgnn_model.pth'

    def __init__(self, config: Namespace | None = None):
        configs_dict = get_configs("../algo_config/gat.yaml")
        config = argparse.Namespace(**configs_dict)
        # save_configs(configs_dict, os.path.join(config.outputPath, "configs_used.yaml"))
        super(GATAgent, self).__init__(config)
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
        return GATNetwork(self.config)

    def _build_learner(self, *args):
        return GATLearner(self.config, self.policy)

class GATNetwork(Module):
    def __init__(self, config: Namespace):
        super(GATNetwork, self).__init__()
        # Initialize the MHGNN network components here based on config
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # self.n_order_adj = config.n_order_adj
        self.tau = config.tau
        
        num_layers = getattr(config, 'num_layers', 1)
        n_order_adj = getattr(config, 'n_order_adj', 1)

        self.qNetwork = GATModel(config.input_dim, config.gat_hidden_feats, config.num_heads, num_layers, config.mlp_hidden_feats, config.action_size, n_order_adj).to(self.device)

        self.qTarget = deepcopy(self.qNetwork).to(self.device)
        self.sNetwork = deepcopy(self.qNetwork).to(self.device)

    def hard_update_target(self):
        """硬更新：直接复制Q网络权重到目标网络"""
        self.qTarget.load_state_dict(self.qNetwork.state_dict())

    def update_student(self):
        """将教师网络的权重复制到学生网络"""
        self.sNetwork.load_state_dict(self.qNetwork.state_dict())


class GATModel(nn.Module):
    def __init__(self, in_feats, gat_hidden_feats, num_heads, num_layers, mlp_hidden_feats, out_classes, n_order_adj):
        super(GATModel, self).__init__()
        
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        
        # Layer 1
        self.layers.append(GATConv(in_feats, gat_hidden_feats, num_heads=num_heads, allow_zero_in_degree=True))
        
        # Subsequent layers
        # Input dim is previous layer's hidden_feats * num_heads (because we flatten heads)
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(gat_hidden_feats * num_heads, gat_hidden_feats, num_heads=num_heads, allow_zero_in_degree=True))
        
        # Calculate num_nodes
        self.num_nodes = 2 * n_order_adj * (n_order_adj + 1) + 1
        
        # MLP input dimension
        # We concatenate features of all nodes
        # Each node has feature dim: gat_hidden_feats * num_heads
        mlp_in_dim = self.num_nodes * gat_hidden_feats * num_heads
        
        hidden_dims = mlp_hidden_feats if isinstance(mlp_hidden_feats, (list, tuple)) else [mlp_hidden_feats]
        hidden_dims = [int(dim) for dim in hidden_dims if dim is not None]

        mlp_layers = []
        prev_dim = mlp_in_dim
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5), # 防止过拟合
            ])
            prev_dim = hidden_dim
        mlp_layers.append(nn.Linear(prev_dim, out_classes))

        self.mlp = nn.Sequential(*mlp_layers)

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
            raise TypeError("GATModel.forward requires a DGLGraph via argument 'g' or by setting self.g")

        # --- GAT Layers ---
        for i, layer in enumerate(self.layers):
            h = layer(g, h) # (N, num_heads, hidden_feats)
            h = h.flatten(1) # (N, num_heads * hidden_feats)
            
            if i < self.num_layers - 1:
                h = F.elu(h)

        # --- 处理多头维度 & Flatten ---
        total_nodes = g.num_nodes()
        batch_size = getattr(g, 'batch_size', None)
        if batch_size is None:
            if total_nodes % self.num_nodes != 0:
                batch_size = total_nodes // self.num_nodes
            else:
                batch_size = total_nodes // self.num_nodes
        
        feature_embedding = h.reshape(batch_size, -1)
        
        # 可以在这里加一个激活函数，比如 ELU (GAT 论文常用)
        feature_embedding = F.elu(feature_embedding)

        # --- MLP 决策 ---
        # logits 的形状: (N, out_classes)
        logits = self.mlp(feature_embedding)
        
        return logits