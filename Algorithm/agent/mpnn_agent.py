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

from Algorithm.learner.mpnn_learner import MPNNLearner
from dgl import DGLGraph
import dgl, random
import argparse
from .base_agent import get_configs, save_configs
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, AvgPooling


class MPNNAgent(BaseAgent):
    model_name = 'mpnn_model.pth'

    def __init__(self, config: Namespace | None = None):
        configs_dict = get_configs("../algo_config/mpnn.yaml")
        config = argparse.Namespace(**configs_dict)
        # save_configs(configs_dict, os.path.join(config.outputPath, "configs_used.yaml"))
        super(MPNNAgent, self).__init__(config)
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
        return MPNNNetwork(self.config)

    def _build_learner(self, *args):
        return MPNNLearner(self.config, self.policy)

class MPNNNetwork(Module):
    def __init__(self, config: Namespace):
        super(MPNNNetwork, self).__init__()
        # Initialize the MHGNN network components here based on config
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # self.n_order_adj = config.n_order_adj
        self.tau = config.tau

        self.qNetwork = MPNNModel(config.input_dim, config.hidden_feats, config.num_layers, config.mlp_hidden_feats, config.action_size, config.n_order_adj).to(self.device)

        self.qTarget = deepcopy(self.qNetwork).to(self.device)
        self.sNetwork = deepcopy(self.qNetwork).to(self.device)

    def hard_update_target(self):
        """硬更新：直接复制Q网络权重到目标网络"""
        self.qTarget.load_state_dict(self.qNetwork.state_dict())

    def update_student(self):
        """将教师网络的权重复制到学生网络"""
        self.sNetwork.load_state_dict(self.qNetwork.state_dict())


class MPNNModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_layers, mlp_hidden_feats, out_classes, n_order_adj):
        super(MPNNModel, self).__init__()
        
        self.layers = nn.ModuleList()
        # Layer 1
        self.layers.append(GraphConv(in_feats, hidden_feats, allow_zero_in_degree=True))
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.layers.append(GraphConv(hidden_feats, hidden_feats, allow_zero_in_degree=True))
        
        # Calculate number of nodes based on n_order_adj
        # Formula: 2 * n_order_adj * (n_order_adj + 1) + 1
        self.num_nodes = 2 * n_order_adj * (n_order_adj + 1) + 1
        
        # MLP input dimension
        # We concatenate features of all nodes
        mlp_in_dim = hidden_feats * self.num_nodes
        
        hidden_dims = mlp_hidden_feats if isinstance(mlp_hidden_feats, (list, tuple)) else [mlp_hidden_feats]
        hidden_dims = [int(dim) for dim in hidden_dims if dim is not None]

        mlp_layers = []
        prev_dim = mlp_in_dim
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
            ])
            prev_dim = hidden_dim
        mlp_layers.append(nn.Linear(prev_dim, out_classes))

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, h, g=None):
        if g is None:
            g = getattr(self, 'g', None)
        if g is None:
            raise TypeError("MPNNModel.forward requires a DGLGraph via argument 'g' or by setting self.g")

        # MPNN layers
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        
        # Flatten/Reshape logic
        total_nodes = g.num_nodes()
        batch_size = getattr(g, 'batch_size', None)
        if batch_size is None:
            if total_nodes % self.num_nodes != 0:
                 # Fallback or error if graph size doesn't match expected node count
                 # For robustness, we can try to infer batch size or raise error
                 batch_size = total_nodes // self.num_nodes
            else:
                 batch_size = total_nodes // self.num_nodes

        # Reshape to (batch_size, num_nodes * hidden_feats)
        feature_embedding = h.reshape(batch_size, -1)
        feature_embedding = F.elu(feature_embedding)
        
        logits = self.mlp(feature_embedding)
        return logits