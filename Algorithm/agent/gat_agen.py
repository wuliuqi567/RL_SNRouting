from argparse import Namespace
from copy import deepcopy
from torch.nn import Module
import torch
import numpy as np
import os

from Algorithm.common.experienceReplay import ExperienceReplay
from Utils.utilsfunction import *
from Utils.statefunction import *
from .base_agent import BaseAgent

from Algorithm.learner.gatddqn_learner import GATDDQNLearner
from dgl import DGLGraph
import dgl, random
import argparse
from agent.base_agent import get_configs, save_configs


class GATgent(BaseAgent):
    def __init__(self, config: Namespace):
        configs_dict = get_configs("../algo_config/gat_pd.yaml")
        config = argparse.Namespace(**configs_dict)
        # save_configs(configs_dict, os.path.join(config.outputPath, "configs_used.yaml"))
        super(GATgent, self).__init__(config)
        self.env_config = config.env_config
        self.agent_config = config.agent_config
        self.learner_config = config.learner_config

        # 构建策略网络
        self.policy = self._build_policy().to(self.device)

        # 构建学习器
        self.learner = GATDDQNLearner(self.learner_config, self.policy)

        # 其他初始化操作
        self.n_step = self.agent_config.n_step


        