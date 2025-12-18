
from .base_learner import BaseLearner
from argparse import Namespace
from torch.nn import Module
import torch
import torch.nn as nn
from dgl import DGLGraph
import dgl
from Algorithm.common.pd_loss_fun import kl_distillation_loss, kl_distillation_loss_v2

class GATDDQNLearner(BaseLearner):
    def __init__(self, config: Namespace, policy: Module):
        super(GATDDQNLearner, self).__init__(config, policy)
        self.optimizer = torch.optim.Adam(self.policy.qNetwork.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=config.lr_decay_steps)

        self.student_optimizer = torch.optim.Adam(self.policy.sNetwork.parameters(), lr=config.student_learning_rate)
        self.student_scheduler = torch.optim.lr_scheduler.LinearLR(self.student_optimizer, start_factor=1.0, end_factor=0.0, total_iters=config.student_lr_decay_steps)

        self.gamma = config.gamma

        if config.pd_loss == "MSE":
            self.pd_loss = nn.MSELoss()
        elif config.pd_loss == "Huber":
            self.pd_loss = nn.SmoothL1Loss()
        elif config.pd_loss == "KL":
            self.pd_loss = kl_distillation_loss
        elif config.pd_loss == "KL_V2":
            self.pd_loss = kl_distillation_loss_v2
        else:
            raise AttributeError(f"No pd_loss is implemented for {config.pd_loss}.")


    def update(self, samples, *args):
        self.n_step = args[0] if args else print("No n_step provided to learner update.")
        # 1. 解包数据
        states, actions, rewards, next_states, dones = zip(*samples)

        # 2. 数据转换与移动到设备
        batched_states = dgl.batch(list(states)).to(self.device)
        batched_next_states = dgl.batch(list(next_states)).to(self.device)
        
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 3. 计算 RL Loss
        self.policy.qNetwork.train()
        self.policy.qNetwork.g = batched_states
        current_q_values = self.policy.qNetwork(batched_states.ndata['feat'])
        
        predict_q_values = current_q_values.gather(1, actions.unsqueeze(1) if actions.dim() == 1 else actions)
        