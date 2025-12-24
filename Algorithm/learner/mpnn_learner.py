
from .base_learner import BaseLearner
from argparse import Namespace
from torch.nn import Module
import torch
import torch.nn as nn
from dgl import DGLGraph
import dgl
# from Algorithm.common.pd_loss_fun import kl_distillation_loss, kl_distillation_loss_v2

class MPNNLearner(BaseLearner):
    def __init__(self, config: Namespace, policy: Module):
        super(MPNNLearner, self).__init__(config, policy)
        self.optimizer = torch.optim.Adam(self.policy.qNetwork.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=config.lr_decay_steps)

        # self.student_optimizer = torch.optim.Adam(self.policy.sNetwork.parameters(), lr=config.student_learning_rate)
        # self.student_scheduler = torch.optim.lr_scheduler.LinearLR(self.student_optimizer, start_factor=1.0, end_factor=0.0, total_iters=config.student_lr_decay_steps)

        self.gamma = config.gamma

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
        
        # 计算目标 Q 值 (DDQN)
        with torch.no_grad():
            self.policy.qNetwork.g = batched_next_states
            next_q_values_online = self.policy.qNetwork(batched_next_states.ndata['feat'])
            
            self.policy.qTarget.g = batched_next_states
            next_q_values_target = self.policy.qTarget(batched_next_states.ndata['feat'])
            
            next_action = next_q_values_online.argmax(dim=1, keepdim=True)
            next_q_values = next_q_values_target.gather(1, next_action)
            
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        rl_loss = self.loss_fn(target_q_values, predict_q_values)

        # 4. 优化 Q 网络
        self.optimizer.zero_grad()
        rl_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.qNetwork.parameters(), 0.5)  
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()

        # 7. 返回信息
        info ={
            "lr": self.optimizer.param_groups[0]['lr'],
            "rl_loss": rl_loss.item(),
            "q_value_mean": target_q_values.mean().item(),
        }
        return info