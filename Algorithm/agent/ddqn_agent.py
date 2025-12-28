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
    
    def getNextHop(self, newState: DGLGraph, linkedSats, sat, earth):
        # 转换状态为PyTorch张量并移动到设备
        unavPenalty = -10
        newState = newState.to(self.device)

        if self.train_TA_model and random.uniform(0, 1) < self.alignEpsilon(self.step, sat):
            # 随机探索
            actIndex = random.randrange(self.actionSize)
            action = self.actions[actIndex]
            while linkedSats[action] is None:
                self.memory.store(newState, actIndex, unavPenalty, newState, False) 
                earth.rewards.append([unavPenalty, sat.env.now])
                actIndex = random.randrange(self.actionSize)
                action = self.actions[actIndex]
        else:            
            with torch.no_grad():
                self.policy.qNetwork.eval()
                self.policy.qNetwork.g = newState
                qValues = self.policy.qNetwork(newState.ndata['feat'])
            qValues = qValues.cpu().numpy().flatten()
            actIndex = np.argmax(qValues)
            action = self.actions[actIndex]
            
            # 处理不可用动作
            while linkedSats[action] is None:
                self.memory.store(newState, actIndex, unavPenalty, newState, False)
                earth.rewards.append([unavPenalty, sat.env.now])
                qValues[actIndex] = -np.inf  # 屏蔽不可用动作
                actIndex = np.argmax(qValues)
                action = self.actions[actIndex]
        
        destination = linkedSats[action]
        if destination is None:
            return -1
        return [destination.ID, math.degrees(destination.longitude), math.degrees(destination.latitude)], actIndex


    def makeDeepAction(self, block, sat, g, earth, prevSat=None, *args):
        """
        Docstring for makeDeepAction
        
        if block reaches destination, return 0
        if no available action, return -2
        if exceed max hops, return -1
        else return nextHop: ['18_20', -133.12012370027912, 42.210392220392556] dest.ID, longitude, latitude
        """
        # 1. Configuration and Status
        training_mode = self.train_TA_model
        
        is_reached = sat.linkedGT and block.destination.ID == sat.linkedGT.ID
        is_failure = len(block.QPath) > system_configure.Max_Hops and not is_reached
        
        # 2. Handle Terminal States (Success or Failure)
        if is_reached or is_failure:
            if training_mode and prevSat is not None:
                new_state_g_dgl = get_subgraph_state(block, sat, g, earth, n_order=self.n_order_adj)
                self.step += 1
                reward = self._calculate_reward_v1(block, sat, prevSat, is_terminal=is_reached, is_failure=is_failure)
                self.store_experience(block, reward, new_state_g_dgl, True, sat, earth)
                self.log_infos_no_index({"Reward": sum(block.stepReward) if block.stepReward else reward})
            return -1 if is_failure else 0

        # 3. Prepare State
        linkedSats = getDeepLinkedSats(sat, g, earth)
        new_state_g_dgl = get_subgraph_state(block, sat, g, earth, n_order=self.n_order_adj)
        self.step += 1

        # 4. Get Action
        nextHop, actIndex = self.getNextHop(new_state_g_dgl, linkedSats, sat, earth)
        
        if nextHop == -1:
            if not training_mode:
                print(f"Error in nextHop calculation: Sat {sat.ID}, block {block}")
            return -2

        # 5. Training Logic (Only in training mode)
        if training_mode:
            self.log_infos_no_index({"epsilon": self.epsilon[-1][0] if self.epsilon else 0.0})
            
            if prevSat is not None:
                reward = self._calculate_reward_v1(block, sat, prevSat)
                self.store_experience(block, reward, new_state_g_dgl, False, sat, earth)

            if self.step % self.nTrain == 0:
                self.train(sat, earth)
            
                self.updateF_count += 1
                if self.updateF_count == self.updateF:
                    self.policy.hard_update_target()
                    self.updateF_count = 0

            block.oldState = new_state_g_dgl
            block.oldAction = actIndex

        return nextHop

    def train(self, sat, earth):
        if self.memory.buffeSize < self.config.batch_size:
            return
        for _ in range(self.train_epoch):
            samples = self.memory.getBatch(self.config.batch_size)
            info = self.learner.update(samples, self.step)
            self.log_infos_no_index(info)

        earth.loss.append([info.get('rl_loss', 0.0), sat.env.now])
        earth.trains.append([sat.env.now])
        

    def store_experience(self, block, reward, new_state, is_terminal, sat, earth, recalculate_flag=False):
        if not recalculate_flag:
            block.stepReward.append(reward)
        else:
            if block.stepReward:
                block.stepReward[-1] = reward
            else:
                block.stepReward.append(reward)
        self.memory.store(block.oldState, block.oldAction, reward, new_state, is_terminal)
        if is_terminal:
            earth.rewards.append([sum(block.stepReward), sat.env.now])



    def _calculate_reward_v1(self, block, sat, prevSat, is_terminal=False, is_failure=False):
        """Helper to calculate reward based on state."""
        w1 = 20        # rewards the getting to empty queues
        w2 = 20        # rewards getting closes phisycally   
        w4 = 5         # Normalization for the distance reward, for the traveled distance factor 
        ArriveReward = 50        # Reward given to the system in case it sends the data block to the satellite linked to the destination gateway
        distanceRew = 4          # 1: Distance reward normalized to total distance.
                                 # 2: Distance reward normalized to average moving possibilities
                                 # 3: Distance reward normalized to maximum close up
                                 # 4: Distance reward normalized by max isl distance ~3.700 km for Kepler constellation. This is the one used in the papers.
                                 # 5: Only negative rewards proportional to traveled distance normalized by 1.000 km
        againPenalty= -10       # Penalty if the satellite sends the block to a hop where it has already been

        satDest = block.destination.linkedSat[1]
        if satDest is None:
            print("No linked sat for destination GT")
        if prevSat is None:
            assert False, "Previous satellite is None in reward calculation."

        queueReward = 0
        if block.queueTime:
            queueReward = getQueueReward(block.queueTime[-1], w1)
        distanceReward = getDistanceRewardV4(prevSat, sat, satDest, w2, w4)

        if is_failure:
            return distanceReward + queueReward - ArriveReward
        if is_terminal:
            return distanceReward + queueReward + ArriveReward
        
        hop = [sat.ID, math.degrees(sat.longitude), math.degrees(sat.latitude)]
        if hop in block.QPath[:len(block.QPath)-2]:
            again = againPenalty
        else:
            again = 0
        return distanceReward + queueReward + again


    def alignEpsilon(self, step, sat):
        maxEps = self.config.MAX_EPSILON
        minEps = self.config.MIN_EPSILON
        decayRate = self.config.decayRate
        LAMBDA = self.config.LAMBDA
        CurrentGTnumber = self.config.CurrentGTnumber
        epsilon = minEps + (maxEps - minEps) * math.exp(-LAMBDA * step / (decayRate * (CurrentGTnumber**2)))
        self.epsilon.append([epsilon, sat.env.now])
        return epsilon
    
    def save_model(self, model_name):
        # save the neural networks
        if not os.path.exists(self.model_dir_save):
            os.makedirs(self.model_dir_save)
        qNet_model_path = os.path.join(self.model_dir_save, 'qNet_' + model_name)
        qTarget_model_path = os.path.join(self.model_dir_save, 'qTarget_' + model_name)
        sNet_model_path = os.path.join(self.model_dir_save, 'sNet_' + model_name)
        torch.save(self.policy.qNetwork.state_dict(), qNet_model_path)
        torch.save(self.policy.qTarget.state_dict(), qTarget_model_path)
        torch.save(self.policy.sNetwork.state_dict(), sNet_model_path)

    def load_model(self):
        # load neural networks
        model_name = 'ddqn_model.pth'
        
        # Fix for loading model from train directory when testing
        model_dir = self.model_dir_save
        if 'test_teacher_network' in model_dir:
            model_dir = re.sub(r'test_teacher_network\d*', 'train', model_dir)
        elif 'test_student_network' in model_dir:
            model_dir = re.sub(r'test_student_network\d*', 'train', model_dir)
        
        if not os.path.isabs(model_dir):
             model_dir = os.path.join(self.outputPath, '../train/', model_dir)

        qNet_model_path = os.path.join(model_dir, 'qNet_' + model_name)
        qTarget_model_path = os.path.join(model_dir, 'qTarget_' + model_name)
        sNet_model_path = os.path.join(model_dir, 'sNet_' + model_name)

        print("Loading model from:", qNet_model_path)
        self.policy.qNetwork.load_state_dict(torch.load(qNet_model_path, map_location=self.device, weights_only=True))
        self.policy.qTarget.load_state_dict(torch.load(qTarget_model_path, map_location=self.device, weights_only=True))
        self.policy.sNetwork.load_state_dict(torch.load(sNet_model_path, map_location=self.device, weights_only=True))

    def try_save_model(self):
        if self.train_TA_model:
            self.save_model(model_name='ddqn_model.pth')

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