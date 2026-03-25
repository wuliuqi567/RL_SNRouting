
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

    def getNextHop(self, newState: DGLGraph, linkedSats, sat, earth):
        # 转换状态为PyTorch张量并移动到设备
        unavPenalty = -10
        newState = newState.to(self.device)
        t1st_order_feat = newState.ndata['1st_order_feat']
        feat = newState.ndata['feat']
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
                if self.use_student_network:
                    self.policy.sNetwork.eval()
                    self.policy.sNetwork.g = newState
                    qValues = self.policy.sNetwork(newState.ndata['1st_order_feat'])
                else:
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
        recalculate_flag = args and args[0] == 'recalculate'
        training_mode = self.train_TA_model and not recalculate_flag
        
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
                self.train(sat, earth) # whether adding train_epoch here needs to be considered
            
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
        w1 = self.w1        # rewards the getting to empty queues
        w2 = self.w2        # rewards getting closes phisycally   
        w4 = self.w4         # Normalization for the distance reward, for the traveled distance factor 
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

    def alignEpsilon(self, step, sat):
        maxEps = self.config.MAX_EPSILON
        minEps = self.config.MIN_EPSILON
        decayRate = self.config.decayRate
        LAMBDA = self.config.LAMBDA
        # CurrentGTnumber = self.config.CurrentGTnumber
        epsilon = minEps + (maxEps - minEps) * math.exp(-LAMBDA * step / (decayRate * (2**2)))
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
        # 加载神经网络权重文件。
        # 这里的最终目标是得到 3 个完整文件路径：
        # 1) qNet_mhgnn_model.pth
        # 2) qTarget_mhgnn_model.pth
        # 3) sNet_mhgnn_model.pth
        model_name = 'mhgnn_model.pth'
        
        # Step 1: 先取配置里保存的模型目录（通常来自 BaseAgent / 配置文件）。
        # self.model_dir_save 可能是绝对路径，也可能是相对路径；
        # 也可能在测试阶段包含 test_teacher_network / test_student_network 标记。
        model_dir = self.model_dir_save

        # Step 2: 如果当前是测试目录命名（例如 .../test_teacher_network1/...），
        # 则把该片段替换成 train，确保测试时读取的是训练阶段保存的权重目录。
        # 正则 test_teacher_network[^/\\]* 表示：
        # - 固定前缀 test_teacher_network
        # - 后面可跟 0 个或多个“非路径分隔符”字符（可包含数字或字符串）
        #   例如：test_teacher_network、test_teacher_network1、test_teacher_network_v2、...
        if 'test_teacher_network' in model_dir:
            model_dir = re.sub(r'test_teacher_network[^/\\]*', 'train', model_dir)
        elif 'test_student_network' in model_dir:
            model_dir = re.sub(r'test_student_network[^/\\]*', 'train', model_dir)
        
        # Step 3: 处理相对路径。
        # 若 model_dir 不是绝对路径（例如 "MHGNN/2026-01-01"），
        # 则基于 self.outputPath 的 ../train/ 进行拼接，得到可访问的完整目录。
        # 也就是：最终目录 = os.path.join(self.outputPath, '../train/', model_dir)
        # 注意：这里保持原有实现逻辑，不改变目录组织规则。
        if not os.path.isabs(model_dir):
             model_dir = os.path.join(self.outputPath, '../train/', model_dir)

        # Step 4: 在上面得到的 model_dir 下拼接具体文件名。
        # 最终得到 3 个权重文件的完整路径。
        qNet_model_path = os.path.join(model_dir, 'qNet_' + model_name)
        qTarget_model_path = os.path.join(model_dir, 'qTarget_' + model_name)
        sNet_model_path = os.path.join(model_dir, 'sNet_' + model_name)

        # 打印 qNet 路径用于调试（可快速确认当前到底从哪个目录加载）。
        print("Loading model from:", qNet_model_path)

        # Step 5: 按设备加载参数（CPU/GPU 由 self.device 决定）。
        # weights_only=True 表示只读取权重张量，不反序列化额外对象。
        self.policy.qNetwork.load_state_dict(torch.load(qNet_model_path, map_location=self.device, weights_only=True))
        self.policy.qTarget.load_state_dict(torch.load(qTarget_model_path, map_location=self.device, weights_only=True))
        try:
            self.policy.sNetwork.load_state_dict(torch.load(sNet_model_path, map_location=self.device, weights_only=True))
        except RuntimeError as e:
            print(f"Warning: skip loading incompatible sNetwork checkpoint: {e}")

    def try_save_model(self):
        if self.train_TA_model:
            self.save_model(model_name='mhgnn_model.pth')


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
