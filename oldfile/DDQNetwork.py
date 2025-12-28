import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import os
import socket
import time
from pathlib import Path
from Algorithm.common.experienceReplay import ExperienceReplay
from oldfile.configure import *
from Utils.utilsfunction import *
from Utils.statefunction import *
import swanlab

class DDQNAgent:
    def __init__(self, NGT, hyperparams, earth, sat_ID=None):
        # 设备设置 - 根据配置选择设备
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        if print_device_info:
            print(f"DDQNAgent using device: {self.device}")
            if torch.cuda.is_available():
                print(f"GPU available: {torch.cuda.get_device_name()}")
                if not use_gpu:
                    print("Note: GPU is available but use_gpu is set to False in configure.py")
            else:
                print("GPU not available, using CPU")
            
        self.actions = ('U', 'D', 'R', 'L')
        self.reducedState = hyperparams.reducedState  # 假设hyperparams包含该属性
        self.diff_lastHop = hyperparams.diff_lastHop    # 假设hyperparams包含该属性
        # 定义状态空间
        if not self.reducedState:
            self.states = [
                'UpLinked Up', 'UpLinked Down', 'UpLinked Right', 'UpLinked Left',
                'Up Latitude', 'Up Longitude',
                'DownLinked Up', 'DownLinked Down', 'DownLinked Right', 'DownLinked Left',
                'Down Latitude', 'Down Longitude',
                'RightLinked Up', 'RightLinked Down', 'RightLinked Right', 'RightLinked Left',
                'Right Latitude', 'Right Longitude',
                'LeftLinked Up', 'LeftLinked Down', 'LeftLinked Right', 'LeftLinked Left',
                'Left Latitude', 'Left Longitude',
                'Actual latitude', 'Actual longitude',
                'Destination latitude', 'Destination longitude'
            ]
        elif self.reducedState:
            self.states = (
                'Up Latitude', 'Up Longitude',
                'Down Latitude', 'Down Longitude',
                'Right Latitude', 'Right Longitude',
                'Left Latitude', 'Left Longitude',
                'Actual latitude', 'Actual longitude',
                'Destination latitude', 'Destination longitude'
            )
        if self.diff_lastHop:
            self.states = ['Last Hop'] + list(self.states)
        
        self.actionSize = len(self.actions)
        self.stateSize = len(self.states)
        self.destinations = NGT
        self.earth = earth
        self.sat_ID = sat_ID

        # 超参数
        self.alpha = hyperparams.alpha
        self.gamma = hyperparams.gamma
        self.maxEps = hyperparams.MAX_EPSILON
        self.minEps = hyperparams.MIN_EPSILON
        self.w1 = hyperparams.w1
        self.w2 = hyperparams.w2
        self.w4 = hyperparams.w4
        self.tau = hyperparams.tau
        self.updateF = hyperparams.updateF
        self.batchS = hyperparams.batchSize
        self.bufferS = hyperparams.bufferSize
        self.hardUpd = hyperparams.hardUpdate
        self.importQ = hyperparams.importQ
        self.online = hyperparams.online
        self.ddqn = hyperparams.ddqn  # 新增：是否启用DDQN
        self.outputPath = hyperparams.outputPath if hasattr(hyperparams, 'outputPath') else '../Results'

        self.step = 0
        self.i = 0
        self.epsilon = []
        self.experienceReplay = ExperienceReplay(self.bufferS)  # 假设ExperienceReplay已适配PyTorch

        # 初始化网络
        if not self.importQ:
            self.qNetwork = QNetwork(self.stateSize, self.actionSize).to(self.device)
            if self.ddqn:
                self.qTarget = QNetwork(self.stateSize, self.actionSize).to(self.device)
                self.soft_update_target()  # 初始同步权重
            if sat_ID is None:
                print("Q-NETWORK created:")
                print(self.qNetwork)
                print(f"Network moved to device: {self.device}")
            else:
                print(f"Satellite {sat_ID} Q-Network initialized on {self.device}")
        else:
            # 加载预训练模型（需确保路径正确）
            try:
                self.qNetwork = QNetwork(self.stateSize, self.actionSize).to(self.device)
                self.qNetwork.load_state_dict(torch.load(nnpath, map_location=self.device))
                if self.ddqn:
                    self.qTarget = QNetwork(self.stateSize, self.actionSize).to(self.device)
                    self.qTarget.load_state_dict(torch.load(nnpathTarget, map_location=self.device))
                if sat_ID is None:
                    print("Q-Network imported!!!")
                    print(f"Network loaded on device: {self.device}")
                else:
                    print(f"Satellite {sat_ID} Q-Network imported on {self.device}!")
            except FileNotFoundError:
                print("Wrong Neural Network path")

        # 优化器和损失函数
        self.optimizer = optim.Adam(self.qNetwork.parameters(), lr=self.alpha)
        self.loss_fn = nn.SmoothL1Loss()  # Huber损失对应PyTorch的SmoothL1Loss
        self.mse_loss = nn.MSELoss()  # 均方误差损失
        
        # SwanLab初始化标志
        self.swanlab_initialized = True
        if self.swanlab_initialized:
            self.init_swanlab(hyperparams)

    def init_swanlab(self, hyperparams=None, project_name="DDQN_Satellite_Routing", 
                     entity=None, log_dir="./logs"):
        """
        初始化SwanLab日志记录
        
        Args:
            config: 配置字典
            project_name: 项目名称
            entity: 用户/组织名称
            log_dir: 日志目录
        """
        try:            
            # 创建日志目录
            log_path = self.outputPath
            
            
            # 生成时间戳作为实验名称
            time_string = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            experiment_name = f"DDQN_{time_string}"

            # 此 hyperparams是一个类，将其转为 字典
            config = vars(hyperparams)
            # 初始化SwanLab
            swanlab.init(
                config=config,
                project=project_name,
                entity=entity,
                notes=f"Hostname: {socket.gethostname()}",
                logdir=log_path,
                name=experiment_name,
                reinit=True
            )

            print(f"SwanLab initialized for experiment: {experiment_name}")
            
        except Exception as e:
            print(f"Failed to initialize SwanLab: {e}")
            self.swanlab_initialized = False

    def log_metrics(self, metrics_dict, step=None):
        """
        记录指标到SwanLab
        
        Args:
            metrics_dict: 指标字典
            step: 训练步数
        """
        if self.swanlab_initialized:
            try:
                if step is not None:
                    metrics_dict['step'] = step
                swanlab.log(metrics_dict)
            except Exception as e:
                print(f"Failed to log metrics: {e}")

    def log_reward(self, reward, simulation_time):
        """记录奖励"""
        if self.swanlab_initialized:
            self.log_metrics({
                "reward": reward
            })

    def getNextHop(self, newState, linkedSats, sat, block):
        # 转换状态为PyTorch张量并移动到设备
        state_tensor = torch.tensor(newState, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        if explore and random.uniform(0, 1) < self.alignEpsilon(self.step, sat):
            # 随机探索
            actIndex = random.randrange(self.actionSize)
            action = self.actions[actIndex]
            while linkedSats[action] is None:
                self.experienceReplay.store(newState, actIndex, unavPenalty, newState, False)
                self.earth.rewards.append([unavPenalty, sat.env.now])
                actIndex = random.randrange(self.actionSize)
                action = self.actions[actIndex]
        else:
            # 利用Q网络预测
            with torch.no_grad():
                qValues = self.qNetwork(state_tensor)
            qValues = qValues.cpu().numpy().flatten()
            actIndex = np.argmax(qValues)
            action = self.actions[actIndex]
            
            # 处理不可用动作
            while linkedSats[action] is None:
                self.experienceReplay.store(newState, actIndex, unavPenalty, newState, False)
                self.earth.rewards.append([unavPenalty, sat.env.now])
                qValues[actIndex] = -np.inf  # 屏蔽不可用动作
                actIndex = np.argmax(qValues)
                action = self.actions[actIndex]
        
        destination = linkedSats[action]
        if destination is None:
            return -1
        return [destination.ID, math.degrees(destination.longitude), math.degrees(destination.latitude)], actIndex

    def makeDeepAction(self, block, sat, g, earth, prevSat=None):
        linkedSats = getDeepLinkedSats(sat, g, earth)
        
        # 获取状态（需确保getDeepState系列函数返回numpy数组）
        if self.reducedState:
            newState = getDeepStateReduced(block, sat, linkedSats)
        elif self.diff_lastHop:
            newState = getDeepStateDiffLastHop(block, sat, linkedSats)
        else:
            newState = getDeepState(block, sat, linkedSats)
        
        if newState is None:
            earth.lostBlocks += 1
            return 0
        self.step += 1

        # 检查是否到达目标网关
        if sat.linkedGT and block.destination.ID == sat.linkedGT.ID:
            # 计算奖励并存储经验
            if distanceRew == 4:
                satDest = block.destination.linkedSat[1]
                distanceReward  = getDistanceRewardV4(prevSat, sat, satDest, self.w2, self.w4)
                # distanceReward  = getDistanceRewardV4(prevSat, sat, block.destination, self.w2, self.w4)
                queueReward     = getQueueReward   (block.queueTime[len(block.queueTime)-1], self.w1)
                reward          = distanceReward + queueReward + ArriveReward

            elif distanceRew == 5:
                distanceReward  = getDistanceRewardV5(prevSat, sat, self.w2)
                reward          = distanceReward + ArriveReward
            else:
                reward = ArriveReward  # 需根据具体逻辑调整
            self.experienceReplay.store(block.oldState, block.oldAction, reward, newState, True)
            self.earth.rewards.append([reward, sat.env.now])
            
            # 记录到达奖励
            self.log_reward(reward, sat.env.now)
            
            if self.online:
                self.train(sat, earth)
            return 0

        # 选择动作
        nextHop, actIndex = self.getNextHop(newState, linkedSats, sat, block)
        if nextHop == -1:
            return 0

        # 计算奖励（需根据具体逻辑调整）
        reward = 0.0
        if prevSat is not None:
            hop = [sat.ID, math.degrees(sat.longitude), math.degrees(sat.latitude)]
            # if hop in block.QPath[:-2]:
            #     reward -= 1.0  # 惩罚重复访问
            # # 计算距离奖励和队列奖励...
            if hop in block.QPath[:len(block.QPath)-2]:
                again = againPenalty
            else:
                again = 0

            if distanceRew == 4:
                satDest = block.destination.linkedSat[1]
                distanceReward  = getDistanceRewardV4(prevSat, sat, satDest, self.w2, self.w4)
                # distanceReward  = getDistanceRewardV4(prevSat, sat, block.destination, self.w2, self.w4)
            elif distanceRew == 5:
                distanceReward  = getDistanceRewardV5(prevSat, sat, self.w2)

            try:
                queueReward     = getQueueReward   (block.queueTime[len(block.queueTime)-1], self.w1)
            except IndexError:
                queueReward = 0 # FIXME In some hop the queue time was not appended to block.queueTime, line 620
            reward          = distanceReward + again + queueReward

            # 存储经验
            self.experienceReplay.store(block.oldState, block.oldAction, reward, newState, False)
            # self.earth.rewards.append([reward, sat.env.now])
            if self.online:
                self.train(sat, earth)

        if Train and self.step % nTrain == 0:
            self.train(sat, earth)
            
        
        # 更新目标网络
        if self.ddqn:
            if self.hardUpd:
                self.i += 1
                if self.i == self.updateF:
                    self.hard_update_target()
                    self.i = 0
            else:
                self.soft_update_target()

        block.oldState = newState
        block.oldAction = actIndex
        return nextHop

    def alignEpsilon(self, step, sat):
        epsilon = self.minEps + (self.maxEps - self.minEps) * math.exp(-LAMBDA * step / (decayRate * (CurrentGTnumber**2)))
        self.epsilon.append([epsilon, sat.env.now])
        return epsilon

    def hard_update_target(self):
        """硬更新：直接复制Q网络权重到目标网络"""
        self.qTarget.load_state_dict(self.qNetwork.state_dict())

    def soft_update_target(self, tau=None):
        """软更新：指数移动平均"""
        tau = self.tau if tau is None else tau
        for target_param, source_param in zip(self.qTarget.parameters(), self.qNetwork.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def train(self, sat, earth):
        if len(self.experienceReplay.buffer) < self.batchS:
            return -1
        n_epochs = 1  # 每次训练的迭代次数
        for _ in range(n_epochs):
            # 从经验回放中采样
            miniBatch = self.experienceReplay.getBatch(self.batchS)
            states, actions, rewards, next_states, dones = zip(*miniBatch)
            
            # 转换为张量并移动到设备
            states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
            # 确保states是正确的2D张量 [batch_size, state_size]
            if states.dim() == 3 and states.size(1) == 1:
                states = states.squeeze(1)  # [32, 1, 29] -> [32, 29]
            
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
            
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
            # 确保next_states也是正确的2D张量
            if next_states.dim() == 3 and next_states.size(1) == 1:
                next_states = next_states.squeeze(1)  # [32, 1, 29] -> [32, 29]
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

            # 计算当前Q值
            current_q_values = self.qNetwork(states)  # 应该是 [batch_size, action_size]
            
            
            # 如果current_q_values是3维的，需要压缩多余的维度
            if current_q_values.dim() == 3 and current_q_values.size(1) == 1:
                current_q_values = current_q_values.squeeze(1)  # [32, 1, 29] -> [32, 29]
            
            # 确保actions有正确的维度用于gather操作
            if actions.dim() == 1:
                actions = actions.unsqueeze(1)  # [32] -> [32, 1]
            
            # 现在current_q_values应该是[32, 29]，actions是[32, 1]
            current_q_values = current_q_values.gather(1, actions)

            # 计算目标Q值（DDQN逻辑）
            with torch.no_grad():
                if self.ddqn:
                    next_action = self.qNetwork(next_states).argmax(dim=1, keepdim=True)
                    next_q_values = self.qTarget(next_states).gather(1, next_action)
                else:
                    next_q_values = self.qNetwork(next_states).max(dim=1, keepdim=True)[0]
                target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

            # 计算损失并优化
            # loss = self.loss_fn(current_q_values, target_q_values)
            # mseloss = self.mse_loss(current_q_values, target_q_values)
            mseloss = self.mse_loss(target_q_values, current_q_values)
            
            self.optimizer.zero_grad()
            mseloss.backward()
            self.optimizer.step()

            # 记录损失
            earth.loss.append([mseloss.item(), sat.env.now])
            earth.trains.append([sat.env.now])
            
            # SwanLab日志记录
            if hasattr(self, 'swanlab_initialized') and self.swanlab_initialized:
                info = {
                    "train/Qloss": mseloss.item(),
                    "train/learning_rate": self.alpha,
                    "train/predictQ": target_q_values.mean().item(),
                    "train/epsilon": self.epsilon[-1][0] if self.epsilon else 0.0,
                    "train/step": self.step,
                    "train/simulation_time": sat.env.now
                }
                swanlab.log(info)
            
            
        return mseloss.item()

# 定义Q网络结构（PyTorch版）
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
    
    def forward(self, x):
        return self.layers(x)
    
    