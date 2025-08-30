import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from Class.experienceReplay import ExperienceReplay
from configure import *
from Utils.utilsfunction import *
from Utils.statefunction import *


class DDQNAgent:
    def __init__(self, NGT, hyperparams, earth, sat_ID=None):
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

        self.step = 0
        self.i = 0
        self.epsilon = []
        self.experienceReplay = ExperienceReplay(self.bufferS)  # 假设ExperienceReplay已适配PyTorch

        # 初始化网络
        if not self.importQ:
            self.qNetwork = QNetwork(self.stateSize, self.actionSize)
            if self.ddqn:
                self.qTarget = QNetwork(self.stateSize, self.actionSize)
                self.soft_update_target()  # 初始同步权重
            if sat_ID is None:
                print("Q-NETWORK created:")
                print(self.qNetwork)
            else:
                print(f"Satellite {sat_ID} Q-Network initialized")
        else:
            # 加载预训练模型（需确保路径正确）
            try:
                self.qNetwork = torch.load(nnpath)
                if self.ddqn:
                    self.qTarget = torch.load(nnpathTarget)
                if sat_ID is None:
                    print("Q-Network imported!!!")
                else:
                    print(f"Satellite {sat_ID} Q-Network imported!")
            except FileNotFoundError:
                print("Wrong Neural Network path")

        # 优化器和损失函数
        self.optimizer = optim.Adam(self.qNetwork.parameters(), lr=self.alpha)
        self.loss_fn = nn.SmoothL1Loss()  # Huber损失对应PyTorch的SmoothL1Loss

    def getNextHop(self, newState, linkedSats, sat, block):
        # 转换状态为PyTorch张量
        state_tensor = torch.tensor(newState, dtype=torch.float32).unsqueeze(0)
        
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
            qValues = qValues.numpy()[0]
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
            reward = ArriveReward  # 需根据具体逻辑调整
            self.experienceReplay.store(block.oldState, block.oldAction, reward, newState, True)
            self.earth.rewards.append([reward, sat.env.now])
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
            if hop in block.QPath[:-2]:
                reward -= 1.0  # 惩罚重复访问
            # 计算距离奖励和队列奖励...

            # 存储经验
            self.experienceReplay.store(block.oldState, block.oldAction, reward, newState, False)
            self.earth.rewards.append([reward, sat.env.now])
            if self.online:
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

        # 从经验回放中采样
        miniBatch = self.experienceReplay.getBatch(self.batchS)
        states, actions, rewards, next_states, dones = zip(*miniBatch)
        
        # 转换为张量
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # 计算当前Q值
        current_q_values = self.qNetwork(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值（DDQN逻辑）
        with torch.no_grad():
            if self.ddqn:
                next_action = self.qNetwork(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.qTarget(next_states).gather(1, next_action)
            else:
                next_q_values = self.qNetwork(next_states).max(dim=1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # 计算损失并优化
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 记录损失
        earth.loss.append([loss.item(), sat.env.now])
        earth.trains.append([sat.env.now])
        return loss.item()

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