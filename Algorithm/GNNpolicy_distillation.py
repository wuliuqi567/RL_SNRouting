import logging
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
from Class.experienceReplay import ExperienceReplay
from configure import *
from Utils.utilsfunction import *
from Utils.statefunction import *
import swanlab
from copy import deepcopy
from .kits import *
import argparse
from .graphUtils.gutils import set_seeds, reorginize_self_loop_edges
from .GNNmodel.utils import save_config, save_model, remove_models
from .GNNmodel.MAGNA_model import MAGNA
from dgl import DGLGraph
import dgl

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument('--cuda', default=True, action='store_true', help='use GPU')
    parser.add_argument('--do_train', default=True, action='store_true')
    parser.add_argument("--num_heads", type=int, default=2,
                        help="number of hidden attention heads")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="number of hidden layers")
    parser.add_argument("--project_dim", type=int, default=-1,
                        help="projection dimension")
    parser.add_argument("--num_hidden", type=int, default=24,
                        help="number of hidden units")
    # parser.add_argument("--residual", action="store_true", default=True,
    #                     help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.25,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=0.25,
                        help="attention dropout")
    parser.add_argument("--edge_drop", type=float, default=.1,
                        help="edge dropout")
    parser.add_argument("--clip", type=float, default=1.0, help="grad_clip")
    parser.add_argument("--alpha", type=float, default=.15,
                        help="alpha")
    parser.add_argument("--hop_num", type=int, default=3,
                        help="hop number")
    parser.add_argument("--p_norm", type=int, default=0.0,
                        help="p_norm")
    parser.add_argument("--layer_norm", type=bool, default=True)
    parser.add_argument("--feed_forward", type=bool, default=True)
    # parser.add_argument("--topk_type", type=str, default='local',
    #                     help="topk type")
    # parser.add_argument("--patience", type=int, default=300, help="patience")
    parser.add_argument('-save', '--save_path', default='./MAGNA-models/', type=str)
    # parser.add_argument("--lr", type=float, default=0.0002,
    #                     help="learning rate")
    # parser.add_argument("--lr_reduce_factor", type=float, default=0.5, help="Please give a value for lr_reduce_factor")
    # parser.add_argument("--lr_schedule_patience", type=float, default=25, help="Please give a value for lr_reduce_patience")
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help="weight decay")
    parser.add_argument('--negative_slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--self_loop', default=1, type=int, help='whether self-loop')
    # parser.add_argument('--early-stop', action='store_true', default=False,
    #                     help="indicates whether to use early stop or not")
    # parser.add_argument('--fastmode', action="store_true", default=False,
    #                     help="skip re-evaluate the validation set")
    parser.add_argument('--head_tail_shared', type=int, default=1,
                        help="random seed")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed")
    args = parser.parse_args(args)
    return args

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        full_logits = model(features)
        logits = full_logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels), full_logits

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def preprocess(args):
    random_seed = args.seed
    set_seeds(random_seed)
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    set_logger(args)
    logging.info("Model information...")
    for key, value in vars(args).items():
        logging.info('\t{} = {}'.format(key, value))

    model_folder_name = args2foldername(args)
    model_save_path = os.path.join(args.save_path, model_folder_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    save_config(args, model_save_path)
    logging.info('Model saving path: {}'.format(model_save_path))
    return model_save_path

def args2foldername(args):
    folder_name = "lyer_" + str(args.num_layers) + 'hs_' + str(args.num_heads) + \
                 'ho_' + str(args.hop_num) + 'hi_' + str(args.num_hidden) + \
                 'pd_' + str(args.project_dim) + 'ind_' + str(round(args.in_drop, 4)) + \
                 'att_' + str(round(args.attn_drop, 4)) + 'ed_' + str(round(args.edge_drop, 4)) + 'alpha_' + \
                 str(round(args.alpha, 3)) + 'decay_' + str(round(args.weight_decay, 6))
    return folder_name



class GNNTSDDQNetwork:
    def __init__(self, NGT, hyperparams, earth, sat_ID=None):
        # 设备设置 - 根据配置选择设备
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        args = parse_args()
        self.model_save_path = preprocess(args)

            
        self.actions = ('U', 'D', 'R', 'L')
        self.n_order_adj = hyperparams.n_order_adj    # 假设hyperparams包含该属性
        # 定义状态空间
        args.hop_num =  self.n_order_adj
        if self.n_order_adj:
            self.states = 10 # dimension of each node feature
        
        # num of node is 2*n(n+1)+1, n is the order of adjacency matrix
        
        self.actionSize = len(self.actions)
        self.stateSize = 10 * (2 * self.n_order_adj * (self.n_order_adj + 1) + 1)
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
        self.train_epoch = hyperparams.train_epoch
        self.batchS = hyperparams.batchSize
        self.bufferS = hyperparams.bufferSize
        self.hardUpd = hyperparams.hardUpdate
        self.importQ = hyperparams.importQ
        # self.online = hyperparams.online
        self.ddqn = hyperparams.ddqn  # 新增：是否启用DDQN
        self.algorithm = hyperparams.pathing
        self.outputPath = hyperparams.outputPath if hasattr(hyperparams, 'outputPath') else '../Results'
        self.distillationLR = hyperparams.distillationLR if hasattr(hyperparams, 'distillationLR') else 0.00005
        self.distillationLossFun = hyperparams.distillationLossFun if hasattr(hyperparams, 'distillationLossFun') else 'MSE'
        self.step = 0
        self.i = 0
        self.epsilon = []
        self.experienceReplay = ExperienceReplay(self.bufferS)  # 假设ExperienceReplay已适配PyTorch

        num_feats = self.states
        n_order_adj = self.n_order_adj
        heads = [args.num_heads] * args.num_layers
        self_loop_number = 1

        # 初始化网络
        if Train:
            self.qNetwork = MAGNA(
                num_layers=args.num_layers,
                input_dim=num_feats,
                project_dim=args.project_dim,
                hidden_dim=args.num_hidden,
                action_dim = self.actionSize,
                n_order_adj=n_order_adj,
                heads=heads,
                feat_drop=args.in_drop,
                attn_drop=args.attn_drop,
                alpha=args.alpha,
                hop_num=args.hop_num,
                # top_k=args.top_k,
                # topk_type=args.topk_type,
                edge_drop=args.edge_drop,
                layer_norm=args.layer_norm,
                feed_forward=args.feed_forward,
                self_loop_number=self_loop_number,
                self_loop=(args.self_loop==1),
                head_tail_shared=(args.head_tail_shared == 1),
                negative_slope=args.negative_slope).to(self.device)


            # self.qNetwork = QNetwork(self.stateSize, self.actionSize).to(self.device)
            if self.ddqn:
                self.qTarget =MAGNA(
                    num_layers=args.num_layers,
                    input_dim=num_feats,
                    project_dim=args.project_dim,
                    hidden_dim=args.num_hidden,
                    action_dim = self.actionSize,
                    n_order_adj=n_order_adj,
                    heads=heads,
                    feat_drop=args.in_drop,
                    attn_drop=args.attn_drop,
                    alpha=args.alpha,
                    hop_num=args.hop_num,
                    # top_k=args.top_k,
                    # topk_type=args.topk_type,
                    edge_drop=args.edge_drop,
                    layer_norm=args.layer_norm,
                    feed_forward=args.feed_forward,
                    self_loop_number=self_loop_number,
                    self_loop=(args.self_loop==1),
                    head_tail_shared=(args.head_tail_shared == 1),
                    negative_slope=args.negative_slope).to(self.device)
                self.sNetwork = MAGNA(
                    num_layers=args.num_layers,
                    input_dim=num_feats,
                    project_dim=args.project_dim,
                    hidden_dim=args.num_hidden,
                    action_dim = self.actionSize,
                    n_order_adj=n_order_adj,
                    heads=heads,
                    feat_drop=args.in_drop,
                    attn_drop=args.attn_drop,
                    alpha=args.alpha,
                    hop_num=args.hop_num,
                    # top_k=args.top_k,
                    # topk_type=args.topk_type,
                    edge_drop=args.edge_drop,
                    layer_norm=args.layer_norm,
                    feed_forward=args.feed_forward,
                    self_loop_number=self_loop_number,
                    self_loop=(args.self_loop==1),
                    head_tail_shared=(args.head_tail_shared == 1),
                    negative_slope=args.negative_slope).to(self.device)
                self.hard_update_target()  # 初始同步权重 for teacher and target teacher 
                self.update_student()  # 初始同步权重 for student
            if sat_ID is None:
                print("Q-NETWORK created:")
                print(self.qNetwork)
                print(f"Network moved to device: {self.device}")
            else:
                print(f"Satellite {sat_ID} Q-Network initialized on {self.device}")
        # else:
        #     # 加载预训练模型（需确保路径正确）
        #     try:
        #         self.qNetwork = QNetwork(self.stateSize, self.actionSize).to(self.device)
        #         self.qNetwork.load_state_dict(torch.load(self.outputPath + 'NNs/qNetwork_' + str(len(earth.gateways)) + 'GTs' + '.pth', map_location=self.device))
        #         if self.ddqn:
        #             self.qTarget = QNetwork(self.stateSize, self.actionSize).to(self.device)
        #             self.qTarget.load_state_dict(torch.load(self.outputPath + 'NNs/qTarget_' + str(len(earth.gateways)) + 'GTs' + '.pth', map_location=self.device))
        #             self.sNetwork = QNetwork(self.stateSize, self.actionSize).to(self.device)
        #             self.sNetwork.load_state_dict(torch.load(self.outputPath + 'NNs/sNetwork_' + str(len(earth.gateways)) + 'GTs' + '.pth', map_location=self.device))
        #         if sat_ID is None:
        #             print("Q-Network imported!!!")
        #             print(f"Network loaded on device: {self.device}")
        #         else:
        #             print(f"Satellite {sat_ID} Q-Network imported on {self.device}!")
        #     except FileNotFoundError:
        #         print("Wrong Neural Network path")

        # 优化器和损失函数
        self.optimizer = optim.Adam(self.qNetwork.parameters(), lr=self.alpha, eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0,
                                                           end_factor=0.5,
                                                           total_iters=10000000)
        self.student_optimizer = optim.Adam(self.sNetwork.parameters(), lr=self.distillationLR)
        self.loss_fn = nn.SmoothL1Loss()  # Huber损失对应PyTorch的SmoothL1Loss
        if self.distillationLossFun == 'MSE':
            self.distillation_loss_fn = nn.MSELoss()
        elif self.distillationLossFun == 'Huber':
            self.distillation_loss_fn = nn.SmoothL1Loss()
        elif self.distillationLossFun == 'KL':
            self.distillation_loss_fn = kl_distillation_loss
        elif self.distillationLossFun == 'KL_v2':
            self.distillation_loss_fn = kl_distillation_loss_v2
        self.mse_loss = nn.MSELoss()  # 均方误差损失
        
        # SwanLab初始化标志
        self.swanlab_initialized = True
        if self.swanlab_initialized and Train:
            init_swanlab(hyperparams)



    def getNextHop_dgl(self, newState: DGLGraph, linkedSats, sat, block):
        # 转换状态为PyTorch张量并移动到设备
        # state_tensor = torch.tensor(newState, dtype=torch.float32, device=self.device).unsqueeze(0)
        newState = newState.to(self.device)

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
            self.qNetwork.eval()
            if importSnetwork:
                self.sNetwork.eval()
            with torch.no_grad():
                if importSnetwork:
                    self.sNetwork.g = newState
                    qValues = self.sNetwork(newState.ndata['feat'])
                else:
                    self.qNetwork.g = newState
                    qValues = self.qNetwork(newState.ndata['feat'])
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

    def makeDeepAction(self, block, sat, g, earth, prevSat=None, *args):

        linkedSats = getDeepLinkedSats(sat, g, earth)
        new_state_g_dgl = get_subgraph_state(block, sat, g, earth)
        
        self.step += 1
       
        # if hop exceed max hops, return -1
        if len(block.QPath) > 110:
            if sat.linkedGT and block.destination.ID == sat.linkedGT.ID:
                pass
            else:
                hop_penalty = -ArriveReward
                satDest = block.destination.linkedSat[1]
                distanceReward  = getDistanceRewardV4(prevSat, sat, satDest, self.w2, self.w4)
                queueReward     = getQueueReward   (block.queueTime[len(block.queueTime)-1], self.w1)
                reward = hop_penalty + distanceReward + queueReward
                # 添加奖励大小约束到-1到1之间
                
                                                        
                if not args:
                    block.stepReward.append(reward)
                else:
                    block.stepReward[-1] = reward
                self.experienceReplay.store(block.oldState, block.oldAction, reward, new_state_g_dgl, True)
                self.earth.rewards.append([reward, sat.env.now])
                if Train:
                    log_reward(sum(block.stepReward) if block.stepReward else reward)
                return -1
        # 检查是否到达目标网关
        if sat.linkedGT and block.destination.ID == sat.linkedGT.ID:
            # 计算奖励并存储经验
            if distanceRew == 4:
                satDest = block.destination.linkedSat[1]
                distanceReward  = getDistanceRewardV4(prevSat, sat, satDest, self.w2, self.w4)
                queueReward     = getQueueReward   (block.queueTime[len(block.queueTime)-1], self.w1)
                reward          = distanceReward + queueReward + ArriveReward

            elif distanceRew == 5:
                distanceReward  = getDistanceRewardV5(prevSat, sat, self.w2)
                reward          = distanceReward + ArriveReward
            else:
                reward = ArriveReward  # 需根据具体逻辑调整

            if not args:
                block.stepReward.append(reward)
            else:
                block.stepReward[-1] = reward
            self.experienceReplay.store(block.oldState, block.oldAction, reward, new_state_g_dgl, True)
            self.earth.rewards.append([reward, sat.env.now])
            
            # 记录到达奖励
            if Train:
                log_reward(sum(block.stepReward) if block.stepReward else reward)
            
            return 0
        # 选择动作
        nextHop, actIndex = self.getNextHop_dgl(new_state_g_dgl, linkedSats, sat, block)
        info = {
            "epsilon": self.epsilon[-1][0] if self.epsilon else 0.0,
            # "simulation_time": sat.env.now
            }
        swanlab.log(info)

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

            if distanceRew == 1:
                distanceReward  = getDistanceReward(prevSat, sat, block.destination, self.w2)
            elif distanceRew == 2:
                prevLinkedSats  = getDeepLinkedSats(prevSat, g, earth)
                distanceReward  = getDistanceRewardV2(prevSat, sat, prevLinkedSats['U'], prevLinkedSats['D'], prevLinkedSats['R'], prevLinkedSats['L'], block.destination, self.w2)
            elif distanceRew == 3:
                prevLinkedSats  = getDeepLinkedSats(prevSat, g, earth)
                distanceReward  = getDistanceRewardV3(prevSat, sat, prevLinkedSats['U'], prevLinkedSats['D'], prevLinkedSats['R'], prevLinkedSats['L'], block.destination, self.w2)
            elif distanceRew == 4:
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

            if not args:
                block.stepReward.append(reward)
            else:
                if len(block.stepReward) > 0:
                    block.stepReward[-1] = reward
                else:
                    block.stepReward.append(reward)
            # 存储经验
            self.experienceReplay.store(block.oldState, block.oldAction, reward, new_state_g_dgl, False)
            # self.earth.rewards.append([reward, sat.env.now])
            # log_reward(reward)


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

        block.oldState = new_state_g_dgl
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

    def update_student(self):
        """将教师网络的权重复制到学生网络"""
        self.sNetwork.load_state_dict(self.qNetwork.state_dict())

    def train(self, sat, earth):
        if len(self.experienceReplay.buffer) < self.batchS:
            return -1
        for _ in range(self.train_epoch):
            # 从经验回放中采样
            miniBatch = self.experienceReplay.getBatch(self.batchS)
            states, actions, rewards, next_states, dones = zip(*miniBatch)
              
            # 使用 dgl.batch 将多个图合并为一个大图
            batched_states = dgl.batch(list(states)).to(self.device)
            batched_next_states = dgl.batch(list(next_states)).to(self.device)
            
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

            # 计算当前Q值
            self.qNetwork.train()
            self.qNetwork.g = batched_states
            current_q_values = self.qNetwork(batched_states.ndata['feat'])
            
            predict_q_values = current_q_values.gather(1, actions.unsqueeze(1) if actions.dim() == 1 else actions)

            # 计算目标Q值（DDQN逻辑）
            with torch.no_grad():
                if self.ddqn:
                    self.qNetwork.g = batched_next_states
                    next_q_values_online = self.qNetwork(batched_next_states.ndata['feat'])
                    
                    self.qTarget.g = batched_next_states
                    next_q_values_target = self.qTarget(batched_next_states.ndata['feat'])
                    
                    next_action = next_q_values_online.argmax(dim=1, keepdim=True)
                    next_q_values = next_q_values_target.gather(1, next_action)
                else:
                    self.qNetwork.g = batched_next_states
                    all_next_q_values = self.qNetwork(batched_next_states.ndata['feat'])
                    
                    next_q_values = all_next_q_values.max(dim=1, keepdim=True)[0]
                
                target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

            # 计算损失
            # RL损失：主网络学习强化学习任务 loss_fn
            rl_loss = self.loss_fn(target_q_values, predict_q_values) 
            # 优化主网络
            self.optimizer.zero_grad()
            rl_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.qNetwork.parameters(), 0.5)  
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0] if self.scheduler is not None else self.alpha
            # 记录损失
            earth.loss.append([rl_loss.item(), sat.env.now])
            earth.trains.append([sat.env.now])
            
            if self.step > 24000:
                pd_graph = batched_states.local_var() # 创建一个新的图，然后在这个图上进行操作，避免修改原始图，添加新属性不影响原图
                # 将batched_states图中除了is_center和is_first_order节点特征外的其他特征设置为0
                # 修复 KeyError: 'pdfeat' 并优化性能
                keep_mask = pd_graph.ndata['is_center'] | pd_graph.ndata['is_first_order']
                # 扩展 mask 维度以匹配 feat (N, D)
                mask_expanded = keep_mask.unsqueeze(1).float()
                
                pd_graph.ndata['pdfeat'] = pd_graph.ndata['feat'] * mask_expanded
                
                self.sNetwork.train()
                self.sNetwork.g = pd_graph
                student_q_values = self.sNetwork(pd_graph.ndata['pdfeat'])
                
                if self.distillationLossFun == 'KL':
                    distill_loss = self.distillation_loss_fn(student_q_values, current_q_values.detach(), temperature=5.0)
                elif self.distillationLossFun == 'KL_v2':
                    distill_loss = self.distillation_loss_fn(student_q_values, current_q_values.detach(), temperature=5.0)
                else:
                    distill_loss = self.distillation_loss_fn(current_q_values.detach(), student_q_values)

                # 优化学生网络（蒸馏任务）
                self.student_optimizer.zero_grad()
                distill_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.sNetwork.parameters(), 0.5)  
                self.student_optimizer.step()

            # SwanLab日志记录
            
                if hasattr(self, 'swanlab_initialized') and self.swanlab_initialized:
                    info = {
                        "RLloss": rl_loss.item(),
                        "DistillLoss": distill_loss.item(),
                        "learning_rate": lr,
                        "predictQ": target_q_values.mean().item(),
                        # "epsilon": self.epsilon[-1][0] if self.epsilon else 0.0,
                        "simulation_time": sat.env.now
                    }
                    swanlab.log(info)
            
            else:
                if hasattr(self, 'swanlab_initialized') and self.swanlab_initialized:
                    info = {
                        "RLloss": rl_loss.item(),
                        "learning_rate": lr,
                        "predictQ": target_q_values.mean().item(),
                        # "epsilon": self.epsilon[-1][0] if self.epsilon else 0.0,
                        "simulation_time": sat.env.now
                    }
                    swanlab.log(info)
            
        return 


def kl_distillation_loss(student_outputs, teacher_outputs, temperature):
    """
    计算KL散度损失，用于策略蒸馏。

    loss_kl = \sum_{i = 1}^N softmax(\frac{Q^T}{t} \ln \frac{softmax(\frac{Q^T}{t})} {softmax(\frac{Q^S}{t})})

    参数:
    student_outputs: 学生网络的输出 (logits)
    teacher_outputs: 教师网络的输出 (logits)
    temperature: 温度参数，控制软化程度
    
    返回:
    KL散度损失值
    """
    # 应用温度缩放
    student_logits = student_outputs / 1
    teacher_logits = teacher_outputs / temperature
    
    # 计算softmax概率分布
    student_probs = nn.functional.softmax(student_logits, dim=1)  # softmax(Q^S/t)
    teacher_probs = nn.functional.softmax(teacher_logits, dim=1)  # softmax(Q^T/t)
    
    # 根据公式计算KL散度: ∑ softmax(Q^T/t) * ln(softmax(Q^T/t) / softmax(Q^S/t))
    # 等价于: ∑ teacher_probs * ln(teacher_probs / student_probs)
    # 为了数值稳定性，使用 log(teacher_probs) - log(student_probs)
    log_teacher_probs = torch.log(teacher_probs + 1e-8)  # 加小值避免log(0)
    log_student_probs = torch.log(student_probs + 1e-8)
    
    # KL散度计算: teacher_probs * (log_teacher_probs - log_student_probs)
    kl_loss = torch.sum(teacher_probs * (log_teacher_probs - log_student_probs), dim=1)
    kl_loss = torch.mean(kl_loss) * (temperature ** 2)  # 温度平方缩放，返回标量
    
    return kl_loss

def negative_log_likelihood_loss(student_outputs, teacher_outputs):
    """
    计算负对数似然损失，用于策略蒸馏。

    loss_nll = - \sum_{i = 1}^N argmax(Q^T) \ln softmax(Q^S)

    参数:
    student_outputs: 学生网络的输出 (logits)
    teacher_outputs: 教师网络的输出 (logits)

    返回:
    负对数似然损失值
    """
    # 计算教师网络的动作概率分布
    teacher_probs = nn.functional.softmax(teacher_outputs, dim=1)
    # 获取教师网络选择的动作索引
    _, teacher_actions = torch.max(teacher_probs, dim=1)

    # 计算学生网络的动作概率分布
    student_log_probs = nn.functional.log_softmax(student_outputs, dim=1)

    # 计算负对数似然损失
    nll_loss = nn.NLLLoss()  # 默认 reduction='mean'，已经返回平均值
    loss = nll_loss(student_log_probs, teacher_actions)  # 返回标量

    return loss


def kl_distillation_loss_v2(student_outputs, teacher_outputs, temperature):
    """
    计算双重KL散度损失，用于策略蒸馏。

    loss_kl = \sum_{i = 1}^N softmax(\frac{Q^T}{t} \ln \frac{softmax(\frac{Q^T}{t})} {softmax(\frac{Q^S}{t})})

    参数:
    student_outputs: 学生网络的输出 (logits)
    teacher_outputs: 教师网络的输出 (logits)
    temperature: 温度参数，控制软化程度

    返回:
    KL散度损失值
    """
    kl_loss = kl_distillation_loss(student_outputs, teacher_outputs, temperature)
    nll_loss = negative_log_likelihood_loss(student_outputs, teacher_outputs)
    return nll_loss*0.5 + kl_loss*0.5



# 定义Q网络结构（PyTorch版）
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, dropout_rate=0.2):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            # 第一层：输入层到隐藏层
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 第二层：扩展特征表示
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 第三层：特征压缩
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 输出层
            nn.Linear(64, action_size)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.layers(x)

