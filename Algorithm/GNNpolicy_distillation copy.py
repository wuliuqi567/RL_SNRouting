import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import os
from Algorithm.common.experienceReplay import ExperienceReplay
# from configure import *
from Utils.utilsfunction import *
from Utils.statefunction import *
import swanlab
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
    parser.add_argument('-save', '--save_path', default='./MAGNA-models/', type=str)
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help="weight decay")
    parser.add_argument('--negative_slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--self_loop', default=1, type=int, help='whether self-loop')
    parser.add_argument('--head_tail_shared', type=int, default=1,
                        help="random seed")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed")
    args = parser.parse_args(args)
    return args


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



class BaseRLAgent:
    """Base class for RL Agents."""
    def __init__(self, NGT, hyperparams, earth, sat_ID=None):
        self.actions = ('U', 'D', 'R', 'L')
        self.actionSize = len(self.actions)
        self.earth = earth
        self.sat_ID = sat_ID
        
        # Hyperparameters
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
        self.ddqn = hyperparams.ddqn
        
        self.step = 0
        self.i = 0
        self.epsilon = []
        self.experienceReplay = ExperienceReplay(self.bufferS)

        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("Using GPU")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

    def alignEpsilon(self, step, sat):
        epsilon = self.minEps + (self.maxEps - self.minEps) * math.exp(-LAMBDA * step / (decayRate * (CurrentGTnumber**2)))
        self.epsilon.append([epsilon, sat.env.now])
        return epsilon

    def hard_update_target(self):
        """Hard update: copy weights from Q network to Target network."""
        if hasattr(self, 'qTarget') and hasattr(self, 'qNetwork'):
            self.qTarget.load_state_dict(self.qNetwork.state_dict())

    def soft_update_target(self, tau=None):
        """Soft update: exponential moving average."""
        if hasattr(self, 'qTarget') and hasattr(self, 'qNetwork'):
            tau = self.tau if tau is None else tau
            for target_param, source_param in zip(self.qTarget.parameters(), self.qNetwork.parameters()):
                target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


class GNNTSDDQNetwork(BaseRLAgent):
    def __init__(self, NGT, hyperparams, earth, sat_ID=None):
        super().__init__(NGT, hyperparams, earth, sat_ID)
        
        self.args = parse_args()
        self.model_save_path = preprocess(self.args)
            
        self.n_order_adj = hyperparams.n_order_adj    # 假设hyperparams包含该属性
        # 定义状态空间
        self.args.hop_num =  self.n_order_adj
        if self.n_order_adj:
            self.states = 10 # dimension of each node feature
        
        self.stateSize = 10 * (2 * self.n_order_adj * (self.n_order_adj + 1) + 1)
        self.destinations = NGT

        # 特有超参数
        self.importQ = hyperparams.importQ
        self.algorithm = hyperparams.pathing
        self.outputPath = hyperparams.outputPath if hasattr(hyperparams, 'outputPath') else '../Results'
        self.distillationLR = hyperparams.distillationLR if hasattr(hyperparams, 'distillationLR') else 0.00005
        self.distillationLossFun = hyperparams.distillationLossFun if hasattr(hyperparams, 'distillationLossFun') else 'MSE'
        
        num_feats = self.states
        
        # 初始化网络
        self.qNetwork = self._build_model(num_feats)
        self.qTarget = self._build_model(num_feats)
        self.sNetwork = self._build_model(num_feats)
                
        self.hard_update_target()  # 初始同步权重 for teacher and target teacher 
        self.update_student()  # 初始同步权重 for student
        print(self.qNetwork)
        

        if Train:
            self.hard_update_target()  # 初始同步权重 for teacher and target teacher 
            self.update_student()  # 初始同步权重 for student
        else:

            # 加载预训练模型（需确保路径正确）
            try:
                self.qNetwork.load_state_dict(torch.load(self.outputPath + 'NNs/qNetwork_' + str(len(earth.gateways)) + 'GTs' + '.pth', map_location=self.device))
                self.qTarget.load_state_dict(torch.load(self.outputPath + 'NNs/qTarget_' + str(len(earth.gateways)) + 'GTs' + '.pth', map_location=self.device))
                self.sNetwork.load_state_dict(torch.load(self.outputPath + 'NNs/sNetwork_' + str(len(earth.gateways)) + 'GTs' + '.pth', map_location=self.device))

            except FileNotFoundError:
                print("Wrong Neural Network path")

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

    def _build_model(self, num_feats):
        """Helper function to build the MAGNA model."""
        heads = [self.args.num_heads] * self.args.num_layers
        return MAGNA(
            num_layers=self.args.num_layers,
            input_dim=num_feats,
            project_dim=self.args.project_dim,
            hidden_dim=self.args.num_hidden,
            action_dim=self.actionSize,
            n_order_adj=self.n_order_adj,
            heads=heads,
            feat_drop=self.args.in_drop,
            attn_drop=self.args.attn_drop,
            alpha=self.args.alpha,
            hop_num=self.args.hop_num,
            edge_drop=self.args.edge_drop,
            layer_norm=self.args.layer_norm,
            feed_forward=self.args.feed_forward,
            self_loop_number=1,
            self_loop=(self.args.self_loop == 1),
            head_tail_shared=(self.args.head_tail_shared == 1),
            negative_slope=self.args.negative_slope
        ).to(self.device)



    def getNextHop_dgl(self, newState: DGLGraph, linkedSats, sat):
        # 转换状态为PyTorch张量并移动到设备

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
                    qValues = self.sNetwork(newState.ndata['1st_order_feat'])
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

    def _calculate_reward(self, block, sat, prevSat, g, earth, is_terminal=False, is_failure=False):
        """Helper to calculate reward based on state."""
        # Failure case (max hops exceeded)
        if is_failure:
            hop_penalty = -ArriveReward
            satDest = block.destination.linkedSat[1]
            distanceReward = getDistanceRewardV4(prevSat, sat, satDest, self.w2, self.w4)
            queueReward = getQueueReward(block.queueTime[-1], self.w1)
            return hop_penalty + distanceReward + queueReward

        # Success case (arrived at destination)
        if is_terminal:
            if distanceRew == 4:
                satDest = block.destination.linkedSat[1]
                distanceReward = getDistanceRewardV4(prevSat, sat, satDest, self.w2, self.w4)
                queueReward = getQueueReward(block.queueTime[-1], self.w1)
                return distanceReward + queueReward + ArriveReward
            elif distanceRew == 5:
                distanceReward = getDistanceRewardV5(prevSat, sat, self.w2)
                return distanceReward + ArriveReward
            else:
                return ArriveReward

        # Intermediate step
        reward = 0.0
        if prevSat is not None:
            hop = [sat.ID, math.degrees(sat.longitude), math.degrees(sat.latitude)]
            again = againPenalty if hop in block.QPath[:len(block.QPath)-2] else 0

            if distanceRew == 1:
                distanceReward = getDistanceReward(prevSat, sat, block.destination, self.w2)
            elif distanceRew == 2:
                prevLinkedSats = getDeepLinkedSats(prevSat, g, earth)
                distanceReward = getDistanceRewardV2(prevSat, sat, prevLinkedSats['U'], prevLinkedSats['D'], prevLinkedSats['R'], prevLinkedSats['L'], block.destination, self.w2)
            elif distanceRew == 3:
                prevLinkedSats = getDeepLinkedSats(prevSat, g, earth)
                distanceReward = getDistanceRewardV3(prevSat, sat, prevLinkedSats['U'], prevLinkedSats['D'], prevLinkedSats['R'], prevLinkedSats['L'], block.destination, self.w2)
            elif distanceRew == 4:
                satDest = block.destination.linkedSat[1]
                distanceReward = getDistanceRewardV4(prevSat, sat, satDest, self.w2, self.w4)
            elif distanceRew == 5:
                distanceReward = getDistanceRewardV5(prevSat, sat, self.w2)
            else:
                distanceReward = 0

            try:
                queueReward = getQueueReward(block.queueTime[-1], self.w1)
            except IndexError:
                queueReward = 0
            
            reward = distanceReward + again + queueReward
            
        return reward

    def _store_experience(self, block, reward, new_state, is_terminal, args, sat):
        """Helper to store experience and update block rewards."""
        if not args:
            block.stepReward.append(reward)
        else:
            if len(block.stepReward) > 0:
                block.stepReward[-1] = reward
            else:
                block.stepReward.append(reward)
        
        self.experienceReplay.store(block.oldState, block.oldAction, reward, new_state, is_terminal)
        self.earth.rewards.append([reward, sat.env.now])

    def makeDeepAction(self, block, sat, g, earth, prevSat=None, *args):
        linkedSats = getDeepLinkedSats(sat, g, earth)
        new_state_g_dgl = get_subgraph_state(block, sat, g, earth, n_order=self.n_order_adj)
        self.step += 1

        # 1. Check for Max Hops Failure
        if len(block.QPath) > 110:
            if not (sat.linkedGT and block.destination.ID == sat.linkedGT.ID):
                reward = self._calculate_reward(block, sat, prevSat, g, earth, is_failure=True)
                self._store_experience(block, reward, new_state_g_dgl, True, args, sat)
                if Train:
                    log_reward(sum(block.stepReward) if block.stepReward else reward)
                return -1

        # 2. Check for Destination Arrival
        if sat.linkedGT and block.destination.ID == sat.linkedGT.ID:
            reward = self._calculate_reward(block, sat, prevSat, g, earth, is_terminal=True)
            self._store_experience(block, reward, new_state_g_dgl, True, args, sat)
            if Train:
                log_reward(sum(block.stepReward) if block.stepReward else reward)
            return 0

        # 3. Select Action
        nextHop, actIndex = self.getNextHop_dgl(new_state_g_dgl, linkedSats, sat)
        if Train:
            swanlab.log({"epsilon": self.epsilon[-1][0] if self.epsilon else 0.0})

        if nextHop == -1:
            return 0

        # 4. Intermediate Step Reward
        reward = self._calculate_reward(block, sat, prevSat, g, earth)
        self._store_experience(block, reward, new_state_g_dgl, False, args, sat)

        # 5. Train
        if Train and self.step % nTrain == 0:
            self.train(sat, earth)

        # 6. Update Target Network
        if self.ddqn and Train:
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
        return super().alignEpsilon(step, sat)

    def hard_update_target(self):
        super().hard_update_target()

    def soft_update_target(self, tau=None):
        super().soft_update_target(tau)

    def update_student(self):
        """将教师网络的权重复制到学生网络"""
        self.sNetwork.load_state_dict(self.qNetwork.state_dict())

    def _compute_rl_loss(self, batched_states, batched_next_states, actions, rewards, dones):
        """Compute RL loss and return intermediate values."""
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

        # RL损失
        rl_loss = self.loss_fn(target_q_values, predict_q_values)
        return rl_loss, current_q_values, target_q_values

    def _compute_distillation_loss(self, batched_states, current_q_values):
        """Compute distillation loss."""
        pd_graph = batched_states.local_var()
        keep_mask = pd_graph.ndata['is_center'] | pd_graph.ndata['is_first_order']
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
        return distill_loss

    def train(self, sat, earth):
        if len(self.experienceReplay.buffer) < self.batchS:
            return -1
        
        for _ in range(self.train_epoch):
            miniBatch = self.experienceReplay.getBatch(self.batchS)
            states, actions, rewards, next_states, dones = zip(*miniBatch)
              
            batched_states = dgl.batch(list(states)).to(self.device)
            batched_next_states = dgl.batch(list(next_states)).to(self.device)
            
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

            # 1. RL Step
            rl_loss, current_q_values, target_q_values = self._compute_rl_loss(
                batched_states, batched_next_states, actions, rewards, dones
            )
            
            self.optimizer.zero_grad()
            rl_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.qNetwork.parameters(), 0.5)  
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0] if self.scheduler is not None else self.alpha
            
            earth.loss.append([rl_loss.item(), sat.env.now])
            earth.trains.append([sat.env.now])
            
            # 2. Distillation Step
            distill_loss_val = 0.0
            if self.step > 24000:
                distill_loss = self._compute_distillation_loss(batched_states, current_q_values)
                distill_loss_val = distill_loss.item()
                
                self.student_optimizer.zero_grad()
                distill_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.sNetwork.parameters(), 0.5)  
                self.student_optimizer.step()

            # 3. Logging
            if hasattr(self, 'swanlab_initialized') and self.swanlab_initialized:
                info = {
                    "RLloss": rl_loss.item(),
                    "learning_rate": lr,
                    "predictQ": target_q_values.mean().item(),
                    "simulation_time": sat.env.now
                }
                if self.step > 24000:
                    info["DistillLoss"] = distill_loss_val
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


