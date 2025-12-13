
from argparse import Namespace
from copy import deepcopy
from torch import Module
import torch

from Utils.utilsfunction import *
from Utils.statefunction import *
from .base_agent import BaseAgent
from GNNmodel.MAGNA_model import MAGNA
from learner.mhgnnddqn_learner import MHGNNDDQNLearner
from dgl import DGLGraph
import dgl, random
import argparse
import yaml

def get_configs(file_dir):
    """Get dict variable from a YAML file.
    Args:
        file_dir: the directory of the YAML file.

    Returns:
        config_dict: the keys and corresponding values in the YAML file.
    """
    with open(file_dir, "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, file_dir + " error: {}".format(exc)
    return config_dict

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

        self.nTrain = config.nTrain
        self.step = 0
        self.updateF_count = 0
        self.updateF = config.updateF

    def _build_policy(self) -> Module:
        
        return MHGNNNetwork(config=self.config)

    def _build_learner(self, *args):
        return MHGNNDDQNLearner(self.config, self.policy)

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
            if self.use_student_network:
                self.policy.sNetwork.eval()
            else:
                self.policy.qNetwork.eval()

            with torch.no_grad():
                if self.use_student_network:
                    self.policy.sNetwork.g = newState
                    qValues = self.policy.sNetwork(newState.ndata['1st_order_feat'])
                else:
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
        linkedSats = getDeepLinkedSats(sat)
        new_state_g_dgl = get_subgraph_state(block, sat, g, earth, n_order=self.n_order_adj)
        self.step += 1

        # 1. Check for Max Hops Failure
        if len(block.QPath) > 110:
            if not (sat.linkedGT and block.destination.ID == sat.linkedGT.ID):
                reward = self._calculate_reward_v1(block, sat, prevSat, g, earth, is_failure=True)
                self._store_experience(block, reward, new_state_g_dgl, True, args, sat, earth)
                if self.train_TA_model:
                    self.log_infos_no_index(sum(block.stepReward) if block.stepReward else reward)
                return -1

        # 2. Check for Destination Arrival
        if sat.linkedGT and block.destination.ID == sat.linkedGT.ID:
            reward = self._calculate_reward_v1(block, sat, prevSat, g, earth, is_terminal=True)
            self._store_experience(block, reward, new_state_g_dgl, True, args, sat, earth)
            if self.train_TA_model:
                self.log_infos_no_index(sum(block.stepReward) if block.stepReward else reward)
            return 0

        # 3. Select Action
        nextHop, actIndex = self.getNextHop(new_state_g_dgl, linkedSats, sat, earth)
        if self.train_TA_model:
            self.log_infos_no_index({"epsilon": self.epsilon[-1][0] if self.epsilon else 0.0})

        if nextHop == -1:
            return 0

        # 4. Intermediate Step Reward
        reward = self._calculate_reward_v1(block, sat, prevSat, g, earth)
        self._store_experience(block, reward, new_state_g_dgl, False, args, sat, earth)

        # 5. Train
        if self.train_TA_model and self.step % self.nTrain == 0:
            self.train(sat, earth)
        
        # 6. Update Target Network
        if self.train_TA_model:
            self.updateF_count += 1
            if self.updateF_count == self.updateF:
                self.policy.hard_update_target()
                self.updateF_count = 0


        block.oldState = new_state_g_dgl
        block.oldAction = actIndex
        return nextHop

    def train(self, sat, earth):
        if len(self.memory) < self.config.batch_size:
            return
        samples = self.memory.getBatch(self.config.batch_size)
        info = self.learner.update(samples, self.step)
        self.log_infos_no_index(info)

        earth.loss.append([info.get('rl_loss', 0.0), sat.env.now])
        earth.trains.append([sat.env.now])
        

    def _store_experience(self, block, reward, new_state, is_terminal, args, sat, earth):
        if not args:
            block.stepReward.append(reward)
        else:
            if len(block.stepReward) > 0:
                block.stepReward[-1] = reward
            else:
                block.stepReward.append(reward)
        
        self.memory.store(block.oldState, block.oldAction, reward, new_state, is_terminal)
        earth.rewards.append([reward, sat.env.now])


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
        
        # Failure case (max hops exceeded)
        if is_failure:
            hop_penalty = -ArriveReward
            satDest = block.destination.linkedSat[1]
            distanceReward = getDistanceRewardV4(prevSat, sat, satDest, w2, w4)
            queueReward = getQueueReward(block.queueTime[-1], w1)
            return hop_penalty + distanceReward + queueReward

        # Success case (arrived at destination)
        if is_terminal:
            if distanceRew == 4:
                satDest = block.destination.linkedSat[1]
                distanceReward = getDistanceRewardV4(prevSat, sat, satDest, w2, w4)
                queueReward = getQueueReward(block.queueTime[-1], w1)
                return distanceReward + queueReward + ArriveReward
            elif distanceRew == 5:
                distanceReward = getDistanceRewardV5(prevSat, sat, w2)
                return distanceReward + ArriveReward
            else:
                return ArriveReward
        
        # Intermediate Step Reward (Default case)
        if distanceRew == 4:
            satDest = block.destination.linkedSat[1]
            distanceReward = getDistanceRewardV4(prevSat, sat, satDest, w2, w4)
            queueReward = getQueueReward(block.queueTime[-1], w1)
            return distanceReward + queueReward
        elif distanceRew == 5:
            return getDistanceRewardV5(prevSat, sat, w2)
        else:
            return 0

    def alignEpsilon(self, step, sat):
        maxEps = self.config.MAX_EPSILON
        minEps = self.config.MIN_EPSILON
        decayRate = self.config.decayRate
        LAMBDA = self.config.LAMBDA
        CurrentGTnumber = self.config.CurrentGTnumber
        epsilon = minEps + (maxEps - minEps) * math.exp(-LAMBDA * step / (decayRate * (CurrentGTnumber**2)))
        self.epsilon.append([epsilon, sat.env.now])
        return epsilon

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
        self.sNetwork = deepcopy(self.qNetwork).to(self.device)

    def hard_update_target(self):
        """硬更新：直接复制Q网络权重到目标网络"""
        self.qTarget.load_state_dict(self.qNetwork.state_dict())

    def update_student(self):
        """将教师网络的权重复制到学生网络"""
        self.sNetwork.load_state_dict(self.qNetwork.state_dict())
