from abc import ABC, abstractmethod
from modulefinder import Module
import os
import numpy as np
import torch
import random
import math
import re
from ..common.common_tools import get_time_string, create_directory
from ..common.experienceReplay import ExperienceReplay
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
# import swanlab as wandb
import wandb
import socket
from argparse import Namespace
import yaml
import sys
import system_configure
from Utils.statefunction import getDeepLinkedSats, get_subgraph_state
from Utils.utilsfunction import getQueueReward, getDistanceRewardV4

def get_configs(file_dir):
    """Get dict variable from a YAML file.
    Args:
        file_dir: the directory of the YAML file.

    Returns:
        config_dict: the keys and corresponding values in the YAML file.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_config_file_dir = os.path.join(current_dir, "../algo_config/base_config.yaml")
    
    with open(base_config_file_dir, "r") as f:
        try:
            base_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, base_config_file_dir + " error: {}".format(exc)

    # file_dir is passed as relative path like "../algo_config/gnn_pd.yaml"
    # We assume it is relative to this file (mhgnn_agent.py)
    config_file_path = os.path.join(current_dir, file_dir)
    
    with open(config_file_path, "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
            # Let the experiment-specific config override the base config.
            base_config_dict.update(config_dict)
        except yaml.YAMLError as exc:
            assert False, config_file_path + " error: {}".format(exc)
    return base_config_dict

def save_configs(configs_dict, save_path):
    """Save dict variable to a YAML file.
    Args:
        configs_dict: the dict variable to be saved.
        save_path: the directory to save the YAML file.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_dir = os.path.join(save_path, "configs_used.yaml")

    with open(file_dir, "w") as f:
        try:
            yaml.dump(configs_dict, f)
        except yaml.YAMLError as exc:
            assert False, save_path + " error: {}".format(exc)


class BaseAgent(ABC):
    model_name = None

    def __init__(self, config: Namespace):
        self.config = config
        self.outputPath = config.outputPath
        seed = f"seed_{self.config.seed}"
        self.model_dir_save = config.model_dir + seed
        time_string = get_time_string()
        self.model_dir_save = os.path.join(self.outputPath, self.model_dir_save)
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Prepare necessary components.
        self.policy = None
        self.learner = None
        self.memory = ExperienceReplay(config.buffer_size)

        if config.train_TA_model:
            # Create logger.
            if config.logger == "tensorboard":
                log_dir = os.path.join(self.outputPath, config.log_dir + seed)
                if not os.path.exists(log_dir):
                    create_directory(log_dir)

                self.writer = SummaryWriter(log_dir)
                self.use_wandb = False
            elif config.logger == "wandb":
                config_dict = vars(config)
                log_dir = os.path.join(self.outputPath, config.log_dir + seed)
                wandb_dir = Path(os.path.join(self.outputPath, config.log_dir + seed))
                if not os.path.exists(wandb_dir):
                    create_directory(str(wandb_dir))

                try:
                    wandb.init(
                        config=config_dict,
                        project=config.project_name,
                        entity=config.wandb_user_name,
                        notes=socket.gethostname(),
                        dir=wandb_dir,
                        job_type=config.agent,
                        name=time_string,
                        reinit=True,
                        settings=wandb.Settings(start_method="fork"),
                    )
                    self.use_wandb = True
                except Exception as e:
                    # Keep the simulator running even if remote logging is unavailable.
                    self.use_wandb = False
                    print(
                        f"[WARN] swanlab/wandb init failed ({type(e).__name__}: {e}); continuing without remote logging.",
                        file=sys.stderr,
                    )
            else:
                raise AttributeError("No logger is implemented.")
        else:
            log_dir = None
            self.use_wandb = False
        self.log_dir = log_dir
        self.w1 = getattr(config, 'w1', 20)
        self.w2 = getattr(config, 'w2', 20)
        self.w4 = getattr(config, 'w4', 5)

    def _safe_wandb_log(self, payload: dict, step: int | None = None) -> None:
        """Log to swanlab/wandb but never crash the simulation if logging fails."""
        try:
            if step is None:
                wandb.log(payload)
            else:
                wandb.log(payload, step=step)
        except Exception as e:
            # If remote logging fails (network/API bugs), disable it to keep simulation running.
            self.use_wandb = False
            print(f"[WARN] swanlab/wandb logging failed ({type(e).__name__}: {e}); disabling remote logging.", file=sys.stderr)

    def _build_policy(self) -> Module:
        raise NotImplementedError

    def _build_learner(self, *args):
        return NotImplementedError

    def _get_q_values_for_action(self, new_state):
        self.policy.qNetwork.eval()
        self.policy.qNetwork.g = new_state
        return self.policy.qNetwork(new_state.ndata['feat'])

    def getNextHop(self, newState, linkedSats, sat, earth):
        unavPenalty = -10
        newState = newState.to(self.device)

        if self.train_TA_model and random.uniform(0, 1) < self.alignEpsilon(self.step, sat):
            actIndex = random.randrange(self.actionSize)
            action = self.actions[actIndex]
            while linkedSats[action] is None:
                self.memory.store(newState, actIndex, unavPenalty, newState, False)
                earth.rewards.append([unavPenalty, sat.env.now])
                actIndex = random.randrange(self.actionSize)
                action = self.actions[actIndex]
        else:
            with torch.no_grad():
                qValues = self._get_q_values_for_action(newState)
            qValues = qValues.cpu().numpy().flatten()
            actIndex = np.argmax(qValues)
            action = self.actions[actIndex]

            while linkedSats[action] is None:
                self.memory.store(newState, actIndex, unavPenalty, newState, False)
                earth.rewards.append([unavPenalty, sat.env.now])
                qValues[actIndex] = -np.inf
                actIndex = np.argmax(qValues)
                action = self.actions[actIndex]

        destination = linkedSats[action]
        if destination is None:
            return -1
        return [destination.ID, math.degrees(destination.longitude), math.degrees(destination.latitude)], actIndex

    def train(self, sat, earth):
        if self.memory.buffeSize < self.config.batch_size:
            return
        for _ in range(self.train_epoch):
            samples = self.memory.getBatch(self.config.batch_size)
            info = self.learner.update(samples, self.step)
            self.log_infos_no_index(info)

        earth.loss.append([info.get('rl_loss', 0.0), sat.env.now])
        earth.trains.append([sat.env.now])

    # def test(self, env_fn, steps):
    #     raise NotImplementedError
    
    def makeDeepAction(self, block, sat, g, earth, prevSat=None, *args):
        recalculate_flag = args and args[0] == 'recalculate'
        training_mode = self.train_TA_model and not recalculate_flag

        is_reached = sat.linkedGT and block.destination.ID == sat.linkedGT.ID
        is_failure = len(block.QPath) > system_configure.Max_Hops and not is_reached

        if is_reached or is_failure:
            if training_mode and prevSat is not None:
                new_state_g_dgl = get_subgraph_state(block, sat, g, earth, n_order=self.n_order_adj)
                self.step += 1
                reward = self._calculate_reward_v1(block, sat, prevSat, is_terminal=is_reached, is_failure=is_failure)
                self.store_experience(block, reward, new_state_g_dgl, True, sat, earth)
                self.log_infos_no_index({"Reward": sum(block.stepReward) if block.stepReward else reward})
            return -1 if is_failure else 0

        linkedSats = getDeepLinkedSats(sat, g, earth)
        new_state_g_dgl = get_subgraph_state(block, sat, g, earth, n_order=self.n_order_adj)
        self.step += 1

        nextHop, actIndex = self.getNextHop(new_state_g_dgl, linkedSats, sat, earth)

        if nextHop == -1:
            if not training_mode:
                print(f"Error in nextHop calculation: Sat {sat.ID}, block {block}")
            return -2

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

    def save_model(self, model_name=None):
        model_name = model_name or self.model_name
        if model_name is None:
            raise ValueError("model_name is not set for this agent.")
        if not os.path.exists(self.model_dir_save):
            os.makedirs(self.model_dir_save)
        qNet_model_path = os.path.join(self.model_dir_save, 'qNet_' + model_name)
        qTarget_model_path = os.path.join(self.model_dir_save, 'qTarget_' + model_name)
        sNet_model_path = os.path.join(self.model_dir_save, 'sNet_' + model_name)
        torch.save(self.policy.qNetwork.state_dict(), qNet_model_path)
        torch.save(self.policy.qTarget.state_dict(), qTarget_model_path)
        torch.save(self.policy.sNetwork.state_dict(), sNet_model_path)

    def _resolve_model_dir(self):
        model_dir = self.model_dir_save
        if 'test_teacher_network' in model_dir:
            model_dir = re.sub(r'test_teacher_network[^/\\]*', 'train', model_dir)
        elif 'test_student_network' in model_dir:
            model_dir = re.sub(r'test_student_network[^/\\]*', 'train', model_dir)

        if not os.path.isabs(model_dir):
            model_dir = os.path.join(self.outputPath, '../train/', model_dir)
        return model_dir

    def _load_snetwork_state(self, sNet_model_path):
        self.policy.sNetwork.load_state_dict(torch.load(sNet_model_path, map_location=self.device, weights_only=True))

    def load_model(self, model_name=None):
        model_name = model_name or self.model_name
        if model_name is None:
            raise ValueError("model_name is not set for this agent.")

        model_dir = self._resolve_model_dir()
        qNet_model_path = os.path.join(model_dir, 'qNet_' + model_name)
        qTarget_model_path = os.path.join(model_dir, 'qTarget_' + model_name)
        sNet_model_path = os.path.join(model_dir, 'sNet_' + model_name)

        print("Loading model from:", qNet_model_path)

        self.policy.qNetwork.load_state_dict(torch.load(qNet_model_path, map_location=self.device, weights_only=True))
        self.policy.qTarget.load_state_dict(torch.load(qTarget_model_path, map_location=self.device, weights_only=True))
        self._load_snetwork_state(sNet_model_path)

    def try_save_model(self):
        if self.train_TA_model:
            self.save_model()

    def _calculate_reward_v1(self, block, sat, prevSat, is_terminal=False, is_failure=False):
        w1 = self.w1
        w2 = self.w2
        w4 = self.w4
        ArriveReward = 50
        againPenalty = -10

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
        epsilon = minEps + (maxEps - minEps) * math.exp(-LAMBDA * step / (decayRate * (2**2)))
        self.epsilon.append([epsilon, sat.env.now])
        return epsilon

    def log_infos(self, info: dict, x_index: int):
        """
        info: (dict) information to be visualized
        n_steps: current step
        """
        if self.use_wandb:
            for k, v in info.items():
                if v is None:
                    continue
                self._safe_wandb_log({k: v}, step=x_index)
        else:
            if not hasattr(self, 'writer') or self.writer is None:
                return
            for k, v in info.items():
                if v is None:
                    continue
                try:
                    self.writer.add_scalar(k, v, x_index)
                except:
                    self.writer.add_scalars(k, v, x_index)

    def log_infos_no_index(self, info: dict):
        """
        info: (dict) information to be visualized
        """
        if self.use_wandb:
            for k, v in info.items():
                if v is None:
                    continue
                self._safe_wandb_log({k: v})
        else:
            if not hasattr(self, 'writer') or self.writer is None:
                return
            for k, v in info.items():
                if v is None:
                    continue
                try:
                    self.writer.add_scalar(k, v)
                except:
                    self.writer.add_scalars(k, v)