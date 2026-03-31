from abc import ABC, abstractmethod
from modulefinder import Module
import os
import numpy as np
import torch
import random
import math
import re
import networkx as nx
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
from Utils.utilsfunction import getQueueReward, getDistanceRewardV4, getQueues, getSlantRange


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

        self.reward_mode = getattr(config, 'reward_mode', 'layer1').lower()
        self.arrive_reward = float(getattr(config, 'arrive_reward', 50.0))
        self.failure_penalty = float(getattr(config, 'failure_penalty', 50.0))
        self.loop_penalty = float(getattr(config, 'loop_penalty', 10.0))
        self.reward_distance_scale = float(getattr(config, 'reward_distance_scale', 4.0))
        self.reward_distance_ref = float(getattr(config, 'reward_distance_ref', 1.0))
        self.reward_queue_scale = float(getattr(config, 'reward_queue_scale', 8.0))
        self.reward_queue_ref = float(getattr(config, 'reward_queue_ref', 0.003))
        self.reward_hop_penalty = float(getattr(config, 'reward_hop_penalty', 0.3))
        self.reward_prop_scale = float(getattr(config, 'reward_prop_scale', 1.0))
        self.reward_prop_ref = float(getattr(config, 'reward_prop_ref', 0.01))
        self.reward_tx_scale = float(getattr(config, 'reward_tx_scale', 1.0))
        self.reward_tx_ref = float(getattr(config, 'reward_tx_ref', 0.001))
        self.reward_delay_beta = float(getattr(config, 'reward_delay_beta', 3.0))
        self.reward_delay_ref = float(getattr(config, 'reward_delay_ref', 1.0))
        self.reward_remaining_queue_scale = float(getattr(config, 'reward_remaining_queue_scale', 0.5))
        self.reward_remaining_hop_cost = float(getattr(config, 'reward_remaining_hop_cost', 0.2))
        self.reward_local_congestion_scale = float(getattr(config, 'reward_local_congestion_scale', 2.0))
        self.reward_local_congestion_ref = float(getattr(config, 'reward_local_congestion_ref', 0.01))
        self.reward_min_rate = max(float(getattr(config, 'reward_min_rate', 1e6)), 1.0)
        self.reward_cache_time_precision = int(getattr(config, 'reward_cache_time_precision', 6))
        self.reward_log_enabled = bool(getattr(config, 'reward_log_enabled', False))
        self.reward_log_interval = max(int(getattr(config, 'reward_log_interval', 100)), 1)
        self._delay_to_go_cache = {}
        self._sat_lookup_cache = {}

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
        if prevSat is None:
            raise AssertionError("Previous satellite is None in reward calculation.")

        satDest = block.destination.linkedSat[1]
        if satDest is None:
            print("No linked sat for destination GT")
            return self._finalize_reward(0.0, is_terminal=is_terminal, is_failure=is_failure)

        mode = self.reward_mode
        if mode == 'legacy':
            reward, reward_info = self._calculate_reward_legacy(block, sat, prevSat, satDest)
        elif mode == 'layer1':
            reward, reward_info = self._calculate_reward_layer1(block, sat, prevSat, satDest)
        elif mode == 'layer2':
            reward, reward_info = self._calculate_reward_layer2(block, sat, prevSat, satDest)
        elif mode == 'layer3':
            reward, reward_info = self._calculate_reward_layer3(block, sat, prevSat, satDest)
        else:
            print(f"[WARN] Unknown reward_mode '{mode}', fallback to layer1.")
            reward, reward_info = self._calculate_reward_layer1(block, sat, prevSat, satDest)

        final_reward = self._finalize_reward(reward, is_terminal=is_terminal, is_failure=is_failure)
        self._maybe_log_reward_breakdown(
            sat,
            mode,
            reward_info,
            reward,
            final_reward,
            is_terminal=is_terminal,
            is_failure=is_failure,
        )
        return final_reward

    def _calculate_reward_legacy(self, block, sat, prevSat, satDest):
        queue_reward = 0.0
        if block.queueTime:
            queue_reward = getQueueReward(block.queueTime[-1], self.w1)
        distance_reward = getDistanceRewardV4(prevSat, sat, satDest, self.w2, self.w4)
        loop_reward = -self.loop_penalty if self._is_revisited_sat(block, sat) else 0.0
        reward = distance_reward + queue_reward + loop_reward
        return reward, {
            "distance_reward": distance_reward,
            "queue_reward": queue_reward,
            "loop_penalty": loop_reward,
        }

    def _calculate_reward_layer1(self, block, sat, prevSat, satDest):
        queue_time = self._get_last_queue_time(block)
        raw_distance = getDistanceRewardV4(prevSat, sat, satDest, self.w2, self.w4)
        distance_reward = self.reward_distance_scale * math.tanh(raw_distance / max(self.reward_distance_ref, 1e-6))
        queue_penalty = -self.reward_queue_scale * self._log_normalize(queue_time, self.reward_queue_ref)
        reward = distance_reward + queue_penalty - self.reward_hop_penalty
        loop_penalty = 0.0
        if self._is_revisited_sat(block, sat):
            loop_penalty = -self.loop_penalty
            reward += loop_penalty
        return reward, {
            "queue_time": queue_time,
            "raw_distance_reward": raw_distance,
            "distance_reward": distance_reward,
            "queue_penalty": queue_penalty,
            "hop_penalty": -self.reward_hop_penalty,
            "loop_penalty": loop_penalty,
        }

    def _calculate_reward_layer2(self, block, sat, prevSat, satDest):
        graph = self._get_runtime_graph(sat)
        queue_time, prop_delay, tx_delay = self._get_immediate_delays(block, prevSat, sat, graph)

        immediate_cost = (
            self.reward_queue_scale * self._log_normalize(queue_time, self.reward_queue_ref)
            + self.reward_prop_scale * self._normalize_linear(prop_delay, self.reward_prop_ref)
            + self.reward_tx_scale * self._normalize_linear(tx_delay, self.reward_tx_ref)
            + self.reward_hop_penalty
        )

        reward = -immediate_cost
        shaping_reward, shaping_info = self._get_delay_to_go_shaping(block, prevSat, sat, satDest, graph)
        reward += shaping_reward
        loop_penalty = 0.0
        if self._is_revisited_sat(block, sat):
            loop_penalty = -self.loop_penalty
            reward += loop_penalty
        return reward, {
            "queue_time": queue_time,
            "prop_delay": prop_delay,
            "tx_delay": tx_delay,
            "immediate_cost": immediate_cost,
            "queue_penalty": -self.reward_queue_scale * self._log_normalize(queue_time, self.reward_queue_ref),
            "prop_penalty": -self.reward_prop_scale * self._normalize_linear(prop_delay, self.reward_prop_ref),
            "tx_penalty": -self.reward_tx_scale * self._normalize_linear(tx_delay, self.reward_tx_ref),
            "hop_penalty": -self.reward_hop_penalty,
            "delay_shaping": shaping_reward,
            "loop_penalty": loop_penalty,
            **shaping_info,
        }

    def _calculate_reward_layer3(self, block, sat, prevSat, satDest):
        graph = self._get_runtime_graph(sat)
        reward, reward_info = self._calculate_reward_layer2(block, sat, prevSat, satDest)
        local_congestion_delay = self._estimate_local_congestion_delay(sat, block, graph)
        local_penalty = self.reward_local_congestion_scale * math.tanh(
            local_congestion_delay / max(self.reward_local_congestion_ref, 1e-6)
        )
        reward -= local_penalty
        reward_info.update(
            {
                "local_congestion_delay": local_congestion_delay,
                "local_congestion_penalty": -local_penalty,
            }
        )
        return reward, reward_info

    def _finalize_reward(self, reward, is_terminal=False, is_failure=False):
        if is_failure:
            return reward - self.failure_penalty
        if is_terminal:
            return reward + self.arrive_reward
        return reward

    def _get_last_queue_time(self, block):
        if getattr(block, 'queueTime', None):
            return max(float(block.queueTime[-1]), 0.0)
        return 0.0

    def _normalize_linear(self, value, ref):
        return max(float(value), 0.0) / max(float(ref), 1e-9)

    def _log_normalize(self, value, ref):
        return math.log1p(max(float(value), 0.0) / max(float(ref), 1e-9))

    def _is_revisited_sat(self, block, sat):
        history = block.QPath[:max(len(block.QPath) - 2, 0)]
        for hop in history:
            hop_id = hop[0] if isinstance(hop, (list, tuple)) else hop
            if hop_id == sat.ID:
                return True
        return False

    def _get_runtime_graph(self, sat):
        earth = getattr(getattr(sat, 'orbPlane', None), 'earth', None)
        if earth is None:
            return None
        if getattr(earth, 'graph', None) is not None:
            return earth.graph
        gateways = getattr(earth, 'gateways', None)
        if gateways:
            return getattr(gateways[0], 'graph', None)
        return None

    def _get_earth(self, sat):
        return getattr(getattr(sat, 'orbPlane', None), 'earth', None)

    def _get_sat_lookup(self, sat):
        earth = self._get_earth(sat)
        if earth is None:
            return {}

        cache_key = id(earth)
        cached = self._sat_lookup_cache.get(cache_key)
        if cached is not None:
            return cached

        sat_lookup = {}
        for plane in earth.LEO:
            for sat_node in plane.sats:
                sat_lookup[sat_node.ID] = sat_node

        self._sat_lookup_cache = {cache_key: sat_lookup}
        return sat_lookup

    def _get_link_metrics(self, prevSat, sat, graph=None):
        slant_range = getSlantRange(prevSat, sat)
        data_rate = None

        if graph is not None and graph.has_edge(prevSat.ID, sat.ID):
            edge_data = graph.edges[prevSat.ID, sat.ID]
            slant_range = edge_data.get('slant_range', slant_range)
            data_rate = edge_data.get('dataRateOG')

        if data_rate is None or not np.isfinite(data_rate) or data_rate <= 0:
            data_rate = self.reward_min_rate

        return max(float(slant_range), 0.0), max(float(data_rate), self.reward_min_rate)

    def _get_immediate_delays(self, block, prevSat, sat, graph=None):
        queue_time = self._get_last_queue_time(block)
        slant_range, data_rate = self._get_link_metrics(prevSat, sat, graph)
        prop_delay = slant_range / system_configure.Vc
        tx_delay = block.size / data_rate
        return queue_time, prop_delay, tx_delay

    def _iter_neighbor_metrics(self, sat, graph=None):
        queues = getQueues(sat, DDQN=True)
        directions = (
            ('U', getattr(sat, 'upper', None)),
            ('D', getattr(sat, 'lower', None)),
            ('R', getattr(sat, 'right', None)),
            ('L', getattr(sat, 'left', None)),
        )

        metrics = []
        for direction, neighbor in directions:
            if neighbor is None:
                continue

            queue_len = queues.get(direction, np.inf)
            if not np.isfinite(queue_len):
                continue

            slant_range = getSlantRange(sat, neighbor)
            data_rate = self.reward_min_rate
            if graph is not None and graph.has_edge(sat.ID, neighbor.ID):
                edge_data = graph.edges[sat.ID, neighbor.ID]
                slant_range = edge_data.get('slant_range', slant_range)
                data_rate = edge_data.get('dataRateOG', data_rate)

            if not np.isfinite(data_rate) or data_rate <= 0:
                data_rate = self.reward_min_rate

            metrics.append(
                (
                    direction,
                    max(float(queue_len), 0.0),
                    max(float(slant_range), 0.0),
                    max(float(data_rate), self.reward_min_rate),
                )
            )

        return metrics

    def _estimate_best_egress_delay(self, sat, block, graph=None):
        candidates = []
        for _, queue_len, slant_range, data_rate in self._iter_neighbor_metrics(sat, graph):
            queue_wait = queue_len * block.size / data_rate
            prop_delay = slant_range / system_configure.Vc
            candidates.append(queue_wait + prop_delay)

        if not candidates:
            return self.reward_local_congestion_ref * 2.0
        return min(candidates)

    def _estimate_local_congestion_delay(self, sat, block, graph=None):
        candidates = []
        for _, queue_len, slant_range, data_rate in self._iter_neighbor_metrics(sat, graph):
            queue_wait = queue_len * block.size / data_rate
            prop_delay = slant_range / system_configure.Vc
            candidates.append(queue_wait + prop_delay)

        if not candidates:
            return self.reward_local_congestion_ref * 2.0

        candidates.sort()
        return float(np.mean(candidates[: min(2, len(candidates))]))

    def _get_delay_to_go_cache_key(self, graph, satDest, block_size, env_now):
        return (
            id(graph),
            satDest.ID,
            int(block_size),
            round(float(env_now), self.reward_cache_time_precision),
        )

    def _get_delay_to_go(self, block, sat_from, satDest, graph):
        if graph is None or satDest is None:
            return None
        if sat_from.ID == satDest.ID:
            return 0.0
        if satDest.ID not in graph:
            return None

        cache_key = self._get_delay_to_go_cache_key(graph, satDest, block.size, sat_from.env.now)
        lengths = self._delay_to_go_cache.get(cache_key)
        if lengths is None:
            sat_lookup = self._get_sat_lookup(sat_from)
            node_delay = {}
            for sat_id, sat_node in sat_lookup.items():
                if sat_id == satDest.ID:
                    node_delay[sat_id] = 0.0
                else:
                    best_delay = self._estimate_best_egress_delay(sat_node, block, graph)
                    node_delay[sat_id] = self.reward_remaining_queue_scale * self._normalize_linear(
                        best_delay,
                        self.reward_queue_ref,
                    )

            dijkstra_graph = graph.reverse(copy=False) if hasattr(graph, 'reverse') else graph

            def weight(src_id, dst_id, data):
                prop_delay = max(float(data.get('slant_range', 0.0)), 0.0) / system_configure.Vc
                data_rate = data.get('dataRateOG', self.reward_min_rate)
                if not np.isfinite(data_rate) or data_rate <= 0:
                    data_rate = self.reward_min_rate
                tx_delay = block.size / max(float(data_rate), self.reward_min_rate)
                return (
                    self.reward_prop_scale * self._normalize_linear(prop_delay, self.reward_prop_ref)
                    + self.reward_tx_scale * self._normalize_linear(tx_delay, self.reward_tx_ref)
                    + self.reward_remaining_hop_cost
                    + node_delay.get(dst_id, 0.0)
                )

            lengths = nx.single_source_dijkstra_path_length(dijkstra_graph, satDest.ID, weight=weight)
            self._delay_to_go_cache = {cache_key: lengths}

        return lengths.get(sat_from.ID)

    def _get_delay_to_go_shaping(self, block, prevSat, sat, satDest, graph):
        prev_delay = self._get_delay_to_go(block, prevSat, satDest, graph)
        cur_delay = self._get_delay_to_go(block, sat, satDest, graph)

        if prev_delay is None or cur_delay is None:
            fallback_distance = getDistanceRewardV4(prevSat, sat, satDest, self.w2, self.w4)
            fallback_reward = 0.5 * self.reward_distance_scale * math.tanh(
                fallback_distance / max(self.reward_distance_ref, 1e-6)
            )
            return fallback_reward, {
                "delay_to_go_prev": float("nan"),
                "delay_to_go_cur": float("nan"),
                "delay_to_go_improvement": float("nan"),
                "delay_shaping_fallback": fallback_reward,
            }

        improvement = prev_delay - cur_delay
        shaping_reward = self.reward_delay_beta * math.tanh(improvement / max(self.reward_delay_ref, 1e-6))
        return shaping_reward, {
            "delay_to_go_prev": prev_delay,
            "delay_to_go_cur": cur_delay,
            "delay_to_go_improvement": improvement,
            "delay_shaping_fallback": 0.0,
        }

    def _maybe_log_reward_breakdown(
        self,
        sat,
        mode,
        reward_info,
        base_reward,
        final_reward,
        is_terminal=False,
        is_failure=False,
    ):
        if not self.reward_log_enabled:
            return
        if self.step % self.reward_log_interval != 0:
            return

        payload = {
            "reward/mode_id": {
                "legacy": 0.0,
                "layer1": 1.0,
                "layer2": 2.0,
                "layer3": 3.0,
            }.get(mode, -1.0),
            "reward/base": base_reward,
            "reward/final": final_reward,
            "reward/is_terminal": float(bool(is_terminal)),
            "reward/is_failure": float(bool(is_failure)),
        }
        for key, value in reward_info.items():
            if value is None:
                continue
            if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                payload[f"reward/{key}"] = float(value)

        self.log_infos(payload, self.step)

    def alignEpsilon(self, step, sat):
        maxEps = self.config.MAX_EPSILON
        minEps = self.config.MIN_EPSILON
        decayRate = self.config.decayRate
        LAMBDA = self.config.LAMBDA
        power = getattr(self.config, 'power', 2)
        epsilon = minEps + (maxEps - minEps) * math.exp(-LAMBDA * step / (decayRate * (power**2)))
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
