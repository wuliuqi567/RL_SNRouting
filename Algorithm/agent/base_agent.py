from abc import ABC, abstractmethod
from modulefinder import Module
import os
import numpy as np
import torch
from ..common.common_tools import get_time_string, create_directory
from ..common.experienceReplay import ExperienceReplay
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import swanlab as wandb
import socket
from argparse import Namespace
import yaml
import sys

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
            config_dict.update(base_config_dict)
        except yaml.YAMLError as exc:
            assert False, config_file_path + " error: {}".format(exc)
    return config_dict

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

    def getNextHop(self, observations):
        raise NotImplementedError

    def train(self, steps):
        raise NotImplementedError

    # def test(self, env_fn, steps):
    #     raise NotImplementedError
    
    @abstractmethod
    def makeDeepAction(self, block, sat, *args):
        raise NotImplementedError("Subclasses must implement this method")
    
    # @abstractmethod
    def store_experience(self, *args, **kwargs):
        raise NotImplementedError

    def save_model(self, model_name):
        # save the neural networks
        if not os.path.exists(self.model_dir_save):
            os.makedirs(self.model_dir_save)
        model_path = os.path.join(self.model_dir_save, model_name)
        torch.save(self.policy.state_dict(), model_path)

    def load_model(self, model_name):
        # load neural networks
        model_path = os.path.join(self.model_dir_save, model_name)
        self.policy.load_state_dict(torch.load(model_path, map_location=self.device))

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