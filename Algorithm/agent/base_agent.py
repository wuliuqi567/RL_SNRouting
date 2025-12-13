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

class BaseAgent(ABC):
    def __init__(self, config: Namespace):
        self.config = config

        seed = f"seed_{self.config.seed}_"
        self.model_dir_load = config.model_dir
        time_string = get_time_string()
        self.model_dir_save = os.path.join(os.getcwd(), config.model_dir, seed + time_string)
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Prepare necessary components.
        self.policy = None
        self.learner = None
        self.memory = ExperienceReplay(config.memory_size)

        # Create logger.
        if config.logger == "tensorboard":
            log_dir = os.path.join(os.getcwd(), config.log_dir, seed + time_string)
            if not os.path.exists(log_dir):
                create_directory(log_dir)

            self.writer = SummaryWriter(log_dir)
            self.use_wandb = False
        elif config.logger == "wandb":
            config_dict = vars(config)
            log_dir = config.log_dir
            wandb_dir = Path(os.path.join(os.getcwd(), config.log_dir))
            if not os.path.exists(wandb_dir):
                create_directory(str(wandb_dir))

            wandb.init(config=config_dict,
                       project=config.project_name,
                       entity=config.wandb_user_name,
                       notes=socket.gethostname(),
                       dir=wandb_dir,
                       job_type=config.agent,
                       name=time_string,
                       reinit=True,
                       settings=wandb.Settings(start_method="fork")
                       )
            self.use_wandb = True
        else:
            raise AttributeError("No logger is implemented.")
        self.log_dir = log_dir

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
    # def getNextHop(self, block, sat, *args):
    #     raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def store_experience(self, *args, **kwargs):
        raise NotImplementedError

    def save_model(self, model_name):
        # save the neural networks
        if not os.path.exists(self.model_dir_save):
            os.makedirs(self.model_dir_save)
        model_path = os.path.join(self.model_dir_save, model_name)
        self.learner.save_model(model_path)
        # save the observation status
        if self.use_obsnorm:
            obs_norm_path = os.path.join(self.model_dir_save, "obs_rms.npy")
            observation_stat = {'count': self.obs_rms.count,
                                'mean': self.obs_rms.mean,
                                'var': self.obs_rms.var}
            np.save(obs_norm_path, observation_stat)

    def load_model(self, path, model=None):
        # load neural networks
        path_loaded = self.learner.load_model(path, model)
        # recover observation status
        if self.use_obsnorm:
            obs_norm_path = os.path.join(path_loaded, "obs_rms.npy")
            if os.path.exists(obs_norm_path):
                observation_stat = np.load(obs_norm_path, allow_pickle=True).item()
                self.obs_rms.count = observation_stat['count']
                self.obs_rms.mean = observation_stat['mean']
                self.obs_rms.var = observation_stat['var']
            else:
                raise RuntimeError(f"Failed to load observation status file 'obs_rms.npy' from {obs_norm_path}!")

    def log_infos(self, info: dict, x_index: int):
        """
        info: (dict) information to be visualized
        n_steps: current step
        """
        if self.use_wandb:
            for k, v in info.items():
                if v is None:
                    continue
                wandb.log({k: v}, step=x_index)
        else:
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
                wandb.log({k: v})
        else:
            for k, v in info.items():
                if v is None:
                    continue
                try:
                    self.writer.add_scalar(k, v)
                except:
                    self.writer.add_scalars(k, v)