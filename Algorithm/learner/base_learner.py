import os
import torch
import numpy as np
from abc import ABC, abstractmethod
# from xuance.common import Optional, List, Union
from argparse import Namespace
from operator import itemgetter
# from xuance.torch import Tensor
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Union, List, Dict, Sequence, Callable, Any, Tuple, SupportsFloat, Type, Mapping
from ..common.pd_loss_fun import kl_distillation_loss, kl_distillation_loss_v2

class BaseLearner(ABC):
    def __init__(self, config: Namespace, policy: torch.nn.Module):
        self.config = config
        self.policy = policy
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # 优化器和损失函数
        self.optimizer = Union[dict, list, Optional[torch.optim.Optimizer]] = None
        self.scheduler = Union[dict, list, Optional[torch.optim.lr_scheduler.LinearLR]] = None
        self.student_optimizer = Union[dict, list, Optional[torch.optim.Optimizer]] = None
        self.student_scheduler = Union[dict, list, Optional[torch.optim.lr_scheduler.LinearLR]] = None

        self.loss_fn = nn.SmoothL1Loss()  # Huber损失对应PyTorch的SmoothL1Loss

    def save_model(self, model_path):
        """
        保存模型参数到指定路径。

        参数:
            model_path (str): 模型保存的完整路径（包括文件名）。
        
        示例:
            learner.save_model("./models/dqn_agent.pth")
        """
        torch.save(self.policy.state_dict(), model_path)

    def load_model(self, path, model=None):
        """
        从指定路径加载模型参数。

        参数:
            path (str): 包含模型文件夹的根目录路径。
            model (str, optional): 指定要加载的具体模型文件夹名称。
                                   如果为 None，则自动查找 path 下最新的以 'seed_' 开头的文件夹。
        
        返回:
            str: 加载的模型文件的完整路径。

        异常:
            RuntimeError: 如果路径不存在或未找到模型文件。

        示例:
            # 自动加载最新的 seed 文件夹中的最新模型
            learner.load_model("./results/DQN")
            
            # 加载指定文件夹中的最新模型
            learner.load_model("./results/DQN", model="seed_42")
        """
        if not os.path.exists(path):
            raise RuntimeError(f"The path '{path}' does not exist.")

        if model is not None:
            model_dir = os.path.join(path, model)
            if not os.path.exists(model_dir):
                raise RuntimeError(f"The folder '{model_dir}' does not exist, please specify a correct path to load model.")
        else:
            # Use list comprehension to safely filter 'seed_' folders
            seed_folders = [f for f in os.listdir(path) if "seed_" in f]
            if not seed_folders:
                raise RuntimeError(f"No 'seed_' folders found in '{path}'.")
            
            seed_folders.sort()
            model_dir = os.path.join(path, seed_folders[-1])

        if not os.path.exists(model_dir):
             raise RuntimeError(f"Model directory '{model_dir}' does not exist.")

        model_files = os.listdir(model_dir)
        if not model_files:
            raise RuntimeError(f"There is no model file in '{model_dir}'!")
        
        model_files.sort()
        model_path = os.path.join(model_dir, model_files[-1])
        
        # Simplify map_location to directly use self.device
        self.policy.load_state_dict(torch.load(str(model_path), map_location=self.device))
        print(f"Successfully loaded model from '{model_path}'.")
        return model_path
    
    @abstractmethod
    def update(self, *args):
        raise NotImplementedError