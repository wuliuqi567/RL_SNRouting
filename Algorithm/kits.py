import swanlab
import time
import socket

# Initialize SwanLab
def init_swanlab(hyperparams=None, project_name="Satellite_Routingv5", 
                     entity=None, log_dir="./logs"):
        """
        初始化SwanLab日志记录
        
        Args:
            config: 配置字典
            project_name: 项目名称
            entity: 用户/组织名称
            log_dir: 日志目录
        """
        config = vars(hyperparams) # 将类转换为字典
        try:            
            # 创建日志目录
            log_path = config['outputPath']
            algorithm_name = config['pathing']
            time_string = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            experiment_name = f"{algorithm_name}_{time_string}"

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

def log_metrics(metrics_dict, step=None):
    """
    记录指标到SwanLab

    Args:
        metrics_dict: 指标字典
        step: 训练步数
    """    """记录指标到SwanLab
    Args:
        metrics_dict: 指标字典
        step: 训练步数
    """

    try:
        if step is not None:
            metrics_dict['step'] = step
        swanlab.log(metrics_dict)
    except Exception as e:
        print(f"Failed to log metrics: {e}")

def log_reward(reward):
    """记录奖励"""
    log_metrics({
        "reward": reward
    })
