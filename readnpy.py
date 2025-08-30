import numpy as np
import os
from Class.auxiliaryClass import BlocksForPickle

current_directory = './Results/Congestion_test/dataRate 5s'

try:
    # 读取 .npy 文件
    data = np.load(current_directory+'/blocks_2.npy', allow_pickle=True)
    
    # 打印数据的基本信息
    print(f"数据类型: {type(data)}")
    print(f"数据形状: {data.shape}")
    print(f"数据维度: {data.ndim}")
    print(f"数据元素类型: {data.dtype}")
    
    # 打印数据前 10 个元素（适用于一维数组）
    if data.ndim == 1:
        print(f"数据前 10 个元素: {data[:10]}")
    else:
        print(f"数据前几行:\n{data[:3]}")  # 打印前几行

except FileNotFoundError:
    print(f"错误：找不到文件 'example.npy'")
except Exception as e:
    print(f"错误：读取文件时发生异常: {e}")