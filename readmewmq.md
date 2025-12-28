
# install guide
- pytorch 2.4.0, diff cuda version
```python 
# ROCM 6.1 (Linux only)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/rocm6.1
# CUDA 11.8
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
# CPU only
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu

```

- dgl GNN library install 
```python
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html

pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu118/repo.html
```


# 一 state space

每个节点的状态定义：

### 1. 几何位置特征 (Geometric Features)

* **绝对位置 (归一化):**  $P_i / R_{earth}$。让网络知道该节点在地球的哪个区域（极地/赤道）。
* **相对于当前节点的位移:**​$ \Delta P_{i, curr} = P_i - P_{current\_node}。$

  * 这告诉网络：节点 i 在当前节点的什么方向。
* **相对于目标（目的地面站连接的卫星）的位移:**  $\Delta P_{i, target} = P_i - P_{target\_node}$。

  * 这告诉网络：节点 i 离终点还有多远。

### 2. 队列信息：上下左右四个队列的长度

### 3. 基于当前节点寻找其n阶邻居节点并建立dgl图

# 二 奖励函数

### 方法1. 基于跳数等属性

### 方法2. 基于李雅普诺夫

# 参数整理

分为
- 系统参数，
- RL模型参数
- 超参数，控制训练，执行



# 增加 swap空间大小
假设我们要**额外新增一个 4GB** 的文件，并将其命名为 `/swapfile2`（以此类推，避免和原有的 `/swapfile` 冲突）。

以下是具体步骤：

### 1. 检查磁盘空间

首先确保您的硬盘有足够的空间来存放新文件：

```bash
df -h /

```

### 2. 创建新的 Swap 文件

使用 `fallocate` 创建名为 `/swapfile2` 的文件（这里以 **4G** 为例，您可以改为 8G、16G 等）：

```bash
sudo fallocate -l 4G /swapfile2

```

### 3. 设置权限

出于安全原因，设置只有 root 用户可以访问该文件：

```bash
sudo chmod 600 /swapfile2

```

### 4. 格式化为 Swap

将新文件标记为交换空间：

```bash
sudo mkswap /swapfile2

```

### 5. 启用新的 Swap

启用它，系统现在的 Swap 总量将是“原有大小 + 新增大小”：

```bash
sudo swapon /swapfile2

```

### 6. 验证结果

查看现在的 Swap 状态，您应该能看到列表中有两个文件：

```bash
sudo swapon --show

```

或者查看总量：

```bash
free -h

```

### 7. 设置永久生效 (写入 fstab)

为了防止重启后新增的 Swap 失效，需要将其添加到 `/etc/fstab` 文件中。

运行以下命令将配置追加到文件末尾：

```bash
echo '/swapfile2 none swap sw 0 0' | sudo tee -a /etc/fstab

```
