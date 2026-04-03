# 运行与配置说明

这份文档按“快速开始 -> 常用命令 -> 配置位置 -> 进阶参数”整理，方便：

- 快速查找运行命令
- 直接复制命令块执行
- 定位需要修改的配置文件

## 快速导航

- [快速开始](#快速开始)
- [常用运行命令](#常用运行命令)
- [Shortest Path 模式](#shortest-path-模式不启用-rl)
- [RL 模式](#rl-模式)
- [`fixed_pairs` 配置](#fixed_pairs-配置)
- [自动追加额外流量对](#自动追加额外流量对默认开启)
- [队列直方图开关](#队列直方图开关)
- [Epsilon 衰减](#epsilon-衰减rl)
- [算法参数对比](#算法关键参数汇总)

## 快速开始

推荐按下面顺序执行：

```bash
# 1) 批量运行不同 flow 的 shortest-path 仿真
python intergrated_test/SP-test.py --start-flow 600e6 --end-flow 1000e6 --step-flow 25e6

# 1.1) （可选）MHGNN 模型并行集成测试
python intergrated_test/MHGNN-test.py --start-flow 600e6 --end-flow 1000e6 --step-flow 25e6 --workers 3 --clean-runtime-dir

# 2) 汇总该日期目录下每次运行的 blockInfo.csv
python intergrated_test/collect-sp-flow-data.py \
    --base-dir SimResults_new/ShortestPath/2026-03-26 \
    --sort-by flow_total_gbps

# 3) 根据汇总结果生成分析图
python intergrated_test/analyze_sp_blockinfo.py \
    --input-csv intergrated_test/SP-blockInfo_merged.csv \
    --output-dir intergrated_test/SP_analysis_figures \
    --output-name fig_latency_subplots_vs_flow.png
```

---

## 常用运行命令

### 基础启动

前台运行：

```bash
python simrl.py
```

后台静默运行：

```bash
python simrl.py > /dev/null 2>&1 &
```

### Shortest Path 批量扫流量

默认运行：

```bash
python intergrated_test/SP-test.py
```

自定义流量范围，从 `500e6` 开始，每次增加 `25e6`，直到 `800e6`：

```bash
python intergrated_test/SP-test.py --start-flow 500e6 --end-flow 800e6 --step-flow 25e6
```

仅覆盖第 `2` 个和第 `4~5` 个固定流，其余保持 `system_configure.py` 默认值：

```bash
python intergrated_test/SP-test.py \
    --start-flow 600e6 --end-flow 1000e6 --step-flow 25e6 \
    --target-indices 2,4-5
```

仅覆盖前 `2` 个固定流，其余保持默认值：

```bash
python intergrated_test/SP-test.py \
    --start-flow 400e6 --end-flow 1000e6 --step-flow 25e6 \
    --target-first-n 2
```

组合使用，覆盖前 `2` 个加第 `5` 个固定流：

```bash
python intergrated_test/SP-test.py \
    --start-flow 600e6 --end-flow 1000e6 --step-flow 25e6 \
    --target-first-n 2 --target-indices 5
```

说明：

- 脚本会临时修改 `system_configure.py` 中 `trafficPairs` 的 rate，并对每个 flow 运行一次 `simrl.py`
- `--target-indices` 使用 1-based 索引，支持区间写法，例如 `2,4-6`
- `--target-first-n` 表示覆盖前 N 个 `trafficPairs`
- 同时提供 `--target-first-n` 与 `--target-indices` 时，最终覆盖集合取并集
- 不提供上述两个参数时，默认覆盖全部 `trafficPairs`
- 默认会在扫描结束后自动恢复 `system_configure.py` 原内容
- 如需保留最后一次写入的流量配置，可增加 `--no-restore-config`

### MHGNN 集成测试

并行扫流量：

```bash
python intergrated_test/MHGNN-test.py \
    --start-flow 700e6 --end-flow 1000e6 --step-flow 100e6 \
    --workers 3 --clean-runtime-dir
```

默认只覆盖前 `2` 个 `trafficPairs`：

```bash
python intergrated_test/MHGNN-test.py \
    --start-flow 700e6 --end-flow 1000e6 --step-flow 100e6 \
    --workers 3
```

仅覆盖前 `3` 个固定流：

```bash
python intergrated_test/MHGNN-test.py \
    --start-flow 700e6 --end-flow 1000e6 --step-flow 100e6 \
    --workers 3 --target-first-n 3
```

仅覆盖第 `2` 个和第 `4~5` 个固定流：

```bash
python intergrated_test/MHGNN-test.py \
    --start-flow 700e6 --end-flow 1000e6 --step-flow 100e6 \
    --workers 3 --target-indices 2,4-5
```

常用参数：

- `--step-flow`：生成流量序列的步长。比如 `700e6 -> 1000e6` 且 `step=100e6`，实际会生成 `700e6`、`800e6`、`900e6`、`1000e6`
- `--workers`：并行任务数
- `--target-first-n`：覆盖前 N 个 `trafficPairs`；如果不写，默认只覆盖前 2 个
- `--target-indices`：按 1-based 位置覆盖指定项，例如 `2,4-5`
- `--runtime-dir`：并行隔离工作目录，默认 `intergrated_test/.mhgnn_parallel_runs`
- `--python-exec`：指定运行 `simrl.py` 的 Python 解释器

运行方式说明：

- `--step-flow` 只负责生成待测试的 flow 列表，不负责调度顺序
- `--workers 3` 表示最多同时运行 `3` 个 `simrl.py` 子进程
- 脚本会先生成全部 flow 任务，再放进线程池调度
- 当某个任务跑完后，会立即补上下一个尚未开始的 flow，而不是“等这一批 3 个全部结束后再统一加一步”

例如：

```text
--start-flow 700e6 --end-flow 1000e6 --step-flow 100e6 --workers 3
```

会生成 4 个任务：

```text
700e6, 800e6, 900e6, 1000e6
```

实际调度大致是：

- 先同时启动 `700e6`、`800e6`、`900e6`
- 谁先结束，就立刻补上 `1000e6`
- 因此总并发上限始终是 `3`

### 汇总批量结果

按 `blockInfo.csv` 汇总，并附带目录名解析出的流量强度：

```bash
python intergrated_test/collect-sp-flow-data.py \
    --base-dir SimResults_new/ShortestPath/2026-03-26 \
    --sort-by flow_total_gbps
```

```bash
python intergrated_test/collect-sp-flow-data.py \
    --base-dir SimResults_new_returnGS/ShortestPath/2026-04-02 \
    --sort-by flow_total_gbps
```

常用参数：

- `--desc`：按指定字段降序排序
- `--output-name blockInfo_merged.csv`：自定义输出文件名，默认即此名称

### 分析与绘图

基于汇总后的 CSV 生成分析图：

```bash
python intergrated_test/analyze_sp_blockinfo.py \
    --input-csv intergrated_test/SP-blockInfo_merged.csv \
    --output-dir intergrated_test/SP_analysis_figures
```

指定输出图片名称：

```bash
python intergrated_test/analyze_sp_blockinfo.py \
    --input-csv intergrated_test/blockInfo_merged_returngs.csv \
    --output-dir intergrated_test/SP_analysis_figures \
    --output-name my_latency_plot.png
```

常用参数：

- `--input-csv`：输入汇总 CSV 路径，默认 `intergrated_test/SP-blockInfo_merged.csv`
- `--output-dir`：图片输出目录，默认 `intergrated_test/SP_analysis_figures`
- `--output-name`：输出图片文件名，不写 `.png` 会自动补齐

---

## Shortest Path 模式（不启用 RL）

### 配置文件 1: `Algorithm/algo_config/base_config.yaml`

```yaml
use_rl_model: false
```

### 配置文件 2: `system_configure.py`

```python
GTs = [4]                # 网关数量（示例）

trafficMode = "all2all"  # 全互联流量
avUserLoad = 28          # all2all 模式下的负载

# 或者：
trafficMode = "fixed_pairs"  # 固定源宿对流量
```

---

## RL 模式

### 训练模式

在 `Algorithm/algo_config/base_config.yaml` 中：

```yaml
use_rl_model: true
agent: "MHGNN"  # 可选: GAT, MHGNN, MPNN, DDQN, MHGNNKGE
train_TA_model: true
use_student_network: false  # 仅 MHGNN 生效
```

对应子配置文件中建议设置项目名，例如 `Algorithm/algo_config/gnn_pd.yaml`：

```yaml
project_name: "satNetEnv_MHGNN_fixedGTsPairs_v1"
```

### 测试模式

在 `Algorithm/algo_config/base_config.yaml` 中：

```yaml
use_rl_model: true
train_TA_model: false
use_student_network: false  # 仅 MHGNN 生效
mode_load_dir: "SimResults/MPNN/best_model/20-57-38_Starlink_3s_GTs_[2]/"
```

说明：

- `system_configure.py` 的流量配置方式与训练一致
- 流量模式仍然使用 `all2all` 或 `fixed_pairs`

---

## `fixed_pairs` 配置

当 `trafficMode = "fixed_pairs"` 时，使用 `trafficPairs`：

```python
trafficPairs = [
    ("Malaga, Spain",               "Aalborg, Denmark", 800e6),
    ("Los Angeles, California, US", "Panama",           800e6),
]
```

每个三元组含义：

```text
(源网关, 目的网关, bps)
```

---

## 自动追加额外流量对（默认开启）

项目支持在手工配置的 `trafficPairs` 基础上，自动追加一组可复现的源宿对。

### 默认行为

- 默认模式：`fixed`
- 默认追加对数：`10`
- 默认总流量：`50 Mbps`
- 默认会自动均分到每一对追加流量
- 默认会避开手工已使用节点对和排除列表

### `system_configure.py` 参数

```python
extraTrafficEnabled = True
extraTrafficPairCount = 10
extraTrafficTotalMbps = 50.0
extraTrafficSelectionMode = "fixed"  # "fixed" | "random"
extraTrafficSeed = 42                 # random 模式下生效

extraTrafficExcludedPairs = [
    # ("Malaga, Spain", "Aalborg, Denmark"),
]

extraTrafficExcludeExistingNodes = True
extraTrafficExcludeExistingPairs = True
extraTrafficExcludeReverse = True
```

说明：

- `fixed`：按确定性规则生成，重复运行结果一致
- `random`：按随机抽样生成，可通过 `extraTrafficSeed` 复现

---

## 队列直方图开关

为降低内存占用，默认关闭逐包队列采样：

```python
enableQueueHistogram = False
```

说明：

- `False`：不记录 `earth.queues` 历史采样，不生成队列直方图文件，内存更稳
- `True`：开启旧版队列采样与队列分布图输出，仅在需要队列分布分析时开启

---

## Epsilon 衰减（RL）

代码位置：

- 衰减函数位于各 Agent 的 `alignEpsilon(step, sat)`
- 触发点位于动作选择阶段 `getNextHop`
- `self.step` 在 `makeDeepAction` 中推进

如果要调探索衰减速度，优先修改对应算法配置文件中的这些参数：

- `LAMBDA`
- `decayRate`
- `MAX_EPSILON`
- `MIN_EPSILON`
- `power`，默认 `2`

当前统一公式 `BaseAgent.alignEpsilon`：

```text
epsilon = minEps + (maxEps - minEps) * exp(-LAMBDA * step / (decayRate * power^2))
```

在 `Algorithm/algo_config/base_config.yaml` 中可直接配置：

```yaml
power: 2
```

经验说明：

- `power` 越大，分母越大，`epsilon` 衰减越慢
- `power` 越小，`epsilon` 衰减越快

---

## 算法关键参数汇总

以下参数来自当前配置文件：

- `Algorithm/algo_config/ddqn.yaml`
- `Algorithm/algo_config/gat.yaml`
- `Algorithm/algo_config/gnn_pd.yaml`
- `Algorithm/algo_config/gnnedge_pd.yaml`
- `Algorithm/algo_config/mpnn.yaml`

| 算法 | n_order_adj | num_layers | learning_rate | buffer_size | updateF | nTrain | batch_size |
|---|---:|---:|---:|---:|---:|---:|---:|
| DDQN | 1 | - | 0.0005 | 1000000 | 23 | 120 | 128 |
| GAT | 1 | 1 | 0.0001 | 1000000 | 23 | 120 | 128 |
| MHGNN (`gnn_pd.yaml`) | 4* | 3 | 0.0001 | 1000000 | 23 | 120 | 128 |
| MHGNNKGE (`gnnedge_pd.yaml`) | 4* | 3 | 0.002 | 1000000 | 23 | 120 | 128 |
| MPNN | 3 | 2 | 0.0001 | 1000000 | 20 | 120 | 128 |

备注：

- `MHGNN` 的 `n_order_adj` 在 `gnn_pd.yaml` 中未单独定义，当前使用 `Algorithm/algo_config/base_config.yaml` 的值，现为 `4`
- `MHGNNKGE` 的 `n_order_adj` 在 `gnnedge_pd.yaml` 中未单独定义，当前也使用 `Algorithm/algo_config/base_config.yaml` 的值，现为 `4`
- `DDQN` 配置中没有 `num_layers` 字段，因此表中记为 `-`
