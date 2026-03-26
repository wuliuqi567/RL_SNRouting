# 运行与配置说明

## 1) 启动命令

前台运行：

```bash
python simrl.py
```

后台静默运行：

```bash
python simrl.py > /dev/null 2>&1 &
```

批量流量扫描运行（`intergrated_test/SP-test.py`）：

```bash
python intergrated_test/SP-test.py
```

自定义流量范围（从 500e6 开始，每次 +25e6 到 800e6）：

```bash
python intergrated_test/SP-test.py --start-flow 500e6 --end-flow 800e6 --step-flow 25e6
```

说明：

- 该脚本会临时修改 `system_configure.py` 中 `trafficPairs` 的 rate，并对每个 flow 运行一次 `simrl.py`。
- 默认会在扫描结束后自动恢复 `system_configure.py` 原内容。
- 若希望保留最后一次写入的流量配置，可加参数：`--no-restore-config`。

批量结果汇总（按 `blockInfo.csv`，并附带目录名提取的流量强度）：

```bash
python intergrated_test/collect-sp-flow-data.py \
    --base-dir SimResults_new/ShortestPath/2026-03-26 \
    --sort-by flow_total_gbps
```

可选参数：

- `--desc`：按指定字段降序排序。
- `--output-name blockInfo_merged.csv`：自定义输出文件名（默认即此名）。

MHGNN 集成测试（并行扫流量）：

```bash
python intergrated_test/MHGNN-test.py --start-flow 700e6 --end-flow 1000e6 --step-flow 100e6 --workers 3 --clean-runtime-dir
```

可选参数：

- `--workers`：并行任务数。
- `--runtime-dir`：并行隔离工作目录（默认 `intergrated_test/.mhgnn_parallel_runs`）。
- `--python-exec`：指定运行 `simrl.py` 的 Python 解释器。

完整流程（推荐顺序）：

```bash
# 1) 批量运行不同 flow 的 shortest-path 仿真
python intergrated_test/SP-test.py --start-flow 600e6 --end-flow 1000e6 --step-flow 25e6

# 1.1) （可选）MHGNN 模型并行集成测试
python intergrated_test/MHGNN-test.py --start-flow 600e6 --end-flow 1000e6 --step-flow 25e6 --workers 3 --clean-runtime-dir

# 2) 汇总该日期目录下每次运行的 blockInfo.csv
python intergrated_test/collect-sp-flow-data.py \
    --base-dir SimResults_new/ShortestPath/2026-03-26 \
    --sort-by flow_total_gbps
```

---

## 2) 最短路径模式（不启用 RL）

### `Algorithm/algo_config/base_config.yaml`

```yaml
use_rl_model: false
```

### `system_configure.py`

```python
GTs = [4]                # 网关数量（示例）

trafficMode = "all2all"  # 全互联流量
avUserLoad = 28          # all2all 模式下的负载

# 或者：
trafficMode = "fixed_pairs"  # 固定源宿对流量
```

---

## 3) RL 模式

### 3.1 训练模式

在 `Algorithm/algo_config/base_config.yaml` 中：

```yaml
use_rl_model: true
agent: 'MHGNN'  # 可选: GAT, MHGNN, MPNN, DDQN, MHGNNKGE
train_TA_model: true
use_student_network: false  # 仅 MHGNN 生效
```

在对应子配置（如 `Algorithm/algo_config/gnn_pd.yaml`）中建议设置：

```yaml
project_name: "satNetEnv_MHGNN_fixedGTsPairs_v1"
```

### 3.2 测试模式

在 `Algorithm/algo_config/base_config.yaml` 中：

```yaml
use_rl_model: true
train_TA_model: false
use_student_network: false  # 仅 MHGNN 生效
mode_load_dir: "SimResults/MPNN/best_model/20-57-38_Starlink_3s_GTs_[2]/"
```

`system_configure.py` 的流量配置方式与训练一致（`all2all` 或 `fixed_pairs`）。

---

## 4) 固定流量对（`fixed_pairs`）配置

当 `trafficMode = "fixed_pairs"` 时，使用 `trafficPairs`：

```python
trafficPairs = [
    ("Malaga, Spain",               "Aalborg, Denmark", 800e6),
    ("Los Angeles, California, US", "Panama",           800e6),
]
```

每个三元组含义：`(源网关, 目的网关, bps)`。

---

## 5) 自动追加固定 10 对流量（默认开启）

项目已支持：在你手工配置的 `trafficPairs` 基础上，自动追加一组“可复现”的源宿对。

### 默认行为

- 默认模式：`fixed`（非随机，确定性可复现）
- 默认追加对数：`10`
- 默认总流量：`50 Mbps`（自动均分到追加的每一对）
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

- `fixed`：按确定性规则生成，重复运行结果一致。
- `random`：按随机抽样生成，可通过 `extraTrafficSeed` 复现。

---

## 6) 队列直方图开关（内存相关）

为降低内存占用，默认关闭逐包队列采样：

```python
enableQueueHistogram = False
```

- `False`（推荐）：不记录 `earth.queues` 历史采样，不生成队列直方图文件，内存更稳。
- `True`：开启旧版队列采样与队列分布图输出，仅在需要队列分布分析时开启。

---

## 7) Epsilon 衰减位置（RL）

- 衰减函数在各 Agent 的 `alignEpsilon(step, sat)` 中。
- 触发点在动作选择阶段（`getNextHop`）。
- `self.step` 在 `makeDeepAction` 中推进。

如果要调探索衰减速度，优先在对应算法配置文件中调整：

- `LAMBDA`
- `decayRate`
- `MAX_EPSILON`
- `MIN_EPSILON`
- `power`（新增，默认 `2`）

当前统一公式（`BaseAgent.alignEpsilon`）为：

```text
epsilon = minEps + (maxEps - minEps) * exp(-LAMBDA * step / (decayRate * power^2))
```

在 `Algorithm/algo_config/base_config.yaml` 中可直接配置：

```yaml
power: 2
```

说明：

- `power` 越大，分母越大，epsilon 衰减越慢。
- `power` 越小，epsilon 衰减越快。

---

## 8) 四种算法关键参数汇总

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

- `MHGNN` 的 `n_order_adj` 在 `gnn_pd.yaml` 未单独定义，当前使用 `Algorithm/algo_config/base_config.yaml` 的值（目前为 `4`）。
- `MHGNNKGE` 的 `n_order_adj` 在 `gnnedge_pd.yaml` 未单独定义，当前也使用 `Algorithm/algo_config/base_config.yaml` 的值（目前为 `4`）。
- `DDQN` 配置中没有 `num_layers` 字段，因此表中记为 `-`。
