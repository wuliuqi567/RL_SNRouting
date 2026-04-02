# 奖励函数设计说明

本文档说明当前项目中 RL 路由奖励函数的设计目标、数学形式、代码实现位置、运行流程、配置方式与调参建议。

文档对应的主要实现文件：

- `Algorithm/agent/base_agent.py`
- `Algorithm/algo_config/base_config.yaml`
- `Utils/utilsfunction.py`
- `Utils/statefunction.py`

当前主算法链路为：

- `MHGNNAgent`
- `BaseAgent.makeDeepAction()`
- `BaseAgent._calculate_reward_v1()`

也就是说，`mhgnn_agent.py` 实际训练时使用的是 `BaseAgent` 中实现的奖励逻辑。

---

## 1. 设计目标

本项目的奖励函数不是单纯优化“几何最短路”，而是试图在以下目标之间做平衡：

1. 包尽快到达目的地
2. 尽量减少局部排队时延
3. 尽量减少传播与传输代价
4. 尽量避免回环和无意义跳数
5. 允许“有价值的绕行”，而不是一味朝几何目的地方向贪心推进

为此，奖励函数被设计为多层递进形式：

- `legacy`
- `layer1`
- `layer2`
- `layer3`

不同层级逐步从“几何 shaping”过渡到“时延 shaping”和“局部拥塞前瞻”。

---

## 2. 奖励计算发生在什么时候

奖励不是在“选动作那一刻”立即计算，而是在数据块真正到达下一颗卫星之后，对上一跳动作进行回溯评价。

对应流程如下：

1. 在 `BaseAgent.makeDeepAction()` 中，智能体为当前卫星选择下一跳。
2. 数据块经由链路发送、传播、接收。
3. 当下一颗卫星收到该数据块时，会再次触发 `makeDeepAction()`。
4. 此时使用：
   - `prevSat`：上一颗卫星
   - `sat`：当前到达的卫星
5. 奖励函数 `_calculate_reward_v1(block, sat, prevSat, ...)` 就是在这个时刻评估“上一跳 `prevSat -> sat` 是否合理”。

这样做的好处是：

- 奖励与动作后果对齐
- 可以使用真实观测到的队列等待时间
- 可以在终止状态统一补充到达奖惩

---

## 3. 统一记号

后文使用以下符号。

- `s_prev`：上一跳卫星，即 `prevSat`
- `s_cur`：当前到达卫星，即 `sat`
- `s_dest`：目的地网关当前关联卫星，即 `block.destination.linkedSat[1]`
- `q`：上一跳真实排队时间，即 `block.queueTime[-1]`
- `d_prop(s_prev, s_cur)`：链路传播时延
- `d_tx(s_prev, s_cur)`：链路传输时延
- `J(s)`：从卫星 `s` 到 `s_dest` 的估计剩余代价
- `R`：当前步奖励

辅助常量主要来自 `base_config.yaml`：

- `arrive_reward`
- `failure_penalty`
- `loop_penalty`
- `reward_distance_scale`
- `reward_queue_scale`
- `reward_prop_scale`
- `reward_tx_scale`
- `reward_delay_beta`
- `reward_local_congestion_scale`

---

## 4. 总体结构

当前奖励函数总体结构可以写成：

```text
R_final = R_base + R_terminal_or_failure
```

其中：

```text
R_terminal_or_failure =
    + arrive_reward      if terminal
    - failure_penalty    if failure
    0                    otherwise
```

而 `R_base` 取决于当前 `reward_mode`。

---

## 5. legacy 模式

### 5.1 设计思想

这是最早的奖励形式，核心是：

- 朝目的卫星几何靠近
- 排队越少越好
- 避免重复访问历史卫星

### 5.2 数学形式

```text
R_legacy = R_distance + R_queue + R_loop
```

其中：

```text
R_distance = getDistanceRewardV4(s_prev, s_cur, s_dest, w2, w4)
R_queue    = getQueueReward(q, w1)
R_loop     = -loop_penalty    if revisited
             0                otherwise
```

### 5.3 距离项

`getDistanceRewardV4()` 的核心形式是：

```text
SLr = dist(s_prev, s_dest) - dist(s_cur, s_dest)
TravelDistance = dist(s_prev, s_cur)

R_distance = w2 * (SLr - TravelDistance / w4) / biggestDist
```

含义如下：

- `SLr > 0` 表示当前跳让数据块更接近目的卫星
- `TravelDistance / w4` 是对链路长度本身的代价约束
- `biggestDist` 用于全局归一化

### 5.4 legacy 的局限

该模式容易产生明显的几何偏置：

- 倾向选择“几何上更近”的方向
- 对局部排队与后续拥塞考虑不足
- 即使存在更优绕行路径，也可能过早朝目的地方向挤压

---

## 6. layer1：弱化几何偏置，增强排队敏感性

### 6.1 设计思想

`layer1` 是对 `legacy` 的保守改造，目标是：

1. 保留几何引导作用
2. 降低几何项主导性
3. 提高对毫秒级排队时延的敏感度
4. 增加固定 hop 惩罚

### 6.2 数学形式

```text
R_layer1 = R_distance_sat + R_queue_log + R_hop + R_loop
```

其中：

```text
R_distance_sat = reward_distance_scale * tanh(R_distance_raw / reward_distance_ref)
R_queue_log    = - reward_queue_scale * log(1 + q / reward_queue_ref)
R_hop          = - reward_hop_penalty
R_loop         = - loop_penalty   if revisited
                 0                otherwise
```

这里：

```text
R_distance_raw = getDistanceRewardV4(...)
```

### 6.3 为什么使用 `tanh`

使用 `tanh` 的原因是让几何奖励饱和：

- 防止个别“几何推进特别大”的跳数过度主导训练
- 保留方向性
- 限制 reward scale

### 6.4 为什么使用 `log(1 + q / q_ref)`

相比旧版 `10**q` 形式，对数型惩罚对小排队时间更敏感，也更稳定：

- 1ms 到 10ms 区间更容易被模型感知
- 避免极端指数爆炸
- 更适合作为训练中的连续 shaping 项

### 6.5 适用场景

`layer1` 适合作为从旧奖励迁移到新奖励的第一步：

- 训练稳定性通常较好
- 不会一下子改变目标函数语义太多
- 可用于确认“是不是几何项过强”这一判断

---

## 7. layer2：即时链路时延 + 剩余时延 shaping

### 7.1 设计思想

`layer2` 是本项目奖励设计的核心升级版。

它不再只问：

```text
这一跳是否几何上更靠近目的地？
```

而是问：

```text
这一跳是否降低了估计剩余时延？
```

这使得 agent 可以学习：

- 允许短期绕路
- 只要绕路能换来更低总时延，就应给予正反馈

### 7.2 数学形式

```text
R_layer2 = - C_immediate + R_delay_to_go + R_loop
```

其中即时成本：

```text
C_immediate =
    reward_queue_scale * log(1 + q / reward_queue_ref)
  + reward_prop_scale  * (d_prop / reward_prop_ref)
  + reward_tx_scale    * (d_tx / reward_tx_ref)
  + reward_hop_penalty
```

所以：

```text
- C_immediate
```

表示“当前这一跳消耗的真实成本”。

### 7.3 即时成本各项含义

#### 7.3.1 排队项

```text
q = 上一跳真实队列等待时间
```

这是已经实际发生的等待时间，因此具有很强的训练信号意义。

#### 7.3.2 传播项

```text
d_prop = slant_range / Vc
```

其中：

- `slant_range` 为上一跳链路距离
- `Vc` 为光速

#### 7.3.3 传输项

```text
d_tx = block.size / dataRate
```

其中：

- `block.size` 为当前数据块大小
- `dataRate` 为链路数据率

### 7.4 剩余时延势函数

定义：

```text
J(s) = 从卫星 s 到目的卫星 s_dest 的估计剩余代价
```

则 shaping 项为：

```text
R_delay_to_go =
    reward_delay_beta * tanh((J(s_prev) - J(s_cur)) / reward_delay_ref)
```

解释如下：

- 若 `J(s_cur) < J(s_prev)`，表示本跳降低了剩余代价，给正奖励
- 若 `J(s_cur) > J(s_prev)`，表示本跳使未来更差，给负奖励

### 7.5 `J(s)` 如何估计

当前实现中，`J(s)` 使用当前图上的 Dijkstra 最短路近似：

```text
J(s) = shortest_path_cost(s -> s_dest)
```

边代价定义为：

```text
cost(u, v) =
    reward_prop_scale * normalize(d_prop(u, v))
  + reward_tx_scale   * normalize(d_tx(u, v))
  + reward_remaining_hop_cost
  + node_delay(v)
```

其中：

```text
node_delay(v) =
    reward_remaining_queue_scale * normalize(best_egress_delay(v))
```

这里的 `best_egress_delay(v)` 表示：

```text
从卫星 v 出发，选择其最优邻接出口时的局部最小延迟估计
```

具体近似为：

```text
best_egress_delay(v) = min_over_neighbors (
    queue_wait(v -> n) + prop_delay(v -> n)
)
```

### 7.6 为什么这个设计比纯几何更合理

因为它将未来代价从“空间距离”改成了“时延代价近似”：

- 允许几何绕路
- 只要未来更快，奖励就会变好
- 更贴近最终 KPI：总时延、排队时延、有效吞吐路径

### 7.7 fallback 机制

如果某些时刻 `J(s)` 无法计算，比如：

- 图中找不到 `s_dest`
- 当前节点不连通

则实现会回退到一个弱化版几何 shaping：

```text
R_fallback =
    0.5 * reward_distance_scale * tanh(R_distance_raw / reward_distance_ref)
```

这样可以避免训练过程中出现大量空奖励。

---

## 8. layer3：在 layer2 上加入局部拥塞前瞻

### 8.1 设计思想

`layer3` 的目标是进一步抑制“落到热点节点”的行为。

即便当前跳本身看起来代价不高，如果当前落点卫星周围的出口已经明显拥塞，仍应施加额外惩罚。

### 8.2 数学形式

```text
R_layer3 = R_layer2 - P_local_congestion
```

其中：

```text
P_local_congestion =
    reward_local_congestion_scale
    * tanh(local_congestion_delay / reward_local_congestion_ref)
```

### 8.3 本地拥塞前瞻如何计算

当前实现中，对当前落点卫星 `s_cur` 的每个邻居，先计算：

```text
candidate_delay =
    queue_wait(s_cur -> n)
  + prop_delay(s_cur -> n)
```

然后按 `candidate_delay` 排序，取其中最优的前两个出口，但真正用于 layer3 惩罚的是这两个出口的平均排队等待：

```text
local_congestion_queue_delay = mean(best_2_queue_waits)
P_local_congestion =
    reward_local_congestion_scale
    * tanh(local_congestion_queue_delay / reward_local_congestion_ref)
```

同时仍会在日志里保留：

```text
local_congestion_delay = mean(best_2_candidate_delays)
local_congestion_prop_delay = mean(best_2_prop_delays)
```

为什么这样改：

- 如果直接惩罚 `queue_wait + prop_delay`，会把正常几何传播时延也重复算进 layer3
- 这会让 `layer3` 的 reward 基线天然低于 `layer2`，即便实际路由效果未必更差
- 改成只惩罚“局部出口排队风险”，更符合“local congestion lookahead”的语义

为什么仍取前两个而不是最小一个：

- 只取最小值会过于乐观
- 使用前两个平均值可以降低偶然单个“假优出口”的影响

### 8.4 layer3 的作用

相对于 `layer2`：

- `layer2` 更关注“整体剩余时延”
- `layer3` 在此基础上额外强调“当前落点的局部出口拥塞风险”

这能帮助策略减少：

- 落入局部热点
- 明知前方拥塞仍继续扎堆

### 8.5 layer3 的风险

如果本地拥塞惩罚过强，会出现：

- 路由过度保守
- 因回避局部拥塞而增加过多 hop
- 训练变慢或 reward 长期偏负

因此需要重点关注：

- `reward_local_congestion_scale`
- `reward_remaining_queue_scale`
- `reward_local_congestion_ref`

额外说明：

- 如果 `layer2` 能收敛且 KPI 更好，但 `layer3` 只是 reward 更低，不应只看 reward 曲线下结论
- 因为 `layer3` 相比 `layer2` 多了一项常驻惩罚，数值基线天然不完全可比
- 更应该同时看 `avgTime`、`Queue time`、送达率和平均 hop

---

## 9. 回环惩罚

为了抑制循环路由，当前实现会检查当前卫星是否已经在历史 `QPath` 中出现过。

若重复访问：

```text
R_loop = - loop_penalty
```

否则：

```text
R_loop = 0
```

当前实现是按卫星 `ID` 判重，而不是按经纬度判重，这一点很重要。

原因是卫星在运动：

- 如果按 `[ID, lon, lat]` 比较
- 同一颗卫星在不同时刻坐标不同
- 会导致重复访问漏判

因此按 `sat.ID` 判重更稳健。

---

## 10. 终止与失败奖励

所有模式都会统一经过终止修正：

### 10.1 到达目的地

```text
R_final = R_base + arrive_reward
```

### 10.2 失败

```text
R_final = R_base - failure_penalty
```

失败条件通常为：

- 超出最大跳数
- 仍未到达目的地

### 10.3 作用

这一项提供任务级信号：

- 仅靠 shaping 可能学会“看起来合理”的局部策略
- 终止奖惩强制智能体真正关注送达成功与失败

---

## 11. 代码实现映射

奖励主入口：

- `BaseAgent._calculate_reward_v1()`

各模式实现：

- `_calculate_reward_legacy()`
- `_calculate_reward_layer1()`
- `_calculate_reward_layer2()`
- `_calculate_reward_layer3()`

关键辅助函数：

- `_get_immediate_delays()`
- `_get_delay_to_go()`
- `_get_delay_to_go_shaping()`
- `_estimate_best_egress_delay()`
- `_estimate_local_congestion_delay()`
- `_is_revisited_sat()`

使用到的图与链路信息：

- 当前运行图：`earth.graph` 或 `gateway.graph`
- 边属性：
  - `slant_range`
  - `dataRateOG`

---

## 12. 运行与配置方法

### 12.1 配置奖励模式

在 `Algorithm/algo_config/base_config.yaml` 中设置：

```yaml
reward_mode: 'layer1'
```

可选值：

- `legacy`
- `layer1`
- `layer2`
- `layer3`

### 12.2 典型参数

```yaml
arrive_reward: 50.0
failure_penalty: 50.0
loop_penalty: 10.0

reward_distance_scale: 4.0
reward_distance_ref: 1.0
reward_queue_scale: 8.0
reward_queue_ref: 0.003
reward_hop_penalty: 0.3

reward_prop_scale: 1.0
reward_prop_ref: 0.01
reward_tx_scale: 1.0
reward_tx_ref: 0.001
reward_delay_beta: 3.0
reward_delay_ref: 1.0
reward_remaining_queue_scale: 0.5
reward_remaining_hop_cost: 0.2

reward_local_congestion_scale: 2.0
reward_local_congestion_ref: 0.01
```

### 12.3 推荐实验顺序

建议按以下顺序开展实验：

1. `layer1`
2. `layer2`
3. `layer3`

原因是：

- `layer1` 用来验证“是不是几何偏置过强”
- `layer2` 用来验证“时延 shaping 是否改善总时延”
- `layer3` 用来验证“局部拥塞前瞻是否进一步降低排队热点”

---

## 13. 奖励日志功能

当前实现支持手动开关奖励分量日志。

### 13.1 配置项

```yaml
reward_log_enabled: false
reward_log_interval: 100
```

含义：

- `reward_log_enabled`
  - `false`：默认关闭
  - `true`：开启奖励分量日志
- `reward_log_interval`
  - 每隔多少个 `self.step` 记录一次

### 13.2 记录内容

开启后会记录如下指标中的一部分或全部：

- `reward/base`
- `reward/final`
- `reward/queue_penalty`
- `reward/prop_penalty`
- `reward/tx_penalty`
- `reward/hop_penalty`
- `reward/delay_shaping`
- `reward/delay_to_go_prev`
- `reward/delay_to_go_cur`
- `reward/delay_to_go_improvement`
- `reward/local_congestion_penalty`
- `reward/loop_penalty`
- `reward/is_terminal`
- `reward/is_failure`

### 13.3 用途

这些日志可以帮助判断：

- 哪一项在主导训练
- 几何项是否仍然压过时延项
- `layer3` 是否过度惩罚局部拥塞
- 失败惩罚是否被中间 shaping 抵消

---

## 14. 调参建议

### 14.1 如果仍然几何贪心

优先调整：

- 降低 `reward_distance_scale`
- 提高 `reward_queue_scale`
- 提高 `reward_delay_beta`

### 14.2 如果路径绕得太多

优先调整：

- 提高 `reward_hop_penalty`
- 提高 `reward_remaining_hop_cost`
- 适度降低 `reward_local_congestion_scale`

### 14.3 如果训练过于保守

优先调整：

- 降低 `reward_local_congestion_scale`
- 降低 `reward_remaining_queue_scale`
- 降低 `reward_queue_scale`

### 14.4 如果 reward 波动太大

优先调整：

- 提高 `reward_distance_ref`
- 提高 `reward_delay_ref`
- 调大 `reward_log_interval`，减少日志噪声干扰观测

---

## 15. 当前实现中的几个关键工程点

### 15.1 图类型兼容

项目当前运行图是 `networkx.Graph`，不是 `DiGraph`。

因此在计算 `delay-to-go` 时，代码已兼容：

- 如果图支持 `reverse()`，则反转
- 否则直接用原图做 Dijkstra

### 15.2 `delay-to-go` 节点代价挂载位置

在 Dijkstra 中，节点局部延迟代价应挂到“被进入的节点”上，而不是离开的节点上。

也就是说边代价中的：

```text
node_delay(v)
```

应该作用于目标节点 `v`。

这一点直接影响 `layer2` 和 `layer3` 的合理性。

### 15.3 fallback 的意义

当剩余代价估计失败时，回退到弱化版几何 shaping，可以：

- 保证训练不中断
- 避免奖励全空
- 降低由于动态图暂时不连通导致的极端噪声

---

## 16. 总结

四种奖励模式的关系可以概括为：

- `legacy`
  - 几何推进 + 队列 + 回环
- `layer1`
  - 减弱几何主导，增强排队敏感，加入 hop 惩罚
- `layer2`
  - 引入即时链路时延与剩余时延 shaping
- `layer3`
  - 在 `layer2` 基础上增加局部拥塞前瞻

如果目标是降低“几何贪心导致的高排队时延”，通常推荐：

1. 先用 `layer1` 验证方向
2. 再用 `layer2` 作为主力方案
3. 最后视情况尝试 `layer3`

在实际实验中，建议始终同时观察：

- 平均总时延
- 平均 queue latency
- 平均 hop
- 失败率
- 奖励分量日志

这样才能判断奖励是否真的推动了策略向预期方向演化。
