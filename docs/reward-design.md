会出现这个现象，核心不是“模型不会看拥塞”，而是“奖励把几何推进看得太重了”。

你的当前设计里，`Algorithm/agent/base_agent.py:311` 的总奖励大致是：

```python
reward = distanceReward + queueReward + terminal/failure + again
```

而其中
- `distanceReward` 来自 `Utils/utilsfunction.py:1194`
- `queueReward` 来自 `Utils/utilsfunction.py:1253`

结合你这套状态构造看，模型其实已经能看到不少队列/链路信息，见 `Utils/statefunction.py:516`、`Utils/statefunction.py:596`、`Utils/statefunction.py:635`。所以问题更像是：

- 观测里有拥塞信息
- 但奖励仍然在“鼓励最短几何推进”
- 结果 agent 学成“知道哪里堵，但还是愿意往目标方向挤”

我建议按“从小改到大改”的顺序做。

**先说最有效的方向**
- 最优解不是继续微调几何距离，而是把 shaping 从“几何接近”改成“时延 cost-to-go 接近”
- 也就是：奖励不再问“这一步离目的地近了多少”，而是问“这一步让预计剩余时延降低了多少”
- 这样，绕路只要能降低总时延，就会自然得到正反馈

你可以把修改分成 3 层。

**第一层：最小改动，先把几何偏置压下去**
- 把 `distanceReward` 的正向增益压小，负向惩罚保留
- 把 `queueReward` 改成对毫秒级排队更敏感
- 增加一个小的 hop penalty，防止为了躲拥塞无限绕路

原因很简单：
- 现在 `getQueueReward(queueTime, w1) = w1*(1-10**queueTime)`，对 1ms、5ms、10ms 这种小排队并不够敏感
- 但实际训练里，很多“坏路径”恰恰是每跳都多等一点，累计变得很差
- 所以你要让小排队也早点“疼起来”

我建议先把队列项改成对小量更敏感的形式，比如对数或分段线性，而不是 `10**queueTime`。

例如：

```python
queuePenalty = -wq * math.log1p(queueTime / q_ref)
```

其中
- `q_ref` 可以先取 `0.003 ~ 0.01` 秒
- 含义是：3ms 或 10ms 左右开始明显惩罚
- 这样 1~10ms 区间的梯度会比你现在更实用

距离项也建议不要让“正向推进”太容易拿大分。一个很实用的做法是把它做成饱和型：

```python
x = (SLr - TravelDistance / w4) / d_ref
distanceReward = wd * math.tanh(x)
```

好处是：
- 还是保留“往目标推进”这个 inductive bias
- 但不会因为某一跳几何上特别好，就把排队惩罚完全盖掉
- `tanh` 天然防止 reward scale 爆掉

如果你想更激进一点，可以做成“正向奖励小、负向惩罚大”的非对称版本：

```python
if x >= 0:
    distanceReward = wd_pos * math.tanh(x)
else:
    distanceReward = wd_neg * math.tanh(x)   # wd_neg > wd_pos
```

这会让 agent：
- 不会因为“稍微更近一点”就特别兴奋
- 但如果明显走错方向，会被更强地纠正

一个很实用的起步配置是：
- 几何权重降到原来的 `1/3 ~ 1/2`
- 队列权重提升到原来的 `2 ~ 4` 倍
- 加一个固定 `hopPenalty = -0.2 ~ -1.0`

**第二层：把“几何 shaping”升级成“时延 shaping”**
这是我最推荐的。

你现在的 `distanceReward` 本质是：

```python
离目的卫星更近 = 好
```

但你真正想优化的是：

```python
预计端到端总时延更小 = 好
```

所以更好的势函数应该不是“到目的地的欧氏/斜距”，而是“从当前卫星到目的地的估计剩余时延”。

定义一个 `J(s)`：

```python
J(s) = 预计从卫星 s 到目的地的剩余代价
```

这个代价可以由下面几部分构成：
- 传播时延
- 传输时延
- 预估排队时延
- 小的 hop 代价

然后 shaping 改成：

```python
shaping = beta * (J(prevSat) - J(sat))
```

或者更标准一点，用 potential-based shaping：

```python
F = beta * (gamma * Phi(sat) - Phi(prevSat))
Phi(s) = -J(s)
```

这样做的意义非常大：
- 如果某一步虽然几何上“绕远了”，但让后续路径更畅通，`J(sat)` 还是会下降
- 那么这一步依然会得到正奖励
- 这正是你想要的“允许有价值的绕行”

**怎么得到 `J(s)`**
最现实的工程做法是：
- 在每次图更新后
- 以目的卫星为终点
- 在当前图上跑一次最短路/Dijkstra
- 边权不要再只用 slant range，而用“链路时延代理”

比如：

```python
edge_cost(u, v) =
    a * prop_delay(u, v)
  + b * tx_delay(u, v)
  + c * queue_est(v)
  + d
```

其中
- `prop_delay(u,v)` 可以用链路距离 / 光速
- `tx_delay(u,v)` 可以用 `block.size / dataRate`
- `queue_est(v)` 可以用邻居排队长度、buffer backlog、或你当前 state 里的 queue score 映射出来
- `d` 是小的每跳代价，避免平白增加 hop

然后 reward 可以写成：

```python
reward =
    - immediate_delay_cost
    + beta * (J(prevSat) - J(sat))
    + arrive_bonus / fail_penalty / loop_penalty
```

其中 `immediate_delay_cost` 建议至少包括：
- 上一跳真实 queue time
- 当前链路传播时延
- 当前链路传输时延

这个版本的优点是：
- 跟最终 KPI 更一致
- 会自然学会“为了避堵而绕一下”
- 比单纯调 `w1/w2` 更稳

**第三层：如果不想跑最短路，也可以做“局部拥塞前瞻”**
如果你觉得每次图更新都算 `J(s)` 太重，可以先做个中间版本：

把 reward 从
- “只看上一跳走到了哪里”

改成
- “还要看这个落点卫星接下来的拥塞程度”

也就是加入一个 `next_congestion_penalty`：

```python
reward =
    distanceReward
  + queuePenalty
  - wc * congestion_score(sat)
  - wh * hopPenalty
```

这里 `congestion_score(sat)` 可以来自：
- `sat` 四个方向的 queue score 均值
- `sat` 最优出边的 queue score
- `sat` 的总 backlog / 出链路可服务能力
- 或 `min(queue/rate)` 这样的“最有希望下一跳”的局部代价

这相当于告诉 agent：
- “你虽然离目标近了，但你把包送进了一个堵点，还是不好”

这种做法虽然不如 `J(s)` 干净，但实现很快，而且通常比纯几何 reward 好不少。

如果已经有 `layer2` 的 `delay-to-go` shaping，再叠加这类局部项时要注意一件事：

- 最好罚“额外拥塞风险”，不要再罚一遍正常传播时延
- 否则 `layer3` 很容易出现“能收敛，但 reward 数值整体偏低”的现象
- 这并不一定代表策略更差，而可能只是 reward baseline 被向下平移了
