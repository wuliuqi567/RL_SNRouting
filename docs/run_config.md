# run program

代码运行指令
终端无输出模式

```bash
python simrl.py > /dev/null 2>&1 & 
```


- 最短路径
  
  ```bash
  #修改base_config.yaml中
  use_rl_model: false 
  
  # system_configure.py中
  GTs = [4] # 确保几个网关数量
  
  trafficMode = "all2all" # 流量模式，当选择all2all时候需要对应修改avUserLoad
  avUserLoad  = 28 # 流量负载
  trafficMode = "fixed_pairs" # 流量模式，当选择fixed_pairs时候需要 添加 trafficPairs
  ```

- RL算法
  
  ```bash
  #修改base_config.yaml中
  use_rl_model: true
  
  #选择模型
  agent: 'MHGNN' # GAT, MHGNN, MPNN, DDQN, MHGNNKGE

  # training parameters 设置
  train_TA_model: true # whether to train the teacher agent network
  use_student_network: false # whether to use the student network for action selection

  当训练模式时候，再子yaml文件中明确 项目名称
  project_name: "satNetEnv_MHGNN_fixedGTsPairs_v1"

  当测试模式时候
  train_TA_model: false # whether to train the teacher agent network
  use_student_network: false # whether to use the student network for action selection 只有在MHGNN下生效，其他算法没有
  需要配置模型的地址，
  mode_load_dir: "SimResults/MPNN/best_model/20-57-38_Starlink_3s_GTs_[2]/"


  # system_configure.py中
  GTs = [4] # 确保几个网关数量
  
  trafficMode = "all2all" # 流量模式，当选择all2all时候需要对应修改avUserLoad
  avUserLoad  = 28 # 流量负载
  trafficMode = "fixed_pairs" # 流量模式，当选择fixed_pairs时候需要 添加 trafficPairs
  ```

# Run Config Notes

## Queue histogram switch (memory related)

To reduce memory usage, legacy per-packet queue histogram collection is disabled by default.

- Config key: `enableQueueHistogram`
- File: `system_configure.py`
- Default: `False`

### Behavior

- `False` (recommended):
  - Do **not** append queue length samples into `earth.queues` for every packet enqueue.
  - Do **not** generate `Queues_*_Gateways.csv/png` at the end.
  - Lower memory usage for long or high-load simulations.

- `True`:
  - Enable legacy queue histogram sampling and end-of-run queue histogram plotting.
  - Useful only when you specifically need queue-length distribution figures.

### How to enable

In `system_configure.py`:

```python
enableQueueHistogram = True
```

Then run `simrl.py` as usual.

## Epsilon decay analysis (MHGNN)

### Where epsilon decays

- Epsilon is decayed in `MHGNNAgent.alignEpsilon(step, sat)`.
- The update is triggered in `getNextHop()` by:
  - `if self.train_TA_model and random.uniform(0, 1) < self.alignEpsilon(self.step, sat):`
- `self.step` is incremented in `makeDeepAction()`.
- The logging line `self.log_infos_no_index({"epsilon": ...})` only records value and does not affect decay.

### Decay formula

```
epsilon = minEps + (maxEps - minEps) * exp(-LAMBDA * step / (decayRate * CurrentGTnumber^2))
```

Key parameters (from `Algorithm/algo_config/gnn_pd.yaml`):

- `MAX_EPSILON: 0.60`
- `MIN_EPSILON: 0.001`
- `LAMBDA: 0.0005`
- `decayRate: 50`

`CurrentGTnumber` is written at runtime in `simrl.py` as:

- `config_data['CurrentGTnumber'] = GTs[0]`

### Why decay feels too slow

The effective decay speed is controlled by:

- numerator: `LAMBDA * step`
- denominator: `decayRate * CurrentGTnumber^2`

So increasing gateway count slows decay quadratically.

Examples with current parameters:

- If `CurrentGTnumber=2`: exponent is about `exp(-step/400000)`
  - half-life ≈ `0.693 * 400000 ≈ 277k` steps
- If `CurrentGTnumber=4`: exponent is about `exp(-step/1600000)`
  - half-life ≈ `1.11M` steps
- If `CurrentGTnumber=31`: exponent is about `exp(-step/96100000)`
  - decay is extremely slow in practical training windows

### Practical tuning directions

If you want faster exploration decay:

- Increase `LAMBDA` (most direct)
- Decrease `decayRate`
- Remove or weaken the `CurrentGTnumber^2` scaling (e.g., use `CurrentGTnumber` instead)

Suggested conservative start:

- keep formula unchanged first
- set `LAMBDA` from `0.0005` -> `0.005` (10x)
- keep `MIN_EPSILON` unchanged
- observe first 100k-300k steps and compare reward stability
