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
