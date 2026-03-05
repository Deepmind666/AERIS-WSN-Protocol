# 给 Claude 的补充实验提示词（PEGASIS zero-delta 机理验证）

你现在只执行一个补充实验任务，用于验证 S11 中 PEGASIS `delta=0` 是否由 `pegasis_chain_exempt=True` 导致。  
禁止改论文、禁止跑无关实验。

## 目标
在 `indoor_factory` 下，仅对 PEGASIS 做 patch/control 对比，并比较两种碰撞配置：
- 配置 A（当前）：`pegasis_chain_exempt=True`
- 配置 B（对照）：`pegasis_chain_exempt=False`

## 实验矩阵
- environment: `indoor_factory`
- protocol: `PEGASIS`（仅此）
- num_nodes: `100, 500, 1000`
- rounds: `300`
- replicates: `200` / cell
- tx_power: `10.0`
- run_tier: `publication`
- primary_metric: `pdr_expected`

## 允许的代码改动（最小）
仅在实验脚本层增加一个参数透传，不改核心协议逻辑：
1. `scripts/run_scalability_experiment.py`
   - 新增 CLI 参数 `--pegasis-chain-exempt {true,false}`
   - 构造 `MACCollisionConfig(enabled=True, pegasis_chain_exempt=<value>)`
2. 不允许修改 `src/baseline_protocols/pegasis_protocol.py` 算法主体。

## 输出文件
写到 `results/mega_experiments/`：
- `pegasis_chain_exempt_true_20260218.json`
- `pegasis_chain_exempt_false_20260218.json`
- 对应 `*.provenance.json`
- `pegasis_chain_exempt_sanity_delta_20260218.csv`
- `pegasis_chain_exempt_sanity_significance_20260218.csv`

## 统计要求
- 对每个节点规模比较 `false - true`：
  - Welch t-test
  - Hedges' g
  - Holm 校正

## 回报模板（严格）
1. 文件清单  
2. 本次完成（含每个文件 raw_results / error_runs）  
3. 关键结论（每个节点规模 delta 与显著性）  
4. 仍需核对（若有）
