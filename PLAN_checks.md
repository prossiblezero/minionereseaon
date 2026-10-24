# MiniOneReason 结果稳健性检查方案

方案日期：2026-10-24。本文交给远程执行代理（Codex）。所列命令、文件和结果都是待实现目标，不代表已经存在或已经验证。

## 0 目的与原则

seed 42 主队列已完成（见 `reports/PROJECT_REPORT.md`）。主结论是：只有 `rl_nk_prefix` 让推荐变好，thinking 模式测试 Recall@32 从 mixed SFT 的 2.233% 升到 4.467%，配对 bootstrap 区间 [+0.478, +4.082] 个百分点。写进简历之前，需要回答三个问题：

| 编号 | 问题 | 检查 |
| --- | --- | --- |
| Q1 | 这个提升是否超出评测采样本身的随机波动？同一个 `rl_nk_prefix` 模型的 non-thinking Recall@32 在三次独立评测中分别为 2.978%、3.474%、4.963%，波动接近 2 个百分点，与主效应相当。现有 bootstrap 只重采样测试样本，不包含采样随机数的波动。 | 第 2 节多采样种子重评 |
| Q2 | 提升能否在独立的官方数据上复现？目前官方商品任务只评了 RL 模型，没有基线，数字无法解读。 | 第 3 节官方评测补基线 |
| Q3 | 提升来自“少输出目录外 SID”，还是“推荐更准”？RL 后无效比例从约 39% 降到约 30%。 | 第 4 节有效性分解（纯 CPU） |

原则：

1. 所有被检查的模型都是已经按验证集选定并冻结的适配器，本轮不训练、不重新选择检查点、预算、阈值或变体。用固定模型在测试集上增加采样次数，不构成用测试集调参。
2. 判断规则在第 2.4、3.4、4.3 节预先写定。运行前把本文件的 SHA-256 与判断规则写入 `reports/checks/protocol.json`，所有结果文件的生成时间必须晚于它。
3. 不修改已有的 `reports/results_v2`、`predictions_v2`、`PROJECT_REPORT.md`、`RESUME.md` 和已冻结的选择文件。新产物只写入 `reports/checks/` 和数据盘的 `predictions_checks/`。
4. 结论无论正负都如实报告。若检查削弱了主结论，报告中写清楚，不为了保住结论增加采样或更换指标。

### 0.1 本轮必做范围（精简版，优先于下文）

用户要求压缩工作量。本轮只做下表三项，其余全部降为可选，未经用户确认不运行：

| 必做 | 内容 | GPU | 预计耗时 |
| --- | --- | --- | --- |
| M1 | 第 4 节有效性分解，只用已有原始预测 | 无 | 几分钟 |
| M2 | 第 2 节 non-thinking 重评：只评 `mixed` 和 `rl_nk_prefix`，offset 1 到 4，共 8 次 | 约 9 到 20 GiB | 合计约 15 分钟 |
| M3 | 第 3 节官方评测：只补 `mixed` SFT 一个模型 | 约 20 GiB | 与上一次 RL 官方评测相当 |

精简版的判断方式：

- M2 按第 2.4 节规则只判断 non-thinking 模式。thinking 模式不重评，报告中沿用已有的 5 组配对评测（thinking、non-thinking、预算 0、预算 1024、路由全 non-thinking）中 `rl_nk_prefix` 均高于 `mixed` 这一事实，并注明 thinking 模式的幅度没有经过多采样检验。
- M3 按第 3.4 节规则判断 `rl_nk_prefix − mixed`。不补预训练模型，报告中不写“相对预训练提升”。
- M1 只使用 offset 0 的已有预测；M2 完成后，把 non-thinking 的新预测并入再算一次。

第 2.2 节中的 thinking 重评、`rl_nk` 对照、offset 0 确定性检查，第 3.1 节中的预训练与 `rl_nk`，以及第 5 节全部内容，在本轮均为可选。

## 1 环境与显存

### 1.1 机器

沿用最近一次运行的 5090 服务器 `dsj5090`、数据根目录 `/data3/User/minionereason`、代码目录 `/home/dzh/minionereason`，以 `artifacts/runtime_identity.json` 记录的 GPU 为默认设备。开始前：

1. 检查 `reports/pipeline.lock`、`reports/active_stage.txt` 与实际进程，确认没有本项目或 `minionerec` 的 GPU 任务在运行。
2. 记录两张卡的 `nvidia-smi` 快照到 `reports/checks/gpu_before.json`，选择空闲显存满足第 1.2 节要求的那张卡，并写入 `reports/checks/runtime_identity.json`。
3. 同一时间只运行一个 GPU 任务。

### 1.2 显存需求

| 检查 | GPU 占用 | 说明 |
| --- | --- | --- |
| 第 4 节有效性分解 | 0 | 只读原始预测，纯 CPU |
| 第 2 节本地重评（vLLM） | 默认约 20.4 GiB | vLLM 按 `vllm_memory_fraction=0.65` 预先占用整卡比例，5090 上约 20.4 GiB。实际需求：模型权重约 1.5 GiB；KV cache 每 token 约 112 KiB（28 层 × 8 个 KV 头 × 128 维 × K/V × BF16），thinking 模式 64 条并发、每条约 2000 token 时最多约 14 GiB |
| 第 3 节官方评测 | 默认约 20.4 GiB | 官方脚本同样使用 0.65，`max_model_len=8192`；上一次运行日志显示 KV cache 17.18 GiB、权重 1.50 GiB。合并 LoRA 的步骤另需约 2 到 3 GiB，可在 CPU 上完成 |
| 第 5 节可选：约束解码 | 约 4 到 8 GiB | Transformers 后端，仅 non-thinking |
| 第 5 节可选：补种子 43 | 与原训练相同 | 需要整卡，放在最后，由用户决定 |

默认保持 0.65，与已有评测完全一致，避免批处理差异引入额外的数值变化。仅当所选 GPU 的空闲显存不足 0.65 × 总显存加 1 GiB 余量时，才允许按以下顺序降低，并在每个运行记录中写明实际比例：

1. non-thinking 评测降到 0.30（约 9.4 GiB，KV cache 足够 64 条约 1000 token 的序列）。
2. thinking 评测和官方评测降到 0.45（约 14 GiB）。vLLM 在 KV 不足时会抢占重算，结果仍然正确但变慢。
3. 仍不满足时停止并报告，不挤占其他用户的进程。

降低比例前，先用 32 条固定测试样本、offset 0 与已有原始预测对比，确认解析结果一致；不一致则写入报告，并说明后续结果在该比例下产生。

## 2 Q1：多采样种子重评

### 2.1 机制

`evaluate.py` 的 `evaluate_one` 已支持 `seed_offset`，每条样本的采样种子为 `(seed + seed_offset + digest(sample_id)) mod (2^31 − 1)`。已有主评测对应 offset 0。本节新增 offset 1、2、3、4，每个设置共 5 次独立采样。所有其他参数与主评测相同：测试集 403 条、recent 历史、输入上限 2048、每条 64 次尝试、temperature 0.8、top-p 0.95、vLLM 后端、同一适配器哈希。

### 2.2 运行矩阵

| 模型（适配器 SHA-256 前 8 位） | non-thinking | thinking |
| --- | --- | --- |
| `mixed`（af31b185） | offset 1 到 4，必做 | offset 1 到 4，必做 |
| `rl_nk_prefix`（c46b30a4） | offset 1 到 4，必做 | offset 1 到 4，必做 |
| `rl_nk`（937ecc5c） | offset 1 到 4，必做 | offset 1、2，可选 |

`rl_nk` 是“同样的 N×K 采样、但没有前缀奖励”的对照，用来检验收益是否来自前缀奖励。

另做一次确定性检查：用 `rl_nk_prefix` non-thinking、offset 0 重新生成全部 403 条，与已有原始预测逐条比较。完全一致说明评测可复现；不一致时报告不一致的比例，并说明 offset 0 的已有结果与本轮结果分别来自哪次运行。

运行标签使用 `checks_<variant>_42_test_think<True|False>_offset<k>`，原始预测写入 `predictions_checks/`，每次运行的设置与适配器哈希写入同名 summary。

时间估计（按已有记录，共享负载下）：thinking 每次约 12 秒/样本，403 条约 80 分钟，必做的 8 次约 11 小时；non-thinking 每次不到 1 分钟。

### 2.3 统计

对每个模型与模式输出 `reports/checks/eval_seed_summary.json`：

- 每次采样的 Recall@8、@32、@64、无效比例；5 次的均值与样本标准差。
- 配对差 `rl_nk_prefix − mixed` 与 `rl_nk_prefix − rl_nk`：逐 offset 的差值，以及 5 次中为正的次数。
- 合并估计：先对每条样本把 5 次采样的命中取平均，再按输入组做 1000 次 bootstrap，得到差值的 95% 区间。
- 方差对照：5 次采样之间的标准差，与原单次 bootstrap 标准误并列，说明两种不确定性哪个更大。

### 2.4 判断规则

分 thinking 与 non-thinking 两个模式分别判断，以 Recall@32 为主指标：

- **稳定**：5 次采样的差值全部为正，并且合并 bootstrap 区间下限大于 0。
- **方向一致、幅度不确定**：5 次中至少 4 次为正，但合并区间包含 0。
- **未得到支持**：其他情况。

报告用合并估计的均值和区间替代原来单次的 +2.233 个百分点，作为后续简历的依据。

## 3 Q2：官方商品评测补基线

### 3.1 运行

使用与已完成 RL 评测完全相同的入口 `ops/official.py`、相同的官方代码提交、v3.1-public 数据、`challenge_recommendation_product` 任务、全部 1000 条、seed 42、Race 32 个思考 + 32 个直接候选、4096 token 思考预算、`max_model_len=8192`、相同的运行补丁。LoRA 模型用与 RL 相同的方式合并为 BF16 完整模型。

| 顺序 | 模型 | 是否必做 |
| --- | --- | --- |
| 1 | 预训练模型（不加适配器） | 必做 |
| 2 | `mixed` SFT | 必做 |
| 3 | `rl_nk` | 可选，用于隔离前缀奖励 |
| — | `rl_nk_prefix` | 已有，直接复用 |

### 3.2 逐样本结果

配对比较需要逐样本的命中。若官方脚本已保存逐样本预测，从中计算；否则从批次缓存中解析，解析规则与官方评分代码一致，并用聚合结果反向核对（逐样本平均必须与官方报告的 pass@64、recall@64 一致到小数点后 6 位）。无法得到逐样本结果时，只报告聚合数值并注明不能做配对区间。

### 3.3 统计

输出 `reports/checks/official_baselines.json`：各模型的 pass@32、recall@32、pass@64、recall@64、pid_pass@64、pid_recall@64、sid_validity@64；`rl_nk_prefix` 相对预训练和相对 `mixed` 的差值，以及按样本做 1000 次 bootstrap 的 95% 区间。

### 3.4 判断规则

以 pid_recall@64（真实商品 ID）和 recall@64（SID）为主指标，两者分别报告：

- 若 `rl_nk_prefix − mixed` 的区间下限大于 0，记为“官方独立数据上复现了提升”。
- 若区间包含 0 但点估计为正，记为“方向一致，未达显著”。
- 若点估计不为正，记为“未在官方数据上复现”，并与本地结果并列说明两套协议的差异（Race 混合候选、多目标标签、真实商品 ID 计分）。

## 4 Q3：有效性分解（纯 CPU）

### 4.1 输入

已有主评测与第 2 节新增的原始预测：`mixed`、`rl_nk`、`rl_nk_prefix` 的两种模式。解析规则与评测一致，复用 `evaluate.py` 中的同一解析函数，不另写一套。

### 4.2 指标

输出 `reports/checks/validity_decomposition.json`，每个模型与模式、每个 offset 都计算：

- 无效尝试的组成：目录外 SID、格式错误、思考未闭合、其他。
- 有效尝试中的命中率：命中尝试数 ÷ 有效尝试数。
- 每条样本的不同有效 SID 数。
- **等有效预算的 Recall**：每条样本只取前 k 个有效尝试计算 Recall，k = 8 和 16。只统计两个对比模型在 64 次中都至少有 16 个有效尝试的样本，并报告这部分样本的数量。

### 4.3 判断规则

以 `rl_nk_prefix − mixed` 为对象，对 offset 合并后的结果按样本做 bootstrap：

- 若等有效预算的 Recall@16 差值区间下限大于 0，记为“收益不只来自有效率，推荐本身更准”。
- 若该差值区间包含 0，而原始 Recall@32 的差值显著，记为“收益主要来自减少目录外 SID”。
- 两种情况都在报告中给出有效率与有效命中率各自的变化。

## 5 可选项

只有第 2 到 4 节完成后，且用户确认，才执行：

1. **约束解码对照**：用 Transformers 后端和 SID 前缀树约束，只在 non-thinking 模式下对 `mixed` 与 `rl_nk_prefix` 各评测 offset 0 到 4。若约束后的 `mixed` 追平 `rl_nk_prefix`，说明 RL 的收益可以被约束解码替代。两个模型必须使用同一后端，不与 vLLM 结果混合比较。
2. **补种子 43**：只训练 `mixed` SFT 和 `rl_nk_prefix`，设置与 seed 42 相同，按验证集选择后统一测试。显存与原训练相同，需要整卡。

## 6 交付与同步

1. `reports/checks/CHECKS_REPORT.md`：三个问题的结论、按预设规则给出的判定、全部表格，以及每个数字对应的产物路径。
2. `reports/checks/` 下的 protocol、summary 和统计文件；`reports/checks/gpu_before.json`、`runtime_identity.json`。
3. 不修改 `RESUME.md` 和 `PROJECT_REPORT.md`；只在 `PROJECT_REPORT.md` 末尾追加一行指向 `checks/CHECKS_REPORT.md` 的链接。
4. 完成后同步代码与 `reports/checks/` 到本机 `D:\P_project\minionereason`，原始预测留在服务器。

## 7 执行顺序

精简版顺序：

| 阶段 | 内容 | 依赖 |
| --- | --- | --- |
| K0 | 环境与 GPU 检查，写入 protocol.json（注明采用第 0.1 节精简范围） | 无 |
| K1 | M1：已有预测的有效性分解 | K0 |
| K2 | M2：non-thinking 8 次重评，然后把新预测并入 M1 重算 | K0 |
| K3 | M3：官方评测补 `mixed` | K0，不得与 K2 并行占用 GPU |
| K4 | 报告与同步 | K1 到 K3 |

可选项若经用户确认再运行，按第 2 到 5 节原有规则执行，单独追加到报告中。
