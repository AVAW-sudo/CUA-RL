# CUA-RL：THUDM SCALE-CUA 复现与算法实验

目标是从 **THUDM 的 Qwen3-VL-8B OSWorld SFT checkpoint** 出发，先复现其 AgentRL 原版 RL，再比较算法改动。上游是 [THUDM/SCALE-CUA](https://github.com/THUDM/SCALE-CUA)，固定提交 `3929e2fe364623153f2caa94ead71dc1aea50fb0`。

## 已完成的准备（2026-11-20 UTC）

- 原仓库已放到 `SCALE-CUA/`，包含 `scalecua_rl/`、`osworld_env/`、`osworld_eval/`、`VeriGen/`。当前分支 `cua-rl/local-bringup`；本地兼容性修改保存在 `patches/local-bringup.patch`，未改 GRPO 数学实现。
- SFT 权重已下载，Hub 文件大小、safetensors 头和权重索引校验通过：`reports/model_manifest.json`。
- 已下载 OSWorld 的 **2,049 个公开 RL 任务和 29 个 verifier 相关文件**，记录文件 SHA256：`reports/training-data-manifest.json`。
- 正确的 8B SFT 服务已验证；运行 6 卡 RL 检查时会临时停止以释放 GPU 0。使用官方 `ScaleCUAOSWorldAgent` 完成两轮截图、工具调用、相对坐标和页面交互检查：`reports/smoke-thudm-sft.json`。合成检查之外，现已完成一个公开 OSWorld 任务的 15 轮真实桌面检查，verifier 返回 0，任务未成功；这是接口验证，不能当作 benchmark 分数。详见 `reports/smoke-osworld.json`。
- 已生成 6 卡 smoke、6 卡 baseline、16 卡 baseline 配置以及相对上游的配置差异。专用训练与桌面环境已安装并通过 `pip check`；4 卡真实 8B 多模态 optimizer 检查通过。**6 卡在线原版 GRPO 已完成一次非零梯度的 optimizer step 和参数同步**：四条真实轨迹、50 步上限、24 个视觉分段，最长 17,297 tokens，梯度范数 `1.7613805205`。详见 `reports/smoke-online-rl-signal.json`。这仍是单任务 smoke，没有保存 checkpoint，也未完成全池训练或 benchmark。
- 之前 OpenGVLab 7B 的准备材料已归档至 `archive/opengvlab-prep/`，其服务已停止。该目录不是本项目实验结果。此前下载的第三方代码和权重保留。

## 代码与资源

```text
CUA-RL/
  SCALE-CUA/                   # 独立 Git checkout，THUDM 原代码
    scalecua_rl/               # AgentRL：Ray + vLLM + Megatron
    osworld_env/               # controller、worker、Docker/K8s VM
    osworld_eval/              # 官方 benchmark 和 agent
    VeriGen/                  # 任务生成；首轮直接用发布的数据
  configs/                    # 本机路径、版本锁、训练配置、研究划分
  scripts/                    # 下载、配置生成、GPU 检查和 smoke
  reports/                    # 校验、硬件、配置差异、可运行性记录
  data/ScaleCUA/              # 固定 revision 的公开 OSWorld RL 数据
  runs/                       # 推理日志、截图、以后训练的输出
  archive/opengvlab-prep/      # 早先误认项目时的历史材料
```

外层仓库记录适配脚本与版本锁，`SCALE-CUA/`、`data/`、模型和运行日志不嵌入外层 Git。上游和数据各自有 Git 历史；后续算法改动可直接在 `SCALE-CUA/` 的实验分支中进行。适配代码的目标仓库为 https://github.com/AVAW-sudo/CUA-RL；`reports/` 和 `archive/` 是本地记录，不随代码发布。下文报告路径均指运行后生成的本地文件。

| 用途 | 模型 / revision |
| --- | --- |
| SFT 起点，已下载 | `extreme1228/ScaleCUA-qwen3-vl-osworld-sft` @ `406fe37598dc77b452bee1904c43411fbb6c5fad` |
| 已发布 RL 参考，已下载并校验 | `extreme1228/ScaleCUA-qwen3-vl-osworld` @ `d27cdab2d500f323af27c0c9fdcf123f40c83d42` |
| ScienceBoard 专用 SFT 对照，已下载并校验 | `extreme1228/ScaleCUA-qwen3-vl-scienceboard-sft` @ `48c39bb30e1920e02c440fc256f5bb4e1d44765b` |
| RL 数据，已下载 OSWorld 子集 | `extreme1228/ScaleCUA` @ `ca4266e614883a8bed5b8e9caf4f0fcbccb6a023` |

SFT 存放在 `/aigc/aigc_research/wangjing/MODEL/THUDM-ScaleCUA-qwen3-vl-osworld-sft`，约 17.55 GB。其架构是 `Qwen3VLForConditionalGeneration`。普通 `Qwen3-VL-8B-Instruct` 不能替代这个 SFT 起点。发布版 RL 参考已另外下载至 `/aigc/aigc_research/wangjing/MODEL/THUDM-ScaleCUA-qwen3-vl-osworld-rl-reference`，约 17.55 GB，校验见 `reports/released-rl-model-manifest.json`，尚未评测。

推理地址：`http://research-a-worker-1.job-research-a.training.svc.cluster.local:18080/v1`，模型名 `scalecua-qwen3-vl-8b-sft`，本地服务不验证 API key（客户端可填 `EMPTY`）。SFT 服务在在线训练检查后恢复至 GPU 0；状态记录在 `reports/inference-service.json`。当前评测服务采用 TP=1、上下文上限 65,536、最多 20 张图、vLLM 0.11.0；这不是原版 RL 的部署配置。

## 硬件与当前安排

两节点各有 8×H200（每卡 143,771 MiB）、NV18 互联、约 180 核 CPU 配额和 1,800 GiB 内存配额。仅 worker-0/1 用于计算。节点内 GPU 1/2 的 NCCL 正确性检查已通过；跨节点 NCCL 尚未测。

- worker-0：8 卡都有其他进程，不能启动本项目训练。
- worker-1：按用户最新授权使用全部 8 张 H200。2026-11-20 已逐卡确认空闲后启动 8 个 TP=1 推理副本：GPU 0–2 为 OSWorld SFT，GPU 3–7 为 ScienceBoard SFT。其他项目进程未被停止。
- 六卡 RL 配置保留原有的 2 卡 rollout、4 卡 actor/ref 分配；当前全部 GPU 用于冻结 SFT 评测，未继续训练。后续 RL 启动前重新分配自己的服务与空闲卡，正式全长训练的显存峰值仍需实测。
- 16 卡配置只供将来两节点全部可用时使用：4 卡 rollout、12 卡 actor/ref。当前没有启动或预约这些 GPU。

## 原版 baseline 与 benchmark

当前优先事项已调整为：**先测冻结的 SFT checkpoint，再继续 RL**。此前在线
GRPO 检查的临时权重未保存，当前服务仍加载下载的 OSWorld SFT。
KVM 设备探测与平台开放要求见 [kvm-platform-request.md](reports/kvm-platform-request.md)。

联系不到平台维护者时，**继续使用 TCG 软件模拟推进评测和小规模 RL**。
已在 worker-0 实测 4 个独立 VM 并发启动和控制：约 199 秒全部就绪，
截图与 guest 执行请求组合的中位耗时约 4.03 秒，全部清理成功，见
`reports/tcg-concurrency-worker-0.json`。这不是完整 rollout 吞吐测试。
短时 CPU 配额采样显示有余量，见 `reports/tcg-worker-capacity.json`；它不等同于资源预留。
后续可用 worker-0 的 CPU 跑环境、worker-1 的 GPU 跑模型；仍只使用这两台节点。
KVM 是吞吐优化项，不再作为开始 SFT 评测的硬性前提。软件模拟可能改变 GUI 响应
和超时行为，因此各实验臂需使用相同后端与等待策略，并标明 TCG 运行条件。

原版配置源：`SCALE-CUA/scalecua_rl/examples/training/configs/qw3vl_train_std.yaml`。

保留的主要设置：GRPO、每任务 8 个 rollout、每 batch 16 个任务、学习率 `3e-6`、冻结 ViT、TP=2、50 个环境决策上限、温度 0.8、KL 系数 `1e-4`、Frontier Sampling、截图保留参数 `images_to_keep=5` / `min_removal_threshold=5`。截图按阈值成批移除，并非任意时刻恰好 5 张。训练最大序列长度和 rollout server 上下文上限是两个不同参数，分别保持上游 30,000 / 200,000。

| 配置 | 用途与差异 |
| --- | --- |
| `configs/qwen3vl-osworld-smoke-6gpu.yaml` | 两次更新的预备调试配置；batch=2、n=4、并发=8，关闭 Frontier，缩短服务上下文；不作为算法结果 |
| `configs/qwen3vl-osworld-baseline-6gpu.yaml` | 保留原版算法设置；适配 6 卡分配和并发=16；需四次 SFT 预跑生成初始成功率 |
| `configs/qwen3vl-osworld-baseline-16gpu.yaml` | 将来 16 卡时使用；并发=32；仍需资源及跨节点通信验证 |

全部配置差异记录在 `reports/config-differences.json`。并发改变可能影响异步训练的策略滞后，因此这里是资源适配后的原算法基线；不能声称逐项等价复现论文的吞吐或分数。

本地修正了一个任务接入问题：controller 返回的 `-1` 是自定义任务 RPC 的保留索引，不能作为训练样本。训练 dataset 现在只排除该保留值，controller 仍保留自定义任务接口；真实任务集合不变。实际接口检查见 `reports/task-index-filter.json`。

评测任务列表已转换为 controller 可用格式，放在 `configs/benchmarks/`。主评测先固定官方 `test_nogdrive.json`：**361 任务，50 步，1280×720 模型截图，1920×1080 桌面，相对坐标，温度 0.8，top-p=1**。完整 `test_all.json` 有 369 任务，应单独报告。ScienceBoard 官方清单为 **169 任务**，使用独立 VM 和协议。计划先让同一个 OSWorld SFT 跑两套清单，再将官方 ScienceBoard SFT 作为域内对照单独报告，不能混为同一 checkpoint。两种 SFT 的 revision 已分别固定；下载不代表评测完成。配置中的 `val_tasks: [osworld-rl]` 是原训练池监控，不是独立 benchmark。

建议对照顺序：SFT 零更新评测 → 发布 RL 权重参考评测 → 从同一 SFT 重跑原版 RL → 同预算算法消融。先确认所用协议，不能直接把本地 smoke 或某个任务子集与论文分数比较。

ScienceBoard 的独立 SFT 权重和 44.21 GB VM 已下载并校验，见
`reports/scienceboard-sft-model-manifest.json`、`reports/scienceboard-vm-manifest.json`。
公开 ZIP 内的 `Ubuntu2.qcow2` 已按上游 subtype 映射为
`data/scienceboard-vm/Ubuntu2_scienceboard.qcow2`，镜像内容不变。TCG guest 启动、KAlgebra API 和正式 agent/verifier 的单题流程已通过：4 步、score=1、约 204 秒（包含启动），只是接口检查。
评测计划见 `configs/benchmarks/sft-evaluation-plan.json`；依赖审计发现 OSWorld-361
有 45 个 `proxy=true` 任务。本轮显式使用直连网络：guest 访问 Google 返回 200，但 worker 访问 Wikipedia 返回 403，不能假定所有站点可用。该网络协议与住宅代理配置有差异，网络/判分异常保留并单列，当前没有完整 benchmark 分数。

2026-11-19 开始同一 OSWorld SFT 的全量评测；11-20 OSWorld 继续原队列补跑，ScienceBoard 在工程修复后使用 v5 新队列。两套队列均为 50 步上限、每题一次采样、TCG、每个 VM 4 vCPU / 8 GiB，保留逐步 PNG 和 JSONL，关闭持续视频编码：

| 队列 | 节点 | 任务数 | 并发 |
| --- | --- | --- | --- |
| `runs/benchmarks/osworld-sft__osworld361-tcg-v3` | worker-0 | 361 | 4 |
| `runs/benchmarks/osworld-sft__scienceboard169-tcg-v5` | worker-1 | 169 | 12（由 4→8 扩容） |
| `runs/benchmarks/scienceboard-sft__scienceboard169-tcg-v5-pool` | worker-1 | 169 | 20 |

OSWorld SFT 使用 worker-1 GPU 0–2 的 3 个副本，ScienceBoard SFT 使用 GPU 3–7 的 5 个副本。每个副本 TP=1、上下文 65,536、20 图容量、最多 8 请求；两种 checkpoint 的分数独立报告。OSWorld 显式 top-p=1；ScienceBoard 上游漏传 top-p，因此本轮通过服务默认值固定 top-p=0.5、top-k=20，并在每题 manifest 中记录。

v1/v2 在尚无完成分数时因初始化检查问题停止，保留作诊断并排除出基线。OSWorld 沿用已验证的 v3；ScienceBoard 的 v3 发现 KAlgebra 初始化段错误后保留为诊断，正式 v4 改用 guest socket 表检查监听状态并等待 10 秒，避免启动时读取应用内部数据。此模式已通过 4 步 KAlgebra 单题检查。ChimeraX 的判分仍通过上游 guest 5000 端口代理访问 loopback REST 服务。每个正式队列保存独立 runtime 副本与 SHA256，后续脚本编辑不会混入正在运行的评测。

2026-11-20 最终 SFT 队列：OSWorld 359/361 有效、169 完全成功；OSWorld SFT 在 ScienceBoard 上 163/169 有效、27 完全成功；ScienceBoard SFT 在 ScienceBoard 上 162/169 有效、17 完全成功。缺测分别为 2、6、7；精确全分母成功率保留 null，报告展示缺测上下界，不将有效零分重跑。

ScienceBoard v4 在 12 个有效结果后因连续异常暂停，已保留为诊断数据并排除出基线。`scripts/scienceboard_compat.py` 修复了发布版任务接入中的以下问题：

- KAlgebra `/tab` 需要 JSON 整数，`/add/2d`、`/add/3d` 需要表达式文本；原 controller 发送了对象。现在检查响应 `OK`，格式参照 [ScienceBoard 原始客户端](https://github.com/OS-Copilot/ScienceBoard/blob/main/sci/KAlgebra/kalgebra.py)。
- 26 个 Lean 任务的 theorem 初始化改为追加，保留前一步 `import Mathlib`；其中 12 题删除误插入的空 `chimerax_open`。原始配置、实际配置和逐项变更保存在每个 attempt 的 `task-compatibility/`，instruction 与 evaluator 不变。
- GrassGIS 在模型关闭 Map Display 后，`/dump` 越界返回 500。只在 VM overlay 内修复此状态序列化：工具栏 `shown=null`，因此不会满足“隐藏工具栏”的 `shown=false` 条件。窗口仍在但工具栏隐藏时沿用原判分。补丁前后 SHA256 保存在 VM 目录；只读基础镜像不变。另等待 GUI 完成初始化后再交给模型。

v5 对全部 169 题使用同一修复版本重新评测，不与 v4 拼接分数。修复针对任务初始化和状态读取，没有修改模型输入指令、任务目标或 GRPO 算法。它是明确记录的本地兼容协议，不能直接声称与论文运行环境完全一致。

2026-11-20 09:50 UTC，ScienceBoard v5 已从 4 扩到 8 并发，仍使用同一张 H200。8 VM 验收在约 98.64 秒全部就绪，截图与控制请求的中位耗时约 2.45 秒。迁移保留了之前 30 个有效结果，4 条在途轨迹自然结束后才更换调度器；没有重复模型采样。切换时另有 1 条 GrassGIS 评分连接异常保留待审计。OSWorld 的 PPTX 下载截断已通过读取原 episode 持久化文件并调用原 verifier 补判为 0 分；现有 359/361 个有效结果、169 个完全成功，仍有 2 题待审计。

2026-11-20 10:29 UTC，8 个推理副本就绪，两组 ScienceBoard 队列分别启动 12 / 20 并发。首次 32 和 24 VM 同时启动各有一个启动超时；保留失败报告后，改为间隔 1 秒启动，32 个 VM 在约 153.53 秒全部就绪，三轮并发截图/控制检查中位 2.49 秒、最大 3.36 秒，清理全部成功，见 `reports/tcg-scienceboard-32vm-paced-worker-1-20261120.json`。这是启动及 API 验收，持续 rollout 吞吐仍需单独测量，不能据此宣称黑屏启动问题已根治。OSWorld-SFT 队列排空后保留 55 个有效结果，并完成下面说明的路由迁移；原有 2 个异常未重试。ScienceBoard-SFT 队列独立从完整 169 题开始。

并发扩容：`check_vm_concurrency.py --report <path>` 可保存独立验收报告。超过 4 并发的队列必须通过 `--concurrency-report` 提供同一节点、VM 镜像、vCPU 和内存配置的验收记录；调度器支持上限为 32，但实际启动仍必须通过对应并发验收。并发数、验收报告及其 SHA256 写入 `campaign.json` 的 `launch_history`，每题仍调用原队列冻结的 runtime。并发改变可能影响 GUI 时序，报告须披露切换时间，不能宣称执行条件完全相同。

推理副本通过 `serve_model.py --replica r1 --port 18082 ...` 单独管理状态和日志。池文件格式为 `{"checkpoint":"osworld-sft","service_states":["/absolute/path/to/service-state.json", "..."]}`，交给 `run_benchmark.py --service-pool <pool.json>`。同池必须保持 checkpoint revision、权重目录、服务配置和兼容适配器一致；启动命令仅允许端口不同。每题记录选用副本及其状态文件 SHA256，服务状态在每次协调器启动时冻结。

旧队列先用 `SIGUSR1` 等待所有在途题目自然完成，然后以 `--resume --service-pool <pool.json> --upgrade-service-routing` 迁移。迁移前验证原 runtime 哈希；仅允许给评测 launcher 增加服务选择及来源记录，provider、QEMU 和 ScienceBoard 兼容层必须逐字节一致。旧 runtime、全部已评分题目和尝试历史保留；模型指令、采样参数和 verifier 不变。此项显式迁移不能用于夹带评测逻辑修改。

ScienceBoard SFT 发布文件把 `extra_special_tokens` 存为列表，而当前推理环境的 Transformers 4.x 要求该字段为命名 token 字典。`scripts/scienceboard_tokenizer_compat/sitecustomize.py` 仅在该 checkpoint 的推理进程内把列表映射到 `additional_special_tokens`；不修改权重、checkpoint 文件或 token ID。真实 tokenizer/processor 检查确认全部 13 个特殊 token ID 及中英文和图像标记编码与原始 tokenizer.json 一致，真实图片 API 请求通过。证据见 `reports/scienceboard-sft-tokenizer-compat.json` 和 `reports/scienceboard-sft-multimodal-smoke.json`，后者不计入 benchmark。

`resize_benchmark.py --run-dir <campaign> --workers 8 --concurrency-report <report>` 可在原节点等待已启动的题目自然完成后更换调度器，保留所有有效结果，包括零分；不重跑已评分题目。迁移记录保存为队列内的 `resize-*.json`。旧调度器不支持排空时，只临时暂停调度进程，独立评测子进程继续运行；等待超时或异常会恢复原调度器。新版调度器也支持 `SIGUSR1` 停止领取新任务并等待当前任务完成。

当前结果入口为 `reports/sft-benchmark-results.json` / `.csv`；每个队列的 `summary.json` 和 `campaign.json` 保留分应用统计、逐题状态和全部尝试。总体成功率定义为 score=1 的任务数 / 完整题数；平均分定义为 verifier score 之和 / 完整题数。未全部获得有效结果时，两项正式指标为 null，不把缺题、超时或判分传输故障静默算成 0。

新版协调器取消了“连续三个异常暂停整个队列”：仅对明确且零动作的初始化失败自动恢复，包含历史尝试在内最多 3 次，退避后排到其他任务后面。有效零分不重跑；有动作的异常保留原轨迹和 VM overlay 供补判；没有证据的异常不会自动重新采样。纯 `{"Error": ...}` 诊断行不算模型动作，其他或损坏的轨迹行保守地视作已有动作。副本健康检查失败时暂避该 endpoint，其他副本继续；监督进程只恢复本项目的服务，重启保持原 checkpoint 和推理参数。启动 VM 默认间隔 1 秒，避免同时启动冲击。异常由模型关闭应用引起的情况仍需审计。

```bash
# 查看当前统计（不启动或修改评测）
python scripts/summarize_benchmarks.py

# 同节点续跑：沿用原协议，自动恢复明确的零动作初始化失败
python scripts/worker_exec.py worker-0 -- /aigc/aigc_research/wangjing/ENV/cua-rl-osworld/bin/python /aigc/aigc_research/wangjing/CODE/CUA-RL/scripts/run_benchmark.py start --benchmark osworld --checkpoint osworld-sft --workers 4 --proxy-policy direct --run-dir /aigc/aigc_research/wangjing/CODE/CUA-RL/runs/benchmarks/osworld-sft__osworld361-tcg-v3 --resume
```

### 研究划分与评测含义

实测 2,049 个公开 RL 任务有 359 个基础 `id`、2,049 个独立 `new_id`；全部基础 `id` 都出现在 OSWorld 361 任务清单中，有 1 个 instruction 与该清单任务完全相同。这说明发布的训练池沿用 benchmark 基础任务；仅凭 ID 重合不能把所有生成任务认定为相同任务，但不能将官方评测描述为未见基础任务泛化。详见 `reports/task-protocol-audit.json`。

`configs/research-splits/` 额外提供 1,691 / 181 / 177 个任务的 train/dev/test 列表，先在全局按基础 `id` 分组，再按应用分层；同一基础任务的所有变体只落在一个集合。它用于后续独立算法开发，**未替换**官方全量复现池，也未自动写入 controller。实际启用时让各 worker 的 `examples_path` 读取 `*-examples.json`（按应用分组的映射），并更换注册任务名；`*-indices.json` 用于客户端过滤。原 loader 已验证实际加载 1,691 / 181 / 177 个任务，见 `reports/research-split-loader.json`。不能在全量复现 checkpoint 上再宣称该划分未见；严格实验需要各方法从相同 SFT 重新开始，并披露 SFT 本身的数据可见性限制。

### 完整任务池的 SFT→RL 配置（保留，当前未启用）

此节记录完整 2,049 题版本，当前自动运行的是文末的小规模 pilot。完整版本的 `scripts/pipeline_supervisor.py` 配置为 `configs/original-rl-pipeline.json`，状态及日志在 `runs/original-rl-pipeline/`。现有队列逐个排空后接入新协调器，保留冻结 runtime 和有效结果。监督进程定期检查队列、推理副本和阶段进程；副本按原配置最多重启 3 次，含启动宽限期。阶段异常最多自动恢复两次，错误与每次 receipt 均留档。阶段进程消失时保留独立命令与现场，按持久化回执恢复；仅存活子进程而缺少退出回执时不会重复启动或假定成功。

运行期间持续生成 `reports/sft-evaluation-report.html` 和 `.json`。任务全部尝试且有限恢复结束后，报告成为 `final_complete` 或 `final_with_unscored_cases`：后者列出所有缺测题和全分母上下界，精确总分继续为 null。缺测不会伪造为失败，也不会无限卡住后续准备。自动队列按以下次序执行，**未启用 TA-GRPO、critic 或 GAE 改造**：

1. 用固定 revision 的官方发布 RL 权重分别测 OSWorld 361、ScienceBoard 169，得到同环境参考。
2. 用原始 OSWorld SFT 对公开 2,049 个任务各做 4 次预跑，共 8,196 条，生成 Frontier 初始成功率。逐条持久化、保留终止状态；缺失结果时拒绝生成 rates 或关闭 Frontier。只重试尚未调用模型的初始化故障，最多 3 次。
3. 原版 GRPO + Frontier 首轮执行 25 个更新步，保存 checkpoint。配置 `qwen3vl-osworld-original-8gpu-25updates.yaml` 与完整配置仅停止边界不同。上游从 step=1 开始、循环 `< max_steps`，因此首轮 max_steps=26、完整配置 max_steps=1000；空 batch 跳步单独报告，不冒充有效梯度更新。保留 batch=16、n=8、lr=3e-6、50 轮、KL=1e-4、冻结 ViT、历史图像 5/5 和原损失。
4. 测重训 checkpoint，并生成 `reports/sft-and-original-rl-comparison.html` / `.json`。首轮是小预算复现里程碑；完整约 999 个更新步的配置另存，可在后续授权预算下用同一 run name 续跑。

训练使用 worker-1 的 8 张 H200：2 张 rollout、6 张 actor/reference；保存最近两个 checkpoint，并保存最终边界；即使最终 batch 为空，也保留此前已更新的 actor，真实有效更新数另行报告。训练配置、模型权重、Frontier rates 和上游 commit 均记录 hash。公开源码缺少 `anthropic_rl`，当前显式使用发布的 `scalecua_osworld` agent，且环境为 TCG/direct；因此属于记录了这些差异的原算法复现，不能宣称与论文逐项完全等价。

默认每节点 4 vCPU / 8 GiB 的 VM，worker-1 为 32 路。可用 worker-0 CPU 增加 16 路，总计 48 路，**不分配 worker-0 GPU**；启用前要求同镜像/资源的并发验收及跨节点初始化、截图控制都通过，配置或证据不匹配则继续 32 路。`check_remote_worker.py` 是零 GPU 的工程检查，不计 benchmark 或 Frontier 成功率。当前已完成 worker-0 的 32 VM 启动/API 验收：约 215.35 秒全部就绪、请求中位 5.19 秒、全部清理；持续 rollout 吞吐须另测。实际选用容量写入阶段 provenance；collector 逐条记录初始化、推理和环境调用的耗时，供进一步定位瓶颈。

按已完成 359 条 OSWorld episode 平均 7.31 分钟做线性规划，32 路下 Frontier 约 31 小时、25 步采样约 12 小时；48 路理想约 21 / 8 小时。它们不含额外故障、评测和训练开销，也不代表 RL 任务的实测速率。原先四条轨迹的真实优化小跑中，1,276 秒/步有 1,226 秒用于采样与环境，reference 约 18.6 秒、actor 更新约 31 秒；采样计时包含模型推理，不能全部归为 VM 等待。主线保持原始 8B SFT 权重。

GitHub 环境调研及来源保存在 `reports/environment-acceleration-survey.json`。下一步优先比较同一镜像的 GPFS 与本地盘 I/O，再验证 QEMU 完整 RAM/设备/磁盘快照恢复；仅创建 qcow2 overlay 不会省去开机。当前正式队列保持冻结协议。可用于后续快速研究的 [Lite.ScaleCUA](https://github.com/cua-lite/cua-lite/blob/main/lite/gym/envs/lite/scalecua/README.md) 使用 VM-free 桌面，但有不支持任务的过滤规则且不迁移 ScienceBoard，不能把其筛选后的结果当成本轮完整基线。

## 当前未满足的启动条件

1. **原 agent 缺失**：固定上游代码引用 `mm_agents.anthropic_rl.main.AnthropicRLAgent`，公开目录没有该模块。本地显式新增 `scalecua_osworld` 选项接入已发布 `ScaleCUAOSWorldAgent`，用于连通性验证；不能据此声称完全相同的 RL rollout 协议。正式精确复现需补齐原模块或确认协议等价。
2. **VM 与训练运行时**：已在 worker-1 编译启动过 controller（18520）与桌面 worker（18521），从公开 Ubuntu qcow2 启动真实 QEMU VM。两台宿主机已有 KVM，但当前容器未获设备访问权限：即使创建 `10:232` 节点、赋予普通用户读写权限，open 仍返回 EPERM。需要平台分配设备，详见 `reports/kvm-platform-request.md`。此前检查使用 TCG 软件模拟、每轮独立 overlay、最多 2 个 VM；这不适合全池 RL 吞吐复现。`qemu_runtime.py` 现已加入显式 KVM 模式，`check_vm_concurrency.py` 提供 1→2→4→8→16 VM 的分档验收；目前仅验证设备缺失时正确阻止启动，尚未实测 KVM 吞吐。训练环境 `/ENV/cua-rl-agentrl` 使用 Python 3.12.16、Torch 2.10/cu130、vLLM 0.19、Megatron 0.17、TE 2.12；已通过 CUDA kernel 与实际 8B optimizer 检查。Transformers 4.57.6、Swift 4.1.0 等版本适配，以及 wheel / TE 版本解析修补见 `reports/runtime-adaptations.json`。CPU 桌面环境 `/ENV/cua-rl-osworld` 独立安装。训练 Python 已复制到共享缓存，避免依赖 worker-1 本地 home 目录。两者 `pip check` 均通过。
3. **Frontier 初始成功率**：官方要求选定的每个公开 RL 任务做 4 次 SFT rollout。全池是 8,196 个 episode，结果写入 `configs/initial_pass_rate.osworld-rl.json`。该文件尚不存在；smoke 可以关闭 Frontier，正式 baseline 不能省略这一步后称完全复现。

`configs/rl-indices.json` 是全池 2,049 个任务的显式客户端清单。完成四次独立 SFT 预跑后，可用 `python scripts/build_initial_rates.py <SFT结果.jsonl>` 从原版 `server_agent.py` 聚合结果生成成功率；工具会拒绝漏跑、重复记录、错误结果和混入其他模型的文件。它验证结果覆盖与格式，运行协议、终止状态和基础设施异常仍应结合轨迹核对。当前未生成或伪造初始化成功率。

`configs/qwen3vl-osworld-local-1step.yaml` 是额外的 6 卡连通性配置：batch=1、n=2、1 次更新、15 步上限、关闭 Frontier、使用本地 eval-agent 适配器；与正式 baseline 分开记录。未适配的原配置仍被缺失 `anthropic_rl` 的检查阻止；上述自动流程显式记录了公开 agent 替代，且必须先完成 Frontier rates 才能开始训练。

该 15 步检查的两个 reward 均为 `-0.2`，因此虽然链路通过，梯度为零，不能算有效学习。随后使用 `configs/qwen3vl-osworld-local-signal-1step.yaml`，恢复 50 步上限和 30,000 训练长度，采集四条轨迹，得到 `[1, -0.2, 1, 0]`，完成了非零梯度更新。四条中两条成功只是同一任务上的采样结果，不是 OSWorld 总分，也不能证明更新后的策略更强。本次 step 约 1,276 秒，其中采样与环境约 1,226 秒、reference 前向约 18.6 秒、actor 反传及 step 约 31.0 秒。采样与环境阶段占主要时间；该阶段同时包含模型推理和环境交互，不能把整段耗时都归因于 TCG。

检查结束后已停止本项目的 controller、桌面 worker 和所有 QEMU VM，释放六卡训练进程；SFT 服务单独恢复。复跑在线检查前，先启动 controller 与 osworld-worker，再停止自己的 SFT 服务。清理核验见 `reports/cleanup.json`。

## 常用命令

从 `CUA-RL/` 执行。远程命令通过现有 Kubernetes exec，只允许这两个 worker。

```bash
# 恢复固定版本源码、公开任务与本地适配补丁
python scripts/bootstrap_sources.py --data --local-patch

# 重新校验 / 下载 SFT（已有文件会复用）
python scripts/worker_exec.py worker-1 -- /aigc/aigc_research/wangjing/ENV/cua-rl-policy/bin/python /aigc/aigc_research/wangjing/CODE/CUA-RL/scripts/download_model.py

# 启停本项目 SFT 服务；start 会拒绝占用已有进程的 GPU
python scripts/worker_exec.py worker-1 -- python /aigc/aigc_research/wangjing/CODE/CUA-RL/scripts/serve_model.py stop
python scripts/worker_exec.py worker-1 -- python /aigc/aigc_research/wangjing/CODE/CUA-RL/scripts/serve_model.py start --gpus 0 --max-model-len 65536 --max-images 20 --max-num-seqs 8 --gpu-memory-utilization 0.75 --top-p-default 0.5 --top-k-default 20

# 官方 agent 的本地合成 GUI 检查（服务 ready 后）
python scripts/worker_exec.py worker-1 -- /aigc/aigc_research/wangjing/ENV/cua-rl-eval/bin/python /aigc/aigc_research/wangjing/CODE/CUA-RL/scripts/smoke_sft.py

# 改完 configs/paths.json 中的 controller_url 后，重新生成配置
/aigc/aigc_research/wangjing/ENV/cua-rl-policy/bin/python scripts/prepare_configs.py

# 只检查，不启动训练；未就绪时退出码 2
python scripts/worker_exec.py worker-1 -- /aigc/aigc_research/wangjing/ENV/cua-rl-policy/bin/python /aigc/aigc_research/wangjing/CODE/CUA-RL/scripts/preflight_rl.py
```

## 后续算法改动入口

先监控有效奖励组比例：时间加权乘以零 advantage 仍为零，critic 也不等于额外任务监督。正式基线保留 Frontier；小规模算法对照若关闭它，所有实验臂应使用同一任务序列和交互预算，并报告零方差组比例。另需独立控制视觉分段数量：当前上游在展开分段后按 `group_id` 归一化奖励，各分段携带同一终局奖励；不能将改变这一归一化方式的收益直接归因于时间信用分配。

- `scalecua_rl/trainer/src/agentrl/trainer/agentic/loops.py`：环境决策、截图移除和 trajectory segment 的边界。原实现给分段广播终局 reward；视觉分段本身不提供局部子目标奖励。
- `scalecua_rl/trainer/src/agentrl/trainer/algorithms/advantage.py`、`core_algos.py`：group-relative advantage 和 GAE 函数。
- `scalecua_rl/examples/training/async_trainer.py`：目前实际驱动 actor、reference、rollout；仅将 estimator 改成 `gae` 不会产生 critic 的 `values` 或训练 critic。

优先做一个不改变 actor 输入、分组和奖励的时间加权 GRPO 消融，以 β=1 校验退化行为，再评估 Gamer TA-GRPO 的完整分段/记忆适配。History-conditioned critic 路线需要新增多模态价值网络、环境步 GAE、终止/截断 bootstrap 和 action-token 映射；将“与 actor 相同历史的 critic”作为强基线，再测额外历史是否带来收益。两条路线都应固定初始权重、任务划分和交互预算，单独记录 critic 的额外算力。候选实验矩阵和预算已写入 `configs/experiment-matrix.json`，当前均未启动。Gamer 论文方法依据：[§4](https://arxiv.org/html/2611.18397v1#S4)。公开仓库目前的 [README](https://github.com/RainBowLuoCS/Gamer) 将 Streaming-vLLM 标为 coming soon；未确认有可直接复用的 TA-GRPO 训练代码。

## 本地基础设施与检查命令

所有下列脚本通过 `scripts/worker_exec.py worker-1 -- ...` 在 worker-1 执行，脚本路径使用绝对路径。运行日志在 `runs/`，检查结果在 `reports/`。

- `bootstrap_runtimes.py agentrl|osworld`：按锁文件恢复独立环境；需要基础源码已下载，不修改其他 conda 环境。
- `bootstrap_host_tools.py`、`build_controller.py`：提取私有 QEMU/Go/protoc 工具并编译 API controller。
- `infra_service.py controller start|stop`、`infra_service.py osworld-worker start|stop`：仅管理记录的本项目进程。
- `smoke_training_runtime.py`：真实 TE / FlashAttention 前向反向检查。
- `smoke_actor.py`：worker-1 GPU 1/2/5/6 上的 8B 多模态训练检查；会拒绝占用非空闲卡，不保存权重。
- `smoke_osworld.py`：SFT 服务启动后，跑一个公开任务并检查 verifier 是否返回数值；任务成功与基础设施通过分别记录。
- `evaluate_sft.py`：对指定的冻结 SFT 调用上游正式 benchmark runner，接入独立 QEMU overlay；支持显式任务子集，输出原始轨迹、判分及异常分类。新运行使用新目录，禁止覆盖已有结果；`run_benchmark.py` 提供完整队列和恢复，`summarize_benchmarks.py` 导出 JSON/CSV。ScienceBoard 分支要求服务至少支持 20 张历史图片，当前服务已满足。
- 单独调用评测默认保留上游持续录屏。`evaluate_sft.py --recording-mode steps-only` 可保留每步 PNG/JSONL 并关闭 guest 内持续视频编码；正式队列已统一采用该模式，各实验臂需固定同一模式。此前两题 OSWorld 接口检查使用原录屏方式，记录见 `reports/tcg-evaluation-progress.json`；正式队列进度见 `reports/sft-benchmark-results.json`。
- `check_vm_concurrency.py --accelerator tcg --stages 4 --cpus 2 --memory-mb 4096 --timeout 600`：无 KVM 权限的并发环境验收；默认不加 `--accelerator` 时仍检查 KVM，二者报告分开。
- `launch_local_rl.py`：停止本项目 SFT 服务后，检查 0/1/2/5/6/7 空闲并建立独立 Ray 集群，运行 `local-1step` 配置；不接管现有 Ray 集群。

要复跑四轨迹非零信号检查，给 launcher 加上 `--config configs/qwen3vl-osworld-local-signal-1step.yaml --report reports/smoke-online-rl-signal.json`。这会从原 SFT 重新开始，不续训本次临时权重。采样是随机的，新的批次不保证出现同样的奖励分布。

VM 来源：`xlangai/ubuntu_osworld` @ `a5d9c3eaae98eebf6e3a0beb84e7e47cf72ae133`，`Ubuntu.qcow2.zip` 解压至 `data/osworld-vm/Ubuntu.qcow2`。上游 `osworld_env/README.md` 明确要求将该公开磁盘重命名或链接为 `Ubuntu-modified.qcow2`；本地适配器采用同一文件名映射。当前运行使用 TCG 软件模拟；将来有设备权限时可显式选择 KVM。


## 环境加速实测（2026-11-20）

按用户授权，先在 worker-0 使用 CPU 做独立实验，再在两节点验证 16+32 路快照并发，环境实验额外 GPU 占用为 0。已完成的 SFT 结果和冻结评测协议保持不变；通过验收的快照后端仅启用于单独命名的小规模原版 GRPO pilot。汇总入口为 `reports/environment-acceleration-report.html` / `.json`，原始证据和失败尝试均保留。报告不将基础设施检查称为端到端训练加速。

- **同镜像磁盘对照**：`benchmark_environment_runtime.py` 将冻结 qcow2 复制到节点本地 `/tmp`，校验双方 SHA256，各镜像交替执行 3 轮共享盘 / 本地盘冷启动。OSWorld 已测共享盘冷启动中位 109.02 秒、本地盘 95.39 秒。主机页缓存未清空，其他作业未暂停；第二个诊断 VM 和用户态桌面测试的并行负载也已记录，不能当作隔离硬件测试。
- **完整快照**：同脚本在每个本地盘 VM 中保存 RAM、设备和磁盘状态，执行 3 次 `loadvm`，检查截图、控制、文件和鼠标状态。仅有 qcow2 overlay 不能跳过开机。`check_snapshot_task_reset.py` 进一步用原 task setup/verifier 检查冷启动与三次恢复的状态一致性；测试动作由固定脚本提供，结果不并入 SFT 分数。早期 OSWorld 脚本在终端尚未就绪时输入失败，改正等待后原 verifier 的 0→1 与恢复后归零均通过；失败记录保留。
- **无 KVM 用户态桌面**：隔离安装 udocker 1.3.17，使用 PRoot P1 执行 GHCR 的 Lite.OSWorld 镜像，固定 digest `sha256:2d51f12e1bff4a8621c3a810cd5399c2c678aafae0db310f757c8c240cf954b1`。三轮 Xvnc + GNOME + 测试窗口就绪约 10.57 秒；30 次 HTTP PNG 截图 + 新 Python 进程控制检查约 0.22 秒/次。LibreOffice Writer 的真实键鼠输入、保存 ODT 和内容校验另有通过记录。镜像下载与解包不计入启动时间；桌面配置与测试 HTTP 服务不同，不能将这个速度比直接当作原 benchmark 加速比。
- **现有轨迹剖析**：`profile_evaluation_logs.py` 只读有效结果日志，不重新调用模型。初次统计的 358 条 OSWorld 日志中，11,157 次环境动作中位 3.61 秒，10,799 个动作间隙中位 2.289 秒，启动+setup 中位 115.22 秒。动作间隙含推理和客户端处理，环境动作含等待和截图，均不是 GPU 单独耗时。两组 ScienceBoard 的快照统计和来源日志在同一报告中。

PRoot 不提供足够的隔离来托管任意模型生成的动作；本实验仅执行可信固定脚本，未挂入宿主 home、GPU 设备或系统 D-Bus，仅映射 `/proc` 和 null/zero/random 等最小设备。当前没有可用 Docker 服务；用户命名空间能创建，但实际 tmpfs 挂载失败。原生 GNOME 路线证明用户态执行可行，仍需解决隔离、应用版本和完整任务兼容性后才能考虑正式 rollout。桌面退出时收回其进程组，并以 subreaper 回收双重 fork 的子进程，避免长时间实验积累僵尸进程。

本次 pilot 已选择**同镜像 warm pool + 完整快照**。修复 guest 时钟回退后，16+32 VM 各 3 轮、共 144 次恢复通过；并发恢复中位为 9.66/12.06 秒。生产适配器的 OSWorld 原任务 setup/verifier 四轮也通过。其余应用和原模型吞吐在 pilot 中继续观察；这不等同于完整 benchmark 的后端等价性证明。假设每条轨迹节约 100 秒、48 环境持续满载，8,196 条 Frontier 预跑理论可省约 4.74 小时；这是重置开销的条件估算，不是已测端到端收益。

```bash
# 只读汇总现有证据
/aigc/aigc_research/wangjing/ENV/cua-rl-osworld/bin/python scripts/profile_evaluation_logs.py
/aigc/aigc_research/wangjing/ENV/cua-rl-osworld/bin/python scripts/write_environment_report.py

# 原镜像存储/快照对照：只允许 worker-0，要求新 report 路径
python scripts/worker_exec.py worker-0 -- /aigc/aigc_research/wangjing/ENV/cua-rl-osworld/bin/python /aigc/aigc_research/wangjing/CODE/CUA-RL/scripts/benchmark_environment_runtime.py --report /aigc/aigc_research/wangjing/CODE/CUA-RL/reports/environment-runtime-repeat.json

# 已准备的用户态镜像：脚本验证其 manifest，使用新输出目录
python scripts/worker_exec.py worker-0 -- /aigc/aigc_research/wangjing/ENV/cua-rl-osworld/bin/python /aigc/aigc_research/wangjing/CODE/CUA-RL/scripts/probe_userspace_desktop.py --udocker /tmp/cua-rl-udocker-1000/venv/bin/udocker --repository /tmp/cua-rl-udocker-1000/repository --container-id 01ffb0c6-2cd7-3d4d-8cd7-3d2fc6250b4a --gnome --office --repeats 3 --report /aigc/aigc_research/wangjing/CODE/CUA-RL/reports/userspace-repeat.json
```

GitHub 调研覆盖 13 个上游入口，记录在 `reports/environment-acceleration-survey.json`。最相关的是 [QEMU 完整快照](https://github.com/qemu/qemu/blob/master/docs/system/images.rst)、[ARPO 分布式环境](https://github.com/JIA-Lab-research/ARPO)、[Lite.OSWorld](https://github.com/cua-lite/cua-lite/blob/main/lite/gym/envs/lite/osworld/README.md)、[Lite.ScaleCUA](https://github.com/cua-lite/cua-lite/blob/main/lite/gym/envs/lite/scalecua/README.md) 和 [udocker](https://github.com/indigo-dc/udocker)。Lite.ScaleCUA 导入 2,049 条公开 RL 任务，但有不支持项筛选且未迁移 ScienceBoard/原训练器；不能用筛选后的分数替代原版 361/169 全集。常规本地 Firecracker 仍依赖 KVM；OpenEnv 等接口包装本身不消除 TCG 成本。

## 2026-11-20 小规模原版 GRPO 与无人值守恢复

当前自动队列使用 `configs/original-rl-pilot-pipeline.json`。完整 2,049 题配置
`configs/original-rl-pipeline.json` 保留但关闭，避免两个训练队列争用 worker-1。
本次是 **64 题的小规模 original-GRPO pilot**，不是完整论文复现：固定哈希、按应用轮转选题，
跨应用相同 seed 去重；每题 4 次 SFT Frontier 初始化，25 个更新机会。batch=16、n=8、
GRPO loss、Frontier、学习率、50 轮上限和视觉历史设置保持原版；每 5 步存 checkpoint。
有效梯度步数另行统计，全部零梯度不能记作有效学习成功。

训练后在预先固定的 OSWorld 30 题、ScienceBoard 20 题上测评，再测 released RL reference。
SFT 对照从已完成全量测评抽取相同题目，不重新挑采样。OSWorld 30 题 seed 与本次训练池不重叠；
不声称与 SFT 预训练数据隔离。选择清单、哈希及范围在 `configs/original-pilot/manifest.json`。

环境优先采用原始 OSWorld 镜像和完整 QEMU RAM/设备/磁盘快照；启用必须有同节点、同实现哈希、
匹配并发数的通过报告。初次启动冷开机，之后复用；每次租约有独立 ID，旧租约不能结束新任务。
恢复失败则销毁该 VM，下次冷启动。反复 `loadvm` 会让 guest 时钟回退，已增加任务 setup 前的
宿主机 epoch 校准。正式 SFT/训练后 benchmark 仍用其冻结的冷启动评测协议。

监督层不依赖聊天会话：`pipeline_service.py` 的独立 guard 重启退出的 watchdog，
watchdog 校验 PID、启动时间、boot ID、心跳和文件锁，只重启监督进程。
采样器/训练器由 `persistent_job.py` 独立记录进程和退出回执；阶段恢复时接管已有任务，
不会因父进程消失就重复拉起 GPU 训练。失败命令有界重试、训练从已保存 checkpoint 恢复；
不将缺失退出回执当成功，不给缺测 episode 编造分数。整个节点/所有监督层同时丢失仍不能等同于
进程级恢复，必须重新启动入口；已有共享存储产物保留。

```bash
# worker-1 上的幂等启动入口（无需 tmux；stdin 断开不影响运行）
python scripts/worker_exec.py worker-1 -- \
  /aigc/aigc_research/wangjing/ENV/cua-rl-osworld/bin/python \
  /aigc/aigc_research/wangjing/CODE/CUA-RL/scripts/pipeline_service.py

# 从共享目录查看监督状态和实际更新步数
cat runs/original-rl-pilot-pipeline/state.json
cat runs/original-rl-pilot-pipeline/watchdog.json
cat reports/original-grpo-progress.json
```

报告入口：`reports/sft-evaluation-report.html`（全量 SFT，保留缺测上下界）、
`reports/environment-acceleration-report.html`（环境实验，失败记录也保留）、
`reports/original-grpo-pilot-comparison.html`（固定子集对照）。
`reports/watchdog-recovery-validation.json` 记录杀掉监督进程后接管同一个底层命令、仅启动一次的实测。
用户态 PRoot/GNOME 仅用于可信脚本研究：浏览器、Writer、Calc 三轮验证不能代替原 benchmark
的隔离、应用兼容性和 verifier 等价性，暂不承载模型自由动作。
