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
- worker-1：GPU 0 是本项目 SFT 服务，1/2/5/6/7 空闲，3/4 有其他项目进程。这是检查时快照，每次启动需重查。
- 六卡 RL 候选集合是 worker-1 的 `0,1,2,5,6,7`，启动前先停止自己的 SFT 服务；配置分配 2 卡 rollout、4 卡 actor/ref。actor/ref 按上游共用训练卡，正式全长 RL 的显存峰值仍需实测。
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
| `runs/benchmarks/osworld-sft__scienceboard169-tcg-v5` | worker-1 | 169 | 8（由 4 扩容） |

共同使用 worker-1 GPU 0 的冻结 SFT 推理服务：上下文 65,536、20 图容量、最多 8 请求。OSWorld 显式 top-p=1；ScienceBoard 上游漏传 top-p，因此本轮通过服务默认值固定 top-p=0.5、top-k=20，并在每题 manifest 中记录。ScienceBoard 专用 SFT 已下载，但它是另外一个对照臂，尚未启动全量队列。

v1/v2 在尚无完成分数时因初始化检查问题停止，保留作诊断并排除出基线。OSWorld 沿用已验证的 v3；ScienceBoard 的 v3 发现 KAlgebra 初始化段错误后保留为诊断，正式 v4 改用 guest socket 表检查监听状态并等待 10 秒，避免启动时读取应用内部数据。此模式已通过 4 步 KAlgebra 单题检查。ChimeraX 的判分仍通过上游 guest 5000 端口代理访问 loopback REST 服务。每个正式队列保存独立 runtime 副本与 SHA256，后续脚本编辑不会混入正在运行的评测。

2026-11-20 审计：OSWorld 首轮获得 352/361 个有效结果、168 个完全成功，9 题待处理；当前仅重试其中 7 个模型尚未采取动作的初始化/启动失败题。已有有效零分不会重跑，另 2 个执行过动作的评分异常保留待审计。完整分母上的正式指标仍未产生。

ScienceBoard v4 在 12 个有效结果后因连续异常暂停，已保留为诊断数据并排除出基线。`scripts/scienceboard_compat.py` 修复了发布版任务接入中的以下问题：

- KAlgebra `/tab` 需要 JSON 整数，`/add/2d`、`/add/3d` 需要表达式文本；原 controller 发送了对象。现在检查响应 `OK`，格式参照 [ScienceBoard 原始客户端](https://github.com/OS-Copilot/ScienceBoard/blob/main/sci/KAlgebra/kalgebra.py)。
- 26 个 Lean 任务的 theorem 初始化改为追加，保留前一步 `import Mathlib`；其中 12 题删除误插入的空 `chimerax_open`。原始配置、实际配置和逐项变更保存在每个 attempt 的 `task-compatibility/`，instruction 与 evaluator 不变。
- GrassGIS 在模型关闭 Map Display 后，`/dump` 越界返回 500。只在 VM overlay 内修复此状态序列化：工具栏 `shown=null`，因此不会满足“隐藏工具栏”的 `shown=false` 条件。窗口仍在但工具栏隐藏时沿用原判分。补丁前后 SHA256 保存在 VM 目录；只读基础镜像不变。另等待 GUI 完成初始化后再交给模型。

v5 对全部 169 题使用同一修复版本重新评测，不与 v4 拼接分数。修复针对任务初始化和状态读取，没有修改模型输入指令、任务目标或 GRPO 算法。它是明确记录的本地兼容协议，不能直接声称与论文运行环境完全一致。

2026-11-20 09:50 UTC，ScienceBoard v5 已从 4 扩到 8 并发，仍使用同一张 H200。8 VM 验收在约 98.64 秒全部就绪，截图与控制请求的中位耗时约 2.45 秒。迁移保留了之前 30 个有效结果，4 条在途轨迹自然结束后才更换调度器；没有重复模型采样。切换时另有 1 条 GrassGIS 评分连接异常保留待审计。OSWorld 的 PPTX 下载截断已通过读取原 episode 持久化文件并调用原 verifier 补判为 0 分；现有 359/361 个有效结果、169 个完全成功，仍有 2 题待审计。

并发扩容：`check_vm_concurrency.py --report <path>` 可保存独立验收报告。超过 4 并发的队列必须通过 `--concurrency-report` 提供同一节点、VM 镜像、vCPU 和内存配置的验收记录；当前支持上限为 8。并发数、验收报告及其 SHA256 写入 `campaign.json` 的 `launch_history`，每题仍调用原队列冻结的 runtime。并发改变可能影响 GUI 时序，报告须披露切换时间，不能宣称执行条件完全相同。

`resize_benchmark.py --run-dir <campaign> --workers 8 --concurrency-report <report>` 可在原节点等待已启动的题目自然完成后更换调度器，保留所有有效结果，包括零分；不重跑已评分题目。迁移记录保存为队列内的 `resize-*.json`。旧调度器不支持排空时，只临时暂停调度进程，独立评测子进程继续运行；等待超时或异常会恢复原调度器。新版调度器也支持 `SIGUSR1` 停止领取新任务并等待当前任务完成。

当前结果入口为 `reports/sft-benchmark-results.json` / `.csv`，运行期间每 30 秒更新；每个队列的 `summary.json` 和 `campaign.json` 保留分应用统计、逐题状态和全部尝试。总体成功率定义为 score=1 的任务数 / 完整题数；平均分定义为 verifier score 之和 / 完整题数。未全部获得有效结果时，两项正式指标为 null，不把缺题、超时或判分传输故障静默算成 0。连续三个异常会暂停队列供检查；有效 score=0 不会被续跑机制重试。异常是否由模型关闭应用引起仍需结合轨迹审计，不能一概认定为平台故障。

```bash
# 查看当前统计（不启动或修改评测）
python scripts/summarize_benchmarks.py

# 同节点续跑：必须沿用原协议；只有显式加 --retry-errors 才重试异常题
python scripts/worker_exec.py worker-0 -- /aigc/aigc_research/wangjing/ENV/cua-rl-osworld/bin/python /aigc/aigc_research/wangjing/CODE/CUA-RL/scripts/run_benchmark.py start --benchmark osworld --checkpoint osworld-sft --workers 4 --proxy-policy direct --run-dir /aigc/aigc_research/wangjing/CODE/CUA-RL/runs/benchmarks/osworld-sft__osworld361-tcg-v3 --resume
```

### 研究划分与评测含义

实测 2,049 个公开 RL 任务有 359 个基础 `id`、2,049 个独立 `new_id`；全部基础 `id` 都出现在 OSWorld 361 任务清单中，有 1 个 instruction 与该清单任务完全相同。这说明发布的训练池沿用 benchmark 基础任务；仅凭 ID 重合不能把所有生成任务认定为相同任务，但不能将官方评测描述为未见基础任务泛化。详见 `reports/task-protocol-audit.json`。

`configs/research-splits/` 额外提供 1,691 / 181 / 177 个任务的 train/dev/test 列表，先在全局按基础 `id` 分组，再按应用分层；同一基础任务的所有变体只落在一个集合。它用于后续独立算法开发，**未替换**官方全量复现池，也未自动写入 controller。实际启用时让各 worker 的 `examples_path` 读取 `*-examples.json`（按应用分组的映射），并更换注册任务名；`*-indices.json` 用于客户端过滤。原 loader 已验证实际加载 1,691 / 181 / 177 个任务，见 `reports/research-split-loader.json`。不能在全量复现 checkpoint 上再宣称该划分未见；严格实验需要各方法从相同 SFT 重新开始，并披露 SFT 本身的数据可见性限制。

## 当前未满足的启动条件

1. **原 agent 缺失**：固定上游代码引用 `mm_agents.anthropic_rl.main.AnthropicRLAgent`，公开目录没有该模块。本地显式新增 `scalecua_osworld` 选项接入已发布 `ScaleCUAOSWorldAgent`，用于连通性验证；不能据此声称完全相同的 RL rollout 协议。正式精确复现需补齐原模块或确认协议等价。
2. **VM 与训练运行时**：已在 worker-1 编译启动过 controller（18520）与桌面 worker（18521），从公开 Ubuntu qcow2 启动真实 QEMU VM。两台宿主机已有 KVM，但当前容器未获设备访问权限：即使创建 `10:232` 节点、赋予普通用户读写权限，open 仍返回 EPERM。需要平台分配设备，详见 `reports/kvm-platform-request.md`。此前检查使用 TCG 软件模拟、每轮独立 overlay、最多 2 个 VM；这不适合全池 RL 吞吐复现。`qemu_runtime.py` 现已加入显式 KVM 模式，`check_vm_concurrency.py` 提供 1→2→4→8→16 VM 的分档验收；目前仅验证设备缺失时正确阻止启动，尚未实测 KVM 吞吐。训练环境 `/ENV/cua-rl-agentrl` 使用 Python 3.12.16、Torch 2.10/cu130、vLLM 0.19、Megatron 0.17、TE 2.12；已通过 CUDA kernel 与实际 8B optimizer 检查。Transformers 4.57.6、Swift 4.1.0 等版本适配，以及 wheel / TE 版本解析修补见 `reports/runtime-adaptations.json`。CPU 桌面环境 `/ENV/cua-rl-osworld` 独立安装。训练 Python 已复制到共享缓存，避免依赖 worker-1 本地 home 目录。两者 `pip check` 均通过。
3. **Frontier 初始成功率**：官方要求选定的每个公开 RL 任务做 4 次 SFT rollout。全池是 8,196 个 episode，结果写入 `configs/initial_pass_rate.osworld-rl.json`。该文件尚不存在；smoke 可以关闭 Frontier，正式 baseline 不能省略这一步后称完全复现。

`configs/rl-indices.json` 是全池 2,049 个任务的显式客户端清单。完成四次独立 SFT 预跑后，可用 `python scripts/build_initial_rates.py <SFT结果.jsonl>` 从原版 `server_agent.py` 聚合结果生成成功率；工具会拒绝漏跑、重复记录、错误结果和混入其他模型的文件。它验证结果覆盖与格式，运行协议、终止状态和基础设施异常仍应结合轨迹核对。当前未生成或伪造初始化成功率。

`configs/qwen3vl-osworld-local-1step.yaml` 是额外的 6 卡连通性配置：batch=1、n=2、1 次更新、15 步上限、关闭 Frontier、使用本地 eval-agent 适配器；与正式 baseline 分开记录。原版 baseline 仍会被 `preflight_rl.py` 的 agent 缺失和 Frontier 检查阻止。

该 15 步检查的两个 reward 均为 `-0.2`，因此虽然链路通过，梯度为零，不能算有效学习。随后使用 `configs/qwen3vl-osworld-local-signal-1step.yaml`，恢复 50 步上限和 30,000 训练长度，采集四条轨迹，得到 `[1, -0.2, 1, 0]`，完成了非零梯度更新。四条中两条成功只是同一任务上的采样结果，不是 OSWorld 总分，也不能证明更新后的策略更强。本次 step 约 1,276 秒，其中采样与环境约 1,226 秒、reference 前向约 18.6 秒、actor 反传及 step 约 31.0 秒。TCG 是当前主要吞吐限制。

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
