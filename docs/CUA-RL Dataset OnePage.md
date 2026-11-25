# CUA-RL 数据集 OnePage · 用 256 道题训练 OSWorld 弱项

**决策提案 · 2026-11-25。** 在 8 × A100 上，建议从官方 ScaleCUA Qwen3-VL-8B OSWorld SFT 起跑，优先训练**跨来源读取并写入指定文件**和 **Calc 跨表查价、计算与汇总**。选题时检查每个官方 RL 变体是否保留了失分的关键操作。原 64 道 pilot 已预跑；本轮可复用其中 56 道。扩大后的候选池和实时覆盖见下文。[模型与数据版本](../configs/sources.lock.json) · [官方 RL 题库](https://github.com/THUDM/SCALE-CUA#dataset)

## 我们的 SFT 弱在哪里

官方 OSWorld 361 题中，358 题有分，平均 **44.32%**，含部分得分。计入 3 道未评分题后，361 题均分的上下界为 **43.96%–44.79%**。为避免用测试成绩选训练题，弱项诊断只看其中与 **RL 训练划分**对应的 295 个原始任务。下表是本地 SFT 实测。[SFT 审计记录（共享工作区）](../runs/ais-sft-evaluations/23495234/full/summary.json) · [划分依据](../configs/research-splits/manifest.json)

| 训练侧原题 | 已评分 / 原题 | 已评分均分 | 零分题 | 零分且记录 ≥50 个动作 |
|---|---:|---:|---:|---:|
| 全部 | 293 / 295 | 40.85% | 173 | 109 |
| 多应用 | 77 / 77 | **10.39%** | 69 | **62** |
| Calc | 37 / 37 | **37.84%** | 23 | 11 |
| Impress | 37 / 37 | 45.68% | 20 | 8 |

**同源变体的难度分化很大。** 64 道 pilot 中有 33 道对应零分原题，变体各跑四次，15 道全败、11 道有胜有负、**7 道全胜**。例如“检索 ACL 获奖论文并填表”原题得零分，同源的“表头居中”已能 4/4 成功。后者可练格式操作，无法用来衡量跨来源检索。[预跑分数（共享工作区）](../runs/ais-frontier-campaigns/23500435/verified/initial_pass_rates.json) · [官方表头变体](https://huggingface.co/datasets/extreme1228/ScaleCUA/blob/ca4266e614883a8bed5b8e9caf4f0fcbccb6a023/osworld/rl_tasks/multi_apps/67890eb6-6ce5-4c00-9e3d-fb4972699b06_traj_verify_4.json)

## 256 题如何入选

| 预算 | 入选条件 |
|---|---|
| **128 道弱项主练题** | 保留原题失分的关键操作，检查最终产物。例如从多个来源提取字段并写入正确的行列。 |
| **64 道弱项分步题** | 隔离一个可验证环节。例如 VLOOKUP 变体要求从价格表查价，判分器核对整列数值，只能证实结果正确。[官方查价变体](https://huggingface.co/datasets/extreme1228/ScaleCUA/blob/ca4266e614883a8bed5b8e9caf4f0fcbccb6a023/osworld/rl_tasks/libreoffice_calc/51719eea-10bc-4246-a428-ac7c433dd4b3_traj_verify_2.json) |
| **64 道广覆盖题** | 保留按应用预先抽取的 pilot，复用同协议的 256 条轨迹。能力回退用另一组留出题衡量。[Pilot 选题记录](../configs/original-pilot/manifest.json) |

从官方 2,049 道 OSWorld RL 变体中的 **train 1,691 道 / 295 个原始 UUID** 取题，跨应用优先一题一家族。每题记录失败证据、保留的操作和判分字段。用未操作初态、已知正确产物、关键字段出错的产物核验判分，再做 4 次 SFT 预跑初始化 Frontier。对关键的 0/4 题复核可解性后补采。若主练题不足 128 道，报告实际覆盖的能力和数量。[研究划分](../configs/research-splits/manifest.json) · [Frontier 方法](https://github.com/THUDM/SCALE-CUA#overview)

CUA-Lite 的结果说明为什么要检验留出表现。其 4B 模型在 Lite.ScaleCUA 题上做 RL 时留出分数没有提高，改用 LibreOffice 专项题后，117 道留出办公题均分从 32.8 升到 38.6。这支持我们先检验定向选题能否迁移，再决定是否扩大训练。[CUA-Lite 论文 §4.2](https://arxiv.org/html/2610.26071v1#S4.SS2)

## 首批 32 道候选的实测结果

从训练划分定向挑选 32 个原题 SFT 零分的家族，每家族一题，避开 pilot 的已有变体。逐题对照原题轨迹末段、变体要求与官方判分器，再把故意写错的结果送入判分。7 道入围题都用原始素材独立重算过，离线初态均为 0，正确产物均为 1，改错目标字段后均扣分。[逐题记录、分数与轨迹路径](<CUA-RL Dataset Audit 32.json>)

| 官方变体标签 | 主练入围 | 分步入围 | 待桌面复核 | 排除 |
|---|---:|---:|---:|---:|
| multi_apps · 16 题 | 0 | 1 | 1 | 14 |
| libreoffice_calc · 16 题 | 1 | 5 | 2 | 8 |

**选题时优先找“从哪些来源取值、写到哪些位置、评分检查哪些字段”。** 唯一入围的主练题标在 Calc 下，原题属于多应用。它要求比对两份 Word 答卷，把 10 个判断和总分写进指定学生的成绩行。用原文件重算得到 70 分，全部填对得 1，只改错总分得 10/11。反过来，一道标为多应用的姓名合并题只填对首行、其余 4,999 行全部写错，仍得 1。[逐题验证](<CUA-RL Dataset Audit 32.json>)

排除的 22 道中，17 道给关键错例满分，另有 3 道与留出集共用关键素材，2 道省掉了原题暴露的弱项。两道销售汇总题与 test 从同一 URL 下载 SalesRep.xlsx，一道资助汇总题与 dev 的源 PDF 字节相同。3 道待桌面复核的疑点分别是电影表头复制是否已经学会、工时公式经 Calc 保存后能否正确读回，以及三个等级单元格是否已做对而原题在其他位置失分。[错例及素材匹配依据](<CUA-RL Dataset Audit 32.json>)

此次使用已有轨迹和缓存素材，从开始检查到记完去留理由用时 **10 分 18 秒**，整批摊算约 **19 秒/题**。这覆盖本地检查；新题下载、真实桌面操作和排队另计。7 道入围题进入 Frontier 待跑清单，每题 4 次，共 28 条。只有 1 道主练题通过本批筛选，下一批可在全部训练应用中按输入和产物检索，再检验 128 道主练配额是否可行。[本批计时与入围清单](<CUA-RL Dataset Audit 32.json>)

## 八卡筛选进度与时间预算

**11 月 25 日晚的正式队列**从训练侧 1,691 个变体取样。去除留出集的同指令或共用素材、已知判分问题及不适合当前桌面的候选后，准备了 **535 条不同的官方指令**。其中 109 条已有冻结 SFT 的四次预跑，426 条排入新的单作业队列。原 64 道 pilot 中的 8 道与留出集共用素材，本轮保留 56 道。选题先找这版 SFT 在训练侧 OSWorld 原题上失分的技能，再读变体自身的指令，最后用变体自身的四次成绩估计难度。36 道原题满分但变体指令命中弱技能的题，单列为扩展候选。[当前候选池](../runs/dataset-screening/latest-overnight/prompt-pool.jsonl)

426 道里有 74 道的指令直接匹配跨来源读写、查表或表格计算，其余 352 道可补充其他弱家族和应用覆盖。指令匹配只产生待复核标签。例子是跨表查价并写回营收的变体实测 2/4，保留为主练；多应用家族的“填三个年份”实测 4/4，只提供一般表格操作的证据。人工复核参考原指令、正确产物和改错目标字段后的官方评分。原题得零分本身不能说明变体保留了同样的困难。[上一轮四次试跑](../runs/ais-dataset-screen/23527641/summary.json) · [已复核的弱技能](../configs/dataset-screen/reviewed-weak-skills.json)

过夜作业 23533722 使用 `bash AIS_ENTRY.txt screen-overnight`。每个新候选先以独立桌面执行未操作初态判分，得到零分后再做四次 SFT 预跑。8 个单卡 rank 共 16 路桌面。每五分钟导出一次 256 条暂选、已完成四次评分的子集、满足角色与判分复核的子集。上轮 48 题耗时约 96 分钟，47 题四次齐全。以该吞吐外推，新队列再完成 147 道四评分题所需的纯采样时间约 5 小时，426 道全池约 14 小时，另计排队、启动、初态检查和慢题。首份暂选有 109 道四评分、42 道兼具角色与判分复核，其中主练 3 道、分步 31 道、广覆盖 8 道。128 道主练配额的可达性仍待逐题验证。[最新数量与各配额缺口](../runs/dataset-screening/latest-overnight/prompt-status.json)

统一入口 `bash AIS_ENTRY.txt screen` 在单个作业里启动 8 个单卡 rank，每卡一个冻结的 SFT 服务和两个桌面。8 个分片各自复用已校验的节点本地权重和素材。每个桌面会话单独重置。用 `screen --plan` 查看分片，作业中断后可在同一批次的新作业上加 `--reuse-from 旧作业号`。每个 rank 的 `status.json` 记录进度，汇总在 `runs/ais-dataset-screen/作业号/summary.json`。满分按官方奖励等于 1 计算，部分分保留原值；缺试次的题显示未评分原因，只对四次齐全的题计算成功率。[AIS 入口](../AIS_ENTRY.txt)

- 扫描了训练侧 1,691 个变体，795 个进入初步候选池；按指令中的输入、处理和输出操作找到 143 个优先复核的变体，覆盖 64 个原题家族。[逐题索引（共享工作区）](../runs/dataset-screening/weak-skills-20261125/index.jsonl)
- 独立 Notebook 在跑首批 7 题 × 4 次。八卡第二批另选 16 个家族各一题，共 64 条。离线检查使用原始素材和官方 getter、metric：16 道的初态都得 0，正例都得 1，改错目标字段后均扣分。有 15 道根据素材或指令构造正例；另 1 道会议国家表使用官方答案，历史事实独立复核待做。复核中剔除了一道 Impress 题：正确移动表格并改标题仅得 0.5，改回错误标题仍得 0.5。[第二批题单](../configs/dataset-screen/batch-02.json) · [逐题对照（共享工作区）](../runs/dataset-screening/weak-skills-20261125/offline-batch-02.json)
- 默认作业把后续 **32 题**接在这 16 题之后。新增题覆盖 22 个原题家族，用原始输入、按指令编辑的文件、故意改错的文件分别调用官方 getter 和 metric，选入的 32 题均满足初态 0、正例 1、错例低于 1。各 rank 优先排前两题的四次试跑，随后继续取题，共用同一套桌面服务。48 道均为筛选候选，入选训练集仍需核验四次试跑结果和技能覆盖。[追加题单](../configs/dataset-screen/batch-03.json) · [逐题离线对照](../runs/dataset-screening/weak-skills-20261125/offline-batch-03.json)
- 默认共 192 条。按历史每条 443 秒、16 路并发、70%–85% 有效利用率，纯采样约 **1.7–2.1 小时**，含冷启动和慢分片预计 **2–3 小时**，排队另计。若原 64 题沿用，要给另外 192 道各跑四次，相同假设下仅采样约 **7–8.5 小时**。若预跑后仅一半候选最终入选，采样约 **14–17 小时**。用此次实测速率与入选比例更新预算。

## 实验与验收

**数据收益**用同一 SFT、原版 GRPO + Frontier 和交互预算，对比「按应用分层抽取 256 题」与「定向筛选 256 题」。**算法收益**在同一定向题单上比较原版 GRPO、时间加权消融和 Gamer TA-GRPO。首轮每步 **16 个 prompt × 8 个 rollout，25 次更新机会**。PPO 可沿用划分并单记 critic 开销。[GRPO 配置](../configs/qwen3vl-osworld-original-pilot-8gpu.yaml) · [Gamer 方法 §4](https://arxiv.org/html/2611.18397v1#S4)

**验收**用 dev 181 题选 checkpoint，冻结后在 test 177 题同时报整体均分和预先标记的弱项子集均分，按 32 个原始 UUID 做配对比较并报告区间。官方 OSWorld 361 单独报分、评分覆盖率及与训练原题的重叠数。295 个候选训练 UUID 都出现在官方 361 中，公开 SFT 的逐题暴露情况未知。[研究划分](../configs/research-splits/manifest.json) · [官方评测清单](https://github.com/THUDM/SCALE-CUA/blob/3929e2fe364623153f2caa94ead71dc1aea50fb0/osworld_eval/evaluation_examples/test_nogdrive.json)

<details>
<summary><strong>筛选复核与 8 卡预算</strong></summary>

- 每个候选保存原始 UUID、变体 ID、最后一个完成的操作、失败证据、目标能力、判分字段和选留理由。区分读错来源、写错位置、重复尝试、素材不可用和判分异常。环境异常修复后重测。复核快照可重置和素材可用。
- 已查到训练候选与 dev/test 有 14 条指令在统一大小写和空白后完全相同，涉及 7 种指令，原 pilot 命中 0 条。先剔除这些候选，再复核目标、素材与判分近似的题。现有划分的 train/dev/test 分别覆盖 295/32/32 个原始 UUID。[研究划分](../configs/research-splits/manifest.json)
- 原 64 题各试 4 次，28 题全败、18 题有胜有负、18 题全胜。复用同协议预跑后还需 **192 × 4 = 768 条**。已评分轨迹平均 443 秒，按 8 路预跑和 70%–85% 有效利用率估算，补跑约 **14–17 小时**。按 32 路桌面估算，每方法每种子的训练另留 **20–25 小时**，不含排队、清洗和评测。用实际完整训练步校准，并记录有效更新数、不同任务数、组内奖励有差异的比例、环境步数和 GPU 小时。[预跑审计（共享工作区）](../runs/ais-frontier-campaigns/23500435/verified/audit.json)

</details>

<details>
<summary><strong>GitHub 调研依据 · 哪些做法值得借鉴</strong></summary>

| 项目与场景 | 作者做法 | 对本项目的建议 |
|---|---|---|
| **ScaleCUA** · 桌面 | VeriGen 生成带可执行判分器的变体，Frontier 按逐题成功率采样，并保留一部分均匀探索。 | 先审核变体是否考目标能力，再用本机预跑率初始化 Frontier。保留原方法作为基线。[仓库](https://github.com/THUDM/SCALE-CUA#overview) · [官方数据](https://huggingface.co/datasets/extreme1228/ScaleCUA) |
| **CUA-Lite** · 最接近我们的算力 | 单机 8×A100。Qwen3.5-4B 在 814 道合成 LibreOffice 题上 RL，在 117 道官方办公题上的均分从 32.8 升至 38.6。作者此前在 Lite.ScaleCUA 的 RL 划分上没有观察到留出集提升，将 SFT 饱和列为一种解释。 | 聚焦可迁移的办公操作并设独立留出集。我们的 8B 起点、桌面环境和评测题不同，提升幅度要实测。[仓库](https://github.com/cua-lite/cua-lite) · [论文 §4.2](https://arxiv.org/html/2610.26071v1#S4.SS2) |
| **ARPO** · 最直接的选题先例 | 用 UI-TARS-1.5 对每道 OSWorld 题采样 16 次，保留至少成功一次的 128 题，再训练并回放成功轨迹。 | 我们每题先采 4 次。对确属弱项且判分可靠的 0/4 题追加样本，避免因预跑较少过早淘汰。[仓库](https://github.com/JIA-Lab-research/ARPO) · [论文 §3.4](https://arxiv.org/html/2505.16282v1#S3.SS4) |
| **DART-GUI、MobileRL** · 桌面／移动 | DART 按难度调整 rollout 数和轨迹长度，为难题预采成功示例。MobileRL 在移动端采用难度自适应正例回放和失败课程筛选。 | 区分“没有可靠判分”“当前很难成功”和“已有成功但很稀少”。追加采样、回放或长度奖励应作为单独实验。[DART-GUI](https://github.com/Computer-use-agents/dart-gui) · [DART 论文](https://arxiv.org/abs/2509.23866) · [MobileRL](https://github.com/THUDM/MobileRL) |
| **WebAgent-R1、DigiRL** · 网页／移动 | WebAgent-R1 用 647 题做 RL、165 道人工验证题做评测。DigiRL 提供基于奖励过滤的行为克隆对照。 | 将人工核验优先给独立评测题。若引入成功轨迹回放，可用成功轨迹行为克隆检查收益来源。[WebAgent-R1](https://github.com/weizhepei/WebAgent-R1) · [论文 §3.2](https://arxiv.org/html/2505.16421v1#S3.SS2) · [DigiRL](https://github.com/DigiRL-agent/digirl) |
| **OSWorld、OSWorld 2.0** · 评测质量 | OSWorld-Verified 修订了基准问题。2.0 仓库提供手工执行单题、检查判分和记录轨迹的工具。 | 固定本轮镜像及判分版本。争议题保留人工操作、最终文件和判分记录，以便复核。[OSWorld](https://github.com/xlang-ai/OSWorld) · [2.0 单题检查工具](https://github.com/xlang-ai/OSWorld-V2/blob/main/scripts/python/manual_examine.py) |
| **Gamer** · 游戏长程交互 | TA-GRPO 通过记忆压缩连接各段，在同一任务、同一阶段内比较到达该阶段的轨迹，并用环境可验证结果分配优势。 | 先在同一批轨迹上验证 β=1 时的损失和梯度与原版 GRPO 一致。对有可靠阶段判分的任务，记录每阶段到达数和奖励差异，再评估完整适配。[仓库](https://github.com/RainBowLuoCS/Gamer) · [论文 §4](https://arxiv.org/html/2611.18397v1#S4) |

</details>
