# CUA-RL 数据集 OnePage · 用 256 道题训练 OSWorld 弱项

**决策提案 · 2026-11-25。** 在 8 × A100 上，建议从官方 ScaleCUA Qwen3-VL-8B OSWorld SFT 起跑，优先训练**跨来源读取并写入指定文件**和 **Calc 跨表查价、计算与汇总**。选题时检查每个官方 RL 变体是否保留了失分的关键操作。原 64 道 pilot 已预跑；本轮可复用其中 56 道。扩大后的候选池和实时覆盖见下文。[模型与数据版本](../configs/sources.lock.json) · [官方 RL 题库](https://github.com/THUDM/SCALE-CUA#dataset)

## 截至 11 月 26 日，256 道里已有多少能用

**11 月 26 日夜间扩到了 637 道不同的官方候选。** 晚上第一批新增 35 道，第二批又新增 116 道。其中 100 道首次交给 GPU 试做（15 道主练候选、85 道分步候选），16 道有同协议的四次历史评分，可以直接复核。两批合计新增 151 道。对照官方发布版本逐题核验了文件哈希、题目指令和研究划分。637 道由原有的 61 道严核题、350 道 GPU 队列题、226 道只需本地复核的候选构成，每道只计一次。[官方数据版本](../configs/sources.lock.json) · [GPU 队列](../configs/dataset-screen/expand-queue.jsonl) · [历史候选](../configs/dataset-screen/expand-review.jsonl)

**截至北京时间 23:25，已严核 87/256 道。** 作业 `23560412` 使用 8 张 A100 筛选。CPU 复核已挂起等待，达到配额或发完候选后，会等已领题的桌面写完结果，再逐题核验轨迹、得分和评分器。最终的 `prompts-reviewed.jsonl` 是能训练的题单，`review-decisions.jsonl` 留下本次逐题结论、指令和分数。按本次已准备好的评分器正反例和分类配额，自动复核最多能确认 **141 道**，其中主练最多 59、分步最多 64、广覆盖最多 18。要凑齐 256 道，还需要给更多主练和广覆盖题核验评分器。[复核入口](../AIS_ENTRY.txt) · [夜间评分对照](../configs/dataset-screen/night-review-controls.jsonl) · [对照重算脚本](../runs/dataset-screening/night-20261126/build_controls.py) · [复核状态及输出](../runs/ais-screen-expand/23560412/night-review)

**11 月 26 日 22:59（北京时间）已严核 87/256 道，还差 169 道。** 主练题 27/128，分步题 52/64，广覆盖题 8/64。作业 `23560412` 的两个 GPU rank 已领取本轮新增的主练题，分别要求按会议和年份回填举办地所在洲、在本地学术主页模板中修改作者所在地。[作业实时计数](../runs/ais-screen-expand/23560412/progress.json) · [逐题名单](../runs/ais-screen-expand/23560412/prompts-reviewed.jsonl)

本次从官方训练划分补入 **50 条不同的候选指令**，其中 9 条暂列主练、41 条暂列分步。28 条首次交给 GPU 各试四次；22 条已有四次评分，先在本地复核。另有一道已有候选要求在第二页应用 dissolve 切换效果，以前四次都没拿满分。这次在原始演示文件上验证了正确操作得 1，改错页或选错效果均得 0，已安排补跑四次。GPU 清单由 198 道增至 227 道。[GPU 清单](../configs/dataset-screen/expand-queue.jsonl) · [本地复核清单](../configs/dataset-screen/expand-review.jsonl) · [离线评分对照](../runs/dataset-screening/evening-expansion-20261126/controls.json)

主练题仍缺 101 道。会议题的评分器逐行读取原工作簿，21 行全部填对得 1，任意一行改错均得 20/21。新计入严核的那道 Chrome 题此前四次得 1、0.5、0.5、1。本地确认只关闭深色模式或只启用 Do Not Track 都只能得 0.5 后，沿用了这四次成绩。[逐行评分对照](../runs/dataset-screening/evening-expansion-20261126/controls.json) · [Chrome 题四次实测](../runs/ais-screen-expand/23560412/tasks/chrome_93eabf48-6a27-4cb6-b963-7d5fe1e0d3a9_traj_verify_3/outcome.json)

作业启动时有 61/256 道达标（23.8%）。其中 8 道主练题来自 5 个原始任务，保留了原题失分的关键操作。45 道分步题各练其中一个环节，8 道广覆盖题取自预先抽取的其他桌面操作。本次复用了 5 道广覆盖题的四次历史评分，在 CPU 上核对了封存记录和评分器对照。剩余 56 道广覆盖候选也都有四次评分，先在 CPU 上复核。[累计逐题名单](../configs/dataset-screen/strict-expanded-20261126.jsonl) · [广覆盖待核名单](../runs/dataset-screening/shared-queue-20261126/breadth-review.jsonl)

本轮最先新增的主练题要求把两个源目录的完整层级备份到指定位置。未操作时为零分，四次试做得 1、1、0、0。评分对照中，丢掉一层目录或备份到错误位置都得零分。它入选时累计达到 **62/256**。后续实时进度见作业汇总。[实时进度](../runs/ais-screen-expand/23560412/progress.json) · [逐题名单](../runs/ais-screen-expand/23560412/prompts-reviewed.jsonl)

新增的两道主练分别是用工时和费率算工资再扣税（四次中三次满分）、按等级表回填三位学生的成绩（四次中一次满分）。后者的另三次分别填对两行、零行和一行，评分能区分完成程度。其中五道分步题分别练整列查负责人、单行查负责人、找最早论文并写回标题、逐行提取发表年份、把指定页标题写入同页备注。五道的原题 SFT 都曾满分，变体的失分提供了这些子操作还有训练空间的证据。[8 题实测](../runs/ais-dataset-screen/23549884/summary.json) · [16 题实测](../runs/ais-dataset-screen/23554958/summary.json)

另一道分步题要求逐周计算利润并写入总计行，四次成绩为 1、1、0.7、1。两道广覆盖题来自预先抽取的 pilot，分别练订单数量汇总和在指定路径新建目录，均为四次满分。这两题的旧作业没有逐题记录未操作时的桌面分数，本轮在原始素材或相同目录结构上运行官方评分规则，两题初态均为零分。[补齐的分数](../runs/dataset-screening/primary-20261126/local-file-repair-li84dyby/summary.json) · [广覆盖题评分对照](../runs/dataset-screening/primary-20261126/followup-controls.json)

早上冻结的 256 个候选和过夜作业最后导出的 256 个候选都取得了四次评分。早上的名单包含最初严核的 46 道，过夜最终名单包含其中 45 道。累计名单按跨批次唯一任务计数。[早上冻结的题单](../runs/dataset-screening/skill-review-20261126-P1XYhfLe/prompts-256.jsonl) · [过夜作业最终题单](../runs/ais-dataset-screen/23533722/prompts-256.jsonl)

## 我们的 SFT 弱在哪里

本轮 RL 使用 ScaleCUA Qwen3-VL-8B 的 OSWorld SFT 权重。OSWorld 的 361 道题中，358 道有有效分数，**均分 44.32%，满分率 43.02%**。其中 154 道满分、5 道部分得分、199 道零分。其余 3 道分别遇到截图、桌面输入和环境初始化故障。将缺分分别按 0 或 1 计，361 题的均分范围是 **43.96%–44.79%**。同次评测也测了独立的 ScienceBoard SFT 权重，发布环境中 146/169 道有分，均分 16.24%。恢复原题初始状态后 148/169 道有分，均分 16.06%。两项均分各按本配置已评分题计算。[AIS 作业 23495234 的逐题记录（共享工作区）](../runs/ais-sft-evaluations/23495234/full/summary.json)

整套 OSWorld 的跨应用题中，92/93 道有分，均分 **16.30%**。为避免用留出题成绩筛选训练数据，下面只统计 RL 训练划分中的 295 道原题，其中 77 道跨应用题均有分，均分 **10.39%**。[研究划分](../configs/research-splits/manifest.json)

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

### 11 月 26 日冻结题单复核

09:06 冻结的 256 条官方指令均已完成四次 SFT 试跑，其中 150 条有过 1–3 次满分。本批从这 150 条中选取 31 个原题家族的 46 道，用原始素材和官方评分器复核。7 道入选（1 道主练、6 道分步），4 道同家族回填变体作为备选，35 道排除。姓名合并任务只交前 5 行就能拿满分，原始文件有 5,000 行；收据记账任务填错交易描述和类别也能拿满分。这两道均已排除。[逐题对照分数](<CUA-RL Mixed Review 20261126.jsonl>)

累计严核通过 46 道，含 6 道主练、39 道分步和 1 道广覆盖。主练覆盖 3 个原题家族。新增的整列单位换算保留了 SFT 在原题未完成的换算、舍入和写回操作，与之前的双列换算题同源。此清单按冻结题单的原有配额计数，AIS 作业的滚动导出另行统计。[冻结严核题单](../configs/dataset-screen/strict-reviewed-20261126.jsonl) · [150 道逐题状态（共享工作区）](../runs/dataset-screening/mixed-review-20261126/mixed-150.jsonl)

11 月 26 日上午用现成的四次成绩复查了 108 道混合难度题，含冻结集剩余的 81 道和快照中冻结集外的 27 道。逐题读指令后，97 道以排版、软件设置等其他技能为主。剩下 11 道检查官方判分，9 道会给关键错例满分。例如列车记录去重题接受八条编造的记录，两道幻灯片汇总题接受空白汇总页。另 2 道只需写入题面给定的表头或占位值。本批新增主练 0 道。[108 道逐题去向和对照分数（共享工作区）](../runs/dataset-screening/screen-next-20261126/local-review.jsonl)

`screen-next` 已完成。作业 23549884 于北京时间 12:36 汇总，从首个分片写入题单到汇总约 22 分钟。8 道 Calc 变体的桌面初态均得零分，32 次预跑取得 31 个有效评分。5 个家族的原题 SFT 均满分，变体难度如下。[冻结名单](../configs/dataset-screen/expansion-20261126.json) · [作业汇总（共享工作区）](../runs/ais-dataset-screen/23549884/summary.json)

- 整列查负责人、找最早发表论文并写回标题，各 1/4 满分；从日期提取整列年份 3/4 满分。三道均已通过离线正例和错例检查，列为分步训练候选。[离线判分记录](../runs/dataset-screening/screen-next-20261126/controls.json) · [查负责人和最早论文的判分记录](../runs/dataset-screening/expand-check-20261126-RtcU6sTT/report.json)
- 全表替换、销售额求和、逐周利润率和累计销售额均为 4/4 满分，作为已掌握操作的对照。
- 逐周利润加总计有三次有效奖励（1、0.7、1）。另一次在桌面评分器读取工作簿时发生 `ChunkedEncodingError`，成功率留空。[评分日志](../runs/ais-desktop-services/worker-z7n7k39r/worker.log)

扩围作业 `bash AIS_ENTRY.txt screen-wide` 接在 `screen-next` 后，独立分片试跑 16 道新题，覆盖 11 个原题家族。其中 3 道 Calc 包括工时计税、按评分区间查等级和查负责人；7 道 Impress 覆盖从标题抄到备注、定位指定幻灯片写入标题或表格；6 道 OS 考文本变换和文件整理。工时计税、查等级、按目录层级备份这三道的原题 SFT 为零。其余 13 道用于测相邻技能的难度。已核对训练划分、历史题单及规范化题面，并剔除与留出题面或已知素材重合的候选。[冻结名单](../configs/dataset-screen/wide-20261126.json)

16 道的离线初态均为零，正确产物均为满分，逐题构造的关键错例均扣分。例如双页插表只完成一页得 0.5，把表格放错页得 0。另两道插表题接受包含目标词的错误表头，已排除。八卡按每卡两题执行，先测真实桌面初态，初态零分后再各采四次。训练集收录仍依赖实际难度和技能复核。[逐题对照（共享工作区）](../runs/dataset-screening/wide-20261126/selected-controls.json)

作业 23554958 于北京时间 13:58 完成扩围筛选，八个分片从写入题单到汇总约 34 分钟。16 道题的真实桌面初态均为零，64 次试跑取得 63 个有效评分。15 道四次齐全，其中 7 道有时满分、有时失分，7 道全满分，1 道四次均为零。建议优先保留工时计税（3/4 满分）和按区间查等级（1/4 满分）。两道的原题 SFT 均得零分。查等级四次分别填对 2、0、1、3 个目标单元格，可检验部分完成的进步。[逐题汇总](../runs/ais-dataset-screen/23554958/summary.json) · [评分字段与离线对照](<CUA-RL Dataset Audit 32.json>)

目录备份题四次均正常结束，模型在最后回复中都列出了多一层 `test_environment` 的路径，和评分器要求的 `backup/dir1/…` 等目标不符，列为难题复核候选。改幻灯片标题题已有三次满分，另一次评分时读取来宾文件发生 `ChunkedEncodingError`，四次成功率留空。合并 `screen-next` 与 `screen-wide`，共 24 道新题、94/96 次有效评分，22 道四次齐全，10 道有时满分、有时失分。主练和分步题的配额仍按各题对应的失分操作单独复核。[备份题轨迹](../runs/ais-dataset-screen/23554958/rank-4/episodes/os_5c1075ca-bb34-46a3-a7a0-029bd7463e79_traj_verify_5) · [标题题评分日志](../runs/ais-desktop-services/worker-jso0tdxh/worker.log)

### 11 月 26 日下午的定向扩围

这轮把判分器容易误判的题先留在 CPU 上处理。对官方训练库 1,691 道题，专门检查了用 `cat` 读取计数结果、再用单个数字做子串匹配的评分规则。找到 4 道，分别把正确数值扩大十倍后交给原评分函数，4 道仍得满分。例如 PHP 行数写成 540 和 54 得分相同。另有一道清理 notebook 文件的题，按指令保留所有 `*_failed.ipynb` 仍得零分，因为排除串 `d.ipynb` 同时命中了正确文件名。五道均未进入本轮八卡队列。[数字错例测试](../runs/dataset-screening/primary-20261126/numeric-false-positives.json) · [初态、正确产物与错例](../runs/dataset-screening/primary-20261126/followup-controls.json)

最初为扩围挑了八道题，每题计划在初态零分后跑四次。此次提交已将这八道并入下文的共享队列。五道首次筛选，覆盖按目录层级复制、移动指定目录、按文件名定位后重命名或移动，以及逐行清理标题写回指定列。相同素材只改大小写方向的两个变体保留一个。另三道为难题补采，分别是净销售额与毛利、两列单位换算、保留目录层级备份。这三道在历史四次预跑中均未满分，本地正确产物均得满分。新四次单独记录，原始四次和出处保留在复核清单中。合计最多新增 32 次试跑，八道候选均未计入当时达标的 56 道。[八卡题单](../configs/dataset-screen/targeted-20261126.json) · [用途、历史奖励和判分对照](../configs/dataset-screen/targeted-20261126-review.jsonl)

前两批各有一次评分下载中断，两次在原桌面重试后均收到完整文件。旧的外层检查仍保留了第一次错误。现在只在同一次只读文件调用拿到完整文件后清除该调用的分块传输错误。重试耗尽或其他评分操作失败仍会标记未评分。在本地 A100 复用两道题已有的六条轨迹，各补跑一次。两批合计 24 题、96 次评分已齐。逐周利润题进入分步名单。另一道只要求按指令改幻灯片标题，四次均满分，留作对照。[评分观察器](../scripts/ais_verifier.py) · [回归测试](../scripts/test_ais_verifier_download.py) · [两题补跑结果](../runs/dataset-screening/primary-20261126/local-file-repair-li84dyby/summary.json)

### 本次可以提交的筛选作业

作业 `23560412` 起跑时给 8 张卡准备了 65 道候选，随后在同一份清单末尾追加了 77 道。当时 GPU 队列共 **142 道**，其中 137 道首采、5 道补采，一题四次，最多 568 次试做。选题按训练划分去重，排除与留出题面或已知素材重叠的题。候选中有 64 道暂列主练、78 道暂列分步。52 道已有技能与离线评分对照，其余可以先采样，再按实测结果复核。例如对目录备份会核对层级和目标路径，仅检查文件存在容易把错误备份也判为满分。[GPU 候选](../configs/dataset-screen/expand-queue.jsonl)

当时另有 **172 道**已有四次历史评分，与 GPU 队列及 61 道已入选题均无重复。62 道有时满分、有时失分，其中 40 道可能补主练、22 道可能补分步。另 56 道来自预先选定的广覆盖候选。先用已有记录核对这 118 道的任务目标、评分器和初态。剩余 54 道四次均未拿满分，其中 35 道可能补主练、19 道可能补分步。先检查正确做法能否拿满分及关键错误是否扣分，再决定是否值得补采。旧审查记录中曾因偏离查表、汇总、跨来源写回而搁置的题保留原决定，重新入选要能指出具体失分操作。两份名单合计 314 道候选，各类配额仍分别验收。[历史评分及复核顺序](../configs/dataset-screen/expand-review.jsonl)

本地 A100 已完成队列第一题。它要求移动匹配的 notebook 并保留目录结构，未操作桌面得 0，四次试做得 0、0、0、0.5。正式作业可复用这四次，该题目前作为难例保留。[逐题结果](../runs/ais-screen-expand/notebook-efpbur1v/tasks/os_5c1075ca-bb34-46a3-a7a0-029bd7463e79_traj_verify_6/outcome.json)

每个进程保留一个 SFT 服务和两个桌面，完成一题就从同一份清单领取下一题。作业期间可原子替换清单以追加候选。只要仍有题在跑，空闲进程会等待补入的题。每一类达到严核配额后停止派发该类，在途采样正常收尾。全部满额或当前队列耗尽后作业退出。每 30 秒更新 `runs/ais-screen-expand/作业号/progress.json`，同目录的 `prompts-reviewed.jsonl` 保存可用指令。再次提交同一入口时会从前次相同协议的记录补齐缺失试次。[统一入口](../AIS_ENTRY.txt) · [调度和入选条件](../scripts/ais_screen_queue.py)

此前两批分别用 101 分钟筛 48 题、777 分钟筛 426 题。按两批平均速度外推，142 道全部采完约需 4.3–5.0 小时。当前任务逐题检查初态，实际用时还受桌面启动、长轨迹和提前停发影响。历史打分复核另计。用 `bash AIS_ENTRY.txt screen-expand --plan` 可核对 GPU 题量、历史复核量及各类起始缺口。[48 题结果](../runs/ais-dataset-screen/23527641/summary.json) · [426 题结果](../runs/ais-dataset-screen/23533722/summary.json)

> 按开跑时的数字，主练缺 120 道。把 GPU 的 64 道主练候选和旧作业里 40 道有时满分的主练候选全部算上，也只有 104 道。那 35 道历史难题里至少需要 16 道最终达标，或者继续找到新的主练候选。这是按全部通过计算的上限，实际要以逐题复核为准。

### 11 月 26 日傍晚，筛选队列扩到 198 道

作业 `23560412` 继续使用 8 张卡、16 个桌面。GPU 队列从 142 增至 **198**，含 69 道主练候选和 129 道分步候选。这次增加的 5 道主练排在队首。1 道首次采样，4 道已有四次未满分的记录，经离线核验后再各试四次。51 道新分步候选排在队尾，例如提取指定时刻的视频帧、把查到的结果写回工作表、按条件归档文件。分步题达到 64 道严核配额后就停止派发。[GPU 题单](../configs/dataset-screen/expand-queue.jsonl)

4 道补采题分别要求冻结行列并改表名、填完整列序号并改表名、同时设置幻灯片备注和背景、按会议和年份回填举办国家。用原始文件运行官方评分器，四道题都满足初态 0、正确产物 1、关键错例扣分。例如在冻结题中，只冻首行、只冻两列或改错表名都只能得 0.5。另有一道 Writer 题要求只给首尾段落加删除线，整篇加删除线也得满分，已从待核名单移除。又找到了 9 道可复用四次旧评分的候选，其中 3 道可能补主练。扣除已转入 GPU 的题，现有 177 道只需在 CPU 上做下一步复核。[历史题单](../configs/dataset-screen/expand-review.jsonl) · [逐题对照分数（共享工作区）](../runs/dataset-screening/second-expansion-20261126/controls.json)

## 实验与验收

**256 道训练题收齐并复核后，先比较 16 和 24 个并发桌面，再跑正式 RL。** 用同一个冻结的 SFT 模型、同一组 16 道题，每题各试 8 次，两档都按正式 RL 的布局分配八卡。记录预热后收齐 16 个完整题组的耗时、每小时有效评分数和环境错误数。选定并发数后，用原版 GRPO 验证一次非零梯度更新和 checkpoint 保存，再开始三种方法的对照。用头几个完整训练步的实测耗时修正下方预算。

**数据收益**用同一 SFT、原版 GRPO + Frontier 和交互预算，对比「按应用分层抽取 256 题」与「定向筛选 256 题」。**算法收益**在同一定向题单上比较原版 GRPO、时间加权消融和 Gamer TA-GRPO。首轮每步 **16 个 prompt × 8 个 rollout，25 次更新机会**。PPO 可沿用划分并单记 critic 开销。[GRPO 配置](../configs/qwen3vl-osworld-original-pilot-8gpu.yaml) · [Gamer 方法 §4](https://arxiv.org/html/2611.18397v1#S4)

**先检查长轨迹能否分组比较。** 筛选作业 `23533722` 的 426 道题中，394 道取得了四次有效终局评分。按原始得分计算，176 道在四次尝试间出现分差。若只看超过 20 次模型调用的尝试，253 道题还有至少两条轨迹可比较，其中 110 道的最终得分有差异。超过 30 次调用后分别只剩 197 道和 83 道。较早完成的轨迹已退出后段。正式训练需另记每个阶段的到达数，以及**当前阶段**出现可验证分差的题数。[终局评分汇总](../runs/ais-dataset-screen/23533722/summary.json) · [各 rank 的逐次调用记录（共享工作区）](../runs/ais-dataset-screen/23533722/)

**TA-GRPO 的对照。** Gamer 用记忆摘要连接长轨迹，按同一任务的第几次记忆压缩分组。桌面上同一段号可能对应不同进度，例如有的尝试已开始填表，有的还在找文件。先选能只读核验中途产物的题。固定分段、摘要、阶段奖励和每个动作的损失计算方式，在完全相同的设置下对比段内等权与靠近段末的动作获得更高权重。这样可以单独衡量时间加权的贡献。STEP 在 OSWorld 上同时调整选题和步骤更新，把 UI-TARS-DPO-7B 从轨迹级 GRPO 的 18.9 提升到 23.8，比较时应固定我们的 Frontier 采样规则。[Gamer §4](https://arxiv.org/html/2611.18397v1#S4) · [STEP 实验与消融](https://aclanthology.org/2026.findings-acl.1532.pdf)

**PPO 的对照。** 先试 Turn-PPO，把一次 GUI 操作当作一个时间步。它的价值模型（critic）估计从当前截图和历史继续操作，预计最终能得多少分。首轮沿用相同终局评分，单独核算价值模型的显存、训练时间和真实桌面操作数。GUI-Shepherd 在 AndroidWorld 上的同模型对照中，ARPO/GRPO 和只用结果奖励的 PPO 都是 37.0%，加入单步评分模型后 PPO 达到 40.5%。若同时换算法和奖励，涨分来源就难以判断。工程上可参照 NeMo 的多模态 PPO、verl 的价值模型更新和 ROLL 的多环境调度。[Turn-PPO](https://aclanthology.org/2026.findings-eacl.328/) · [GUI-Shepherd](https://arxiv.org/html/2509.23738v1#S4) · [NeMo RL](https://docs.nvidia.com/nemo/rl/nightly/guides/ppo.html) · [verl](https://verl.readthedocs.io/en/latest/algo/ppo.html) · [ROLL](https://alibaba.github.io/ROLL/docs/Development/Architecture/AgenticPipeline/)

**24 个桌面的正式 RL 预算。** 8 张 A100、每卡所在的进程开 3 个桌面，单方法、单个随机种子的 25 步先预留 **30–40 小时**。16 桌面的 RL 试跑累计取得约 91–113 条有效评分轨迹/小时。24 桌面若完全线性提速，名义 3,200 条约需 19–24 小时纯采样。若吞吐只提高 25%–35%，约需 21–29 小时纯采样。预算另外为首步多采、启动、梯度更新和存盘留出时间。试跑在第一次完整更新前结束，这个预算尚无 24 路端到端实测，排队、筛题和最终评测另计。每个进程开 4 个桌面可做 32 路对照。单进程实测从 2 个加到 4 个时，截图延迟的第 95 百分位从 4.29 秒升到 4.71 秒，也开始遇到 CPU 限速。比较时应看每小时成功评分的轨迹数。[16 桌面 RL 配置](../runs/ais/original-grpo-pilot-vw46xwnd/training.yaml) · [桌面并发实测](../runs/ais-desktop-validation/20261125-15904ca/summary.json)

**验收**用 dev 181 题选 checkpoint，冻结后在 test 177 题同时报整体均分和预先标记的弱项子集均分，按 32 个原始 UUID 做配对比较并报告区间。官方 OSWorld 361 单独报分、评分覆盖率及与训练原题的重叠数。295 个候选训练 UUID 都出现在官方 361 中，公开 SFT 的逐题暴露情况未知。[研究划分](../configs/research-splits/manifest.json) · [官方评测清单](https://github.com/THUDM/SCALE-CUA/blob/3929e2fe364623153f2caa94ead71dc1aea50fb0/osworld_eval/evaluation_examples/test_nogdrive.json)

<details>
<summary><strong>筛选复核与 8 卡预算</strong></summary>

- 每个候选保存原始 UUID、变体 ID、最后一个完成的操作、失败证据、目标能力、判分字段和选留理由。区分读错来源、写错位置、重复尝试、素材不可用和判分异常。环境异常修复后重测。复核快照可重置和素材可用。
- 已查到训练候选与 dev/test 有 14 条指令在统一大小写和空白后完全相同，涉及 7 种指令，原 pilot 命中 0 条。先剔除这些候选，再复核目标、素材与判分近似的题。现有划分的 train/dev/test 分别覆盖 295/32/32 个原始 UUID。[研究划分](../configs/research-splits/manifest.json)
- 原 64 题各试 4 次，28 题全败、18 题有胜有负、18 题全胜。复用同协议预跑后还需 **192 × 4 = 768 条**。已评分轨迹平均 443 秒，按 8 路预跑和 70%–85% 有效利用率估算，补跑约 **14–17 小时**。实际 RL 作业 23524450 有 8 个桌面 worker、共 16 路并发。首个 session 启动后约 37 分钟获得 56 个评分，约 84 分钟累计获得 158 个评分，另有 4 次任务错误。用这两个观测点的累计吞吐（约 91–113 条/小时）估算，256 题各试 8 次约需 18–23 小时采样。正式 GRPO 计划按 16 题 × 8 次 × 25 步计算，名义为 3,200 条轨迹，约需 28–36 小时采样。首步会额外收集，计入启动、更新和存盘，单方法单种子建议预留 **36–48 小时**，另计排队、题目筛选和最终评测。这次在首次完整更新前切换到筛题，端到端耗时仍为估算。[RL 桌面记录](../runs/ais-desktop-services/worker-2djl3a20/controller.log) · [16 路运行配置](../runs/ais/original-grpo-pilot-vw46xwnd/training.yaml) 用实际完整训练步校准，并记录有效更新数、不同任务数、组内奖励有差异的比例、环境步数和 GPU 小时。[预跑审计（共享工作区）](../runs/ais-frontier-campaigns/23500435/verified/audit.json)

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
| **Gamer** · 游戏长程交互 | TA-GRPO 通过记忆压缩连接各段，在同一任务、同一段号内比较到达该段的轨迹，并用环境可验证结果分配优势。 | β=1 时段内动作等权。与原版 GRPO 对齐损失和梯度时，还需使用单段轨迹，并对齐奖励、组内优势归一化、掩码和损失归约。[仓库](https://github.com/RainBowLuoCS/Gamer) · [论文 §4](https://arxiv.org/html/2611.18397v1#S4) |

</details>
