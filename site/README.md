# RL for CUA-Agent 展示页（内部路线预演）

学术项目页风格的单页网站，中文，无构建步骤，直接改 [dist/](dist/) 里的文件。

在仓库根目录运行下面的命令，然后打开 <http://127.0.0.1:8000/>：

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory site/dist
```

## 页面结构（[dist/index.html](dist/index.html) 里的顺序）

| 章节 | id | 内容 | 数字来源 |
| --- | --- | --- | --- |
| 看板 | `bench` | SFT 和 RL 的定义、四个数字、SFT Bench、RL Bench | AIS 作业 23495234；SCALE-CUA 论文表 2 |
| 一道题从头到尾 | `walk` | MP3 题的指令和初始桌面、一步动作与坐标换算、两次试跑的动作条、关键帧对照（含 Thunderbird 题）、完整回放链接 | `case/2026-11-27/`；任务定义来自公开数据集 extreme1228/ScaleCUA |
| 评分器 | `grader` | getter / expected / metric 的真实配置、Thunderbird 的部分得分、评分器会出错的例子和入选前的三项检查 | 同上；`docs/CUA-RL Dataset OnePage.md` |
| GRPO | `grpo` | 用两道题的真实得分算组内优势、同一个优势记在每一步上、四次同分的题 | 同上；作业 23533722 |
| SCALE-CUA 拆解 | `scalecua` | 三段流程图、两道题的 VeriGen 生成记录、Frontier Sampling、Visual Context Segmentation、消融 | 论文正文、表 4、附录 D；任务元数据；`configs/` |
| CUA-Lite 拆解 | `cualite` | 流程图、SFT / RL 前后、桌面开销、负结果、117 道办公题对照、三方设置对照 | 论文 §4 与附录 |
| 我们的数据 | `data` | 公开题库按应用拆分、跨应用题占比、筛题流程、三类配额进度、1,691 道题的去向 | `configs/research-splits/`、`configs/dataset-screen/full-pool.json`、`docs/` |
| 训练与排班 | `run` | 一次更新的八步、时间分布、卡位分配、实验顺序 0–6、GRPO 时长、已跑作业 | README；`docs/2026-11-27-桌面强化学习筛题与实验排期.md`；`AIS_ENTRY.txt` |
| 下一步 | `next` | critic 的定义与 value 曲线示意、五种方法、信号分配示意图、真实续跑示意 | 排期文档 |
| 结论 | `conclude` | 八条已确定的结论，链回各节 | 同上各处 |

每节末尾有一到两个引用块：“暴论”是一句判断，“本质”是一句机制分析。

样式在 [dist/style.css](dist/style.css)，交互在 [dist/app.js](dist/app.js)。整屏截图在 `dist/media/walk/`，取自完整回放里第 4 次试跑的第 1、2 步。

## 常改的内容

- **数字**：都直接写在 `index.html` 里。表格里的条形用 `style="--v:…"`（0–100）控制长度，`<em>` 或 `<span>` 里是显示的数字，两处要一起改。筛题进度在看板第三个数字和“我们的数据”的配额表、流程箭头里各出现一次。
- **动作条和 value 曲线**：`app.js` 顶部的 `TRACE` 是两次试跑每一步的动作类型，从完整回放里逐步统计；value 曲线是示意。
- **Case**：关键帧放在 `dist/media/case/`，文件名是 `<任务>-r<试跑序号>-s<阶段序号>.webp`；文字和步数写在 `app.js` 顶部的 `CASES` 里。`fork` 是两次试跑分岔的阶段序号，没有就写 `-1`。完整回放页原样放在 `dist/case/<日期>/`，内容与仓库 `case/` 下的同名文件一致。
- **“下一步”的信号分配图**：由 `app.js` 里的 `creditMap` 一段生成，每行的数值是示意。

## 检查与发布

[检查脚本](authoring/qa/check-site.cjs) 用 Playwright 打开本机 8000 端口，检查脚本错误、资源加载、Case 切换、信号分配图、回放子页，以及 393 px 手机宽度下没有横向溢出：

```sh
node site/authoring/qa/check-site.cjs
```

可用 `PLAYWRIGHT_MODULE` 指向 playwright 或 playwright-core，用 `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` 指向本机 Chrome，用 `CUA_SITE_PREVIEW_URL` 换预览地址。

当前的私有预览：<https://claude.ai/artifact/2SWDWoc7cLfFhDPVDANvhC>。

字体 Newsreader、Source Sans 3、IBM Plex Mono 内嵌在 `dist/fonts/paper-fonts.css`，均为 SIL Open Font License 1.1；中文标题用 Google Fonts 的 Noto Serif SC，离线时退回系统字体。
