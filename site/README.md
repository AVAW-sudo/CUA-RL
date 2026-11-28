# RL for CUA-Agent 展示页（内部路线预演）

学术项目页风格的单页网站，中文，无构建步骤，直接改 [dist/](dist/) 里的文件。

在仓库根目录运行下面的命令，然后打开 <http://127.0.0.1:8000/>：

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory site/dist
```

## 页面结构（[dist/index.html](dist/index.html) 里的顺序）

| 章节 | id | 内容 | 数字来源 |
| --- | --- | --- | --- |
| 看板 | `bench` | 四个数字、SFT Bench、RL Bench、CUA-Lite 的 117 题对照 | AIS 作业 23495234；SCALE-CUA 论文表 2；CUA-Lite §4 |
| 概念图解 | `concepts` | 八张小图：rollout、GRPO 优势、功劳分配、Frontier、视觉分段、critic、真实续跑、评分器三查 | 页面内 SVG |
| SCALE-CUA 拆解 | `scalecua` | 三段流程图、消融条形图、配置差别 | 论文正文、表 4、附录 D；`configs/` |
| CUA-Lite 拆解 | `cualite` | 流程图、SFT / RL 前后、桌面开销 | 论文 §4、附录 |
| 三方对照 | `compare` | 按行归一化的条形图，文字版在折叠里 | 同上 |
| 数据拆解 | `data` | 2,049 道公开题按应用拆分、筛题流程、三类配额进度 | `configs/research-splits/`、`configs/dataset-screen/full-pool.json`、`docs/` |
| RL 过程拆解 | `rl` | 一次更新的八步、时间分布 | README 的 smoke 和环境实测 |
| Case | `case` | 两道题的两次试跑并排对照，可切阶段 | `case/2026-11-27/` |
| 排班表 | `schedule` | 卡位分配、实验顺序 0–6、GRPO 时长、已跑作业 | `docs/2026-11-27-桌面强化学习筛题与实验排期.md`、`AIS_ENTRY.txt` |
| Insight | `insight` | 八条结论，每条一张迷你图 | 同上各处 |
| 下一步 | `next` | 六种方法怎样把信号分到每一轮、方法对照表、续跑校正 critic 示意 | 排期文档 |

样式在 [dist/style.css](dist/style.css)，交互在 [dist/app.js](dist/app.js)。

## 常改的内容

- **数字**：都直接写在 `index.html` 里。表格里的条形用 `style="--v:…"`（0–100）控制长度，`<em>` 或 `<span>` 里是显示的数字，两处要一起改。筛题进度在看板第三个数字、数据拆解的配额表和 Insight 里各出现一次。
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
