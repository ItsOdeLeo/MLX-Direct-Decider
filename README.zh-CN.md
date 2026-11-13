# MLX Direct Decider：Mac 本地 MLX 决策引擎，无需 fine-tuning（微调）

<p align="center">
  <img src="docs/assets/mlx-direct-decider-hero.zh-CN.svg" alt="MLX Direct Decider：Apple Silicon 上的本地 MLX 决策流程，无需微调" width="1280" />
</p>

<p align="center"><strong>已有 MLX 大模型 → 本地决策引擎。</strong><br />无需 fine-tuning（微调），无需额外决策模型权重。</p>

**MLX Direct Decider** 是一个开源 Python 库，**复用 Apple Silicon Mac 上已有的 MLX 模型，无需额外训练（training-free）即可做决策**。适用于本地大模型动作选择（local LLM action selection）、分类、语义路由（semantic routing）及 enum / boolean 判断。

通过兼容的 [MLX-LM](https://github.com/ml-explore/mlx-lm) 检查点做文本决策，或通过 [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) 检查点做实验性的**文本＋图片候选决策**。提供应用状态、自然语言请求与允许的选项，获得稳定的候选 ID、`no_match` 或 `abstain`。可通过 Python API、CLI 或本地文本 HTTP 服务接入。

**无需额外 fine-tuning（微调）、训练新的决策头或获取专用决策模型权重。** 完整模型照常处理提示词；本库从保持不变的原始输出头读取候选 logits，无需生成后续文本。仍需已有兼容的基础模型权重。

> **JevBench 公开题集实测：74.0%——171/231 题正确，无需额外 fine-tuning。**
> Qwen3.5-9B OptiQ 混合 4/8-bit · Apple M2 Max · 本次评测的问答提示配置。
> 这是旧版公开题集的本地自测结果，尚非官方榜单成绩。
> [完整结果、原配置对比与复现方法](docs/jevbench.zh-CN.md)。

<p align="center">
  <img src="docs/assets/badges/apple-silicon.svg" alt="Apple Silicon" />
  <img src="docs/assets/badges/python.svg" alt="Python 3.11+" />
  <a href="LICENSE.zh-CN.md"><img src="docs/assets/badges/license.svg" alt="MIT License" /></a>
  <a href="CHANGELOG.zh-CN.md"><img src="docs/assets/badges/release.svg" alt="0.2 experimental, GitHub release" /></a>
</p>

<p align="center">
  <a href="#quick-start">快速开始</a> ·
  <a href="#models">已测模型</a> ·
  <a href="#blocks">游戏演示</a> ·
  <a href="docs/jevbench.zh-CN.md">JevBench：74.0%</a> ·
  <a href="#results">实测结果</a> ·
  <a href="#related-projects">AnyJev 与相关项目</a> ·
  <a href="#faq">常见问题</a> ·
  <a href="docs/testing.zh-CN.md">怎么测试</a> ·
  <a href="#connect">联系</a>
</p>

<p align="center">
  <a href="https://x.com/ItsOdeLeo"><img src="docs/assets/badges/follow-x.svg" alt="关注 X / Twitter：欧的Leo（@ItsOdeLeo）" /></a>
  <a href="https://xhslink.com/m/18bjTTf180W"><img src="docs/assets/badges/follow-xiaohongshu.svg" alt="关注小红书：欧的Leo，小红书号 ItsOdeLeo" /></a>
</p>

<p align="center"><a href="README.md">English</a> · <strong>简体中文</strong></p>

---

## 基于 MLX 的动作选择与语义路由

**MLX Direct Decider** 根据当前应用状态、用户的话和动态候选动作，让本地模型选出一个稳定的业务 ID；没有合适选项时，可以返回“不匹配”或“需要澄清”。

它适合嵌入你已有的工具：切换一个入口、选择当前列表里的内容、暂停播放器，或做一个 boolean / enum 判断。模型在 Mac 上运行，权重由你选择；无需先训练新模型。

> **项目与包名：** 项目名为 MLX Direct Decider，发行包与命令为 `mlx-direct-decider`，Python 导入为 `mlx_direct_decider`。0.2 是实验版，已在 GitHub 发布，尚未上传到 PyPI。

| 能力 | 在应用里意味着什么 |
| :--- | :--- |
| **动态候选** | 页面变了，允许动作跟着变；业务 ID 由应用定义。 |
| **明确拒绝** | `no_match` 和 `abstain` 是正式结果，不必把每句话硬变成操作。 |
| **版本保护** | 推理期间状态变化，旧结果不能拿去执行；执行授权只能使用一次。 |
| **前缀复用** | 复用稳定上下文，每句新话仍实际推理；不缓存最终答案。 |
| **便于接入** | 提供 Python API、CLI 和 localhost HTTP 服务。 |

模型会犯错。候选分数用于排序，**不是经过校准的正确概率**；状态保护也不能替代语义判断。

<a name="quick-start"></a>

## 快速开始

需要 **Apple Silicon Mac、原生 ARM Python 3.11+**，以及本地 MLX-LM 模型权重。建议先从已验证的 Qwen 检查点开始，具体版本见[支持模型](docs/models.zh-CN.md)。

从 GitHub 安装 v0.2.0 版本：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "mlx-direct-decider[mlx] @ git+https://github.com/ItsOdeLeo/MLX-Direct-Decider@v0.2.0"
export MLX_DIRECT_MODEL=/absolute/path/to/your/local/mlx-model
```

如需从源码开发，可克隆仓库，并在仓库目录运行 `python -m pip install -e '.[mlx]'`。

```python
import os
from mlx_direct_decider import Candidate, DecisionRequest, MLXDecisionEngine

engine = MLXDecisionEngine(os.environ["MLX_DIRECT_MODEL"])
result = engine.decide(DecisionRequest(
    state={"focused_window": "notes"},
    utterance="Close it",
    candidates=(Candidate("close.notes", "Close the open notes window"),),
    state_version=1,
))

print(result.status, result.candidate_id)
print(result.margin, result.timing)
```

命令行使用同一份请求约定：

```sh
mlx-direct-decider decide --request examples/decision.json
```

返回结果包含候选 ID、原始 logits、候选分数、margin、状态版本、模型身份、真实耗时和缓存信息。[Python API](docs/python-api.zh-CN.md) 包含 boolean 判断、状态更新与执行示例。

## MLX、MLX-LM 和 MLX-VLM 支持

- **MLX：**共用计算底座，无需单独的 Metal 后端。
- **MLX-LM：**通过 `MLXDecisionEngine` 做文本决策，支持有容量上限的前缀复用。
- **MLX-VLM：**通过 `MLXVLMDecisionEngine` 或
  `mlx-direct-decider decide --backend mlx-vlm --image ...` 做文本＋图片决策，无需微调。

使用 `python -m pip install -e '.[mlx,vlm]'` 安装图片支持。
[图片快速开始与真实模型检查](docs/multimodal.zh-CN.md)记录 Qwen3-VL、SmolVLM 的错误，以及 Gemma 检查点的加载限制。VLM 请求目前重新计算完整上下文；HTTP／游戏演示仍使用文本状态。各文本检查点的覆盖范围见[模型指南](docs/models.zh-CN.md)。

<a name="blocks"></a>

## 看本地模型玩方块

![MLX Direct Decider Blocks：完整 20 步开发运行，4 倍速播放](docs/assets/blocks-direct/preview.gif)

**Qwen3.5-9B · 模型选择 20 次落点 · 消除 6 行 · 600 分。** 于 2026 年 10 月 14 日使用 MLX Direct Decider 名称重新录制。GIF 以 **4 倍速**展示完整运行，包含推理等待。模型读取结构化棋盘和规则计算的结果，从所有合法垂直落点中选择；没有“最佳落点”算法代走。这是每块一步的回合制决策，不是看截图或逐帧控制。

[本地运行游戏](docs/blocks.zh-CN.md) · [原速完整录像](docs/assets/blocks-direct/full-run.mp4) · [全部决策记录](docs/assets/blocks-direct/transcript.json)

本次单轮开发演示达到预定 20 步上限，没有重试、浏览器错误或录制错误。包括首次拒绝在内的早期尝试仍保留在[历史游戏报告](docs/blocks.zh-CN.md#recorded-development-attempts)。这不是留出游戏 benchmark，也不是速度声明。手动游玩可使用静态服务器；新的 AI 决策需要本地 MLX 后端。

<a name="models"></a>

## 本地实测模型完整清单

**截至 2026-10-13，共实测 11 个不同检查点 · Apple M2 Max · 64 GiB。**
全部复用现有模型权重，无需额外 fine-tuning 或专用决策权重。下表记录直接候选评分；能够完成运行与决策是否正确分别标明。

| 本地检查点 | 后端／输入 | 已记录的验证 |
| :--- | :--- | :--- |
| `Qwen3.5-9B-OptiQ-4bit` | MLX-LM · 文本 | [JevBench 公开题集，问答配置](docs/jevbench.zh-CN.md) **171/231（74.0%）**；独立的应用状态扩展集 **30/36**。 |
| `GLM-4.7-Flash-4bit` | MLX-LM · 文本 | 完整 28 题及 36 题评测；扩展集 **21/36 正确**。 |
| `gemma-4-26b-a4b-it-4bit` | MLX-LM · 文本 | 完整 28 题及 36 题评测；扩展集 **31/36 正确**。 |
| `Qwen3-4B` | MLX-LM · 文本 | 3 题开发冒烟检查：**1/3 正确**；缓存一致性 **3/3 通过**。 |
| `NVIDIA-Nemotron-3-Nano-30B-A3B-OptiQ-4bit` | MLX-LM · 文本 | 3 题开发冒烟检查：**1/3 正确**；缓存一致性 **3/3 通过**。 |
| `qwen3.6-35b-a3b-mlx-4bit` | MLX-LM · 文本 | 3 题开发冒烟检查：**3/3 正确**；缓存一致性 **3/3 通过**。 |
| `gemma-4-e4b-it-4bit` | MLX-LM · 文本；MLX-VLM · 图片尝试 | 文本冒烟检查 **2/3 正确**，缓存一致性 **3/3 通过**；图片路径在**模型加载阶段受阻**。 |
| `Llama-3.2-1B-Instruct-4bit` | MLX-LM · 文本 | 3 题开发冒烟检查：**0/3 正确**；缓存一致性 **3/3 通过**。 |
| `Mistral-7B-Instruct-v0.3-mlx-4Bit` | MLX-LM · 文本 | 3 题开发冒烟检查：**3/3 正确**；缓存一致性 **3/3 通过**。 |
| `Qwen3-VL-4B-Instruct-4bit` | MLX-VLM · 文本＋图片 | 7 题合成开发用例：**7/7 正确**，覆盖换图、候选重排及布尔判断。 |
| `SmolVLM-500M-Instruct-4bit` | MLX-VLM · 文本＋图片 | 7 题合成开发用例：**3/7 正确**；保留误判及候选顺序敏感的记录。 |

36 题冻结评测、3 题开发检查与 7 题合成视觉检查使用不同输入，应按各自范围理解。完成完整评测的模型也出现过错误动作；小规模冒烟检查不能证明通用可靠性，也不表示同一模型家族的所有版本都兼容。

[文本模型详情与证据](docs/models.zh-CN.md) ·
[图片模型详情与证据](docs/multimodal.zh-CN.md) ·
[完整 36 题结果](docs/extended-results.zh-CN.md)

<a name="results"></a>

## 实测结果，连同局限一起公开

### JevBench 公开题集：171/231（74.0%）

本次问答提示配置使用原有量化 Qwen3.5-9B 检查点，无需额外 fine-tuning，
**171/231 题正确，准确率 74.0%**。5 次原生拒答与 1 道超过 4,096 token
上限的题目都计为错误。
这是已观察过的旧版公开题集上的本地提示实验，与下面的应用测试及当前官方
榜单分别报告。本次问答配置提供独立评测适配器，应用动作选择的默认提示保持原配置。
[完整结果、精确提示词、逐题记录与复现方法](docs/jevbench.zh-CN.md)。

### 应用状态评测：36 个场景

**新增六领域测试 · Apple M2 Max · 64 GiB · 36 个虚构英文场景。**

覆盖文档、日历草稿、文件列表、音乐队列、商品比较和设置。三个模型使用相同的冻结输入、提示词和阈值，每个模型比较四种输出方法。下面是直接评分结果：

| 本地检查点 | 严格正确 / 36 | 动作误选 / 30 | 同页新话语 p50 / p95 |
| :--- | ---: | ---: | ---: |
| Qwen3.5-9B-OptiQ-4bit | 30 / 36（83.3%） | 1 / 30 | 194.8 / 407.5 ms |
| Gemma 4 26B-A4B MoE | 31 / 36（86.1%） | 1 / 30 | 168.9 / 556.5 ms |
| GLM-4.7-Flash-4bit | 21 / 36（58.3%） | 8 / 30 | 170.6 / 330.5 ms |

36 题包含 30 个 enum 请求和 6 个 boolean 判断；动作误选只统计 enum。**这些是模型已加载、页面前缀可复用时的耗时**，不代表启动或任意页面上的请求速度。全部方法的准确率、拒绝率、动作覆盖率、布尔判断、内存和失败见[扩展评测报告](docs/extended-results.zh-CN.md)。

Gemma 选错了一次队列首项，Qwen 选错了一次最长续航产品；GLM 的动作错误更多。原来小样本中的零误操作没有延续到新场景。**当前结果不能支持无人确认的通用动作执行，也不代表任意模型兼容或固定 100 ms。**

<details>
<summary><strong>原始 28 题：保留此前结果</strong></summary>

**Apple M2 Max · 64 GiB · 28 条虚构英文测试场景 · 修订后的缓存实现。**

| 本地检查点 | 严格正确 | 动作误选 / enum 请求 | 同页面新话语 p50 / p95 |
| :--- | ---: | ---: | ---: |
| Qwen3.5-9B-OptiQ-4bit | **25 / 28（89.3%）** | **0 / 26** | **171.7 / 176.4 ms** |
| GLM-4.7-Flash-4bit | 14 / 28（50.0%） | 2 / 26 | 147.8 / 170.6 ms |
| Gemma 4 26B-A4B MoE · mixed 4/8-bit | 26 / 28（92.9%） | 1 / 26 | 134.9 / 283.9 ms |

准确率包含 26 个 enum 请求和 2 个 boolean 判断；动作误选只统计 enum。三行来自各自完整的测量记录，不能拼接为算法提速倍数。Gemma 的四种方法、三种缓存条件和全部失败见[专项报告](docs/gemma4-results.zh-CN.md)。

这里的时间要求**模型已经加载，且页面前缀可复用**。Qwen 在 KV 冷状态下的 p50 是 **2,159.9 ms**，页面更新后的首次决策是 **1,050.2 ms**，因此不能理解成每次请求都约 170 ms。

Qwen 在原始集的 3 个未通过场景包括拒绝状态区分和排序后的选项并列。Gemma 在“筛选后的第一个”上选错内容；GLM 也出现错误动作。原始集和扩展集分别报告，不合并为一个未见测试成绩。

</details>

- **161 项核心测试通过**：请求约定、状态更新、过期结果、单次执行授权、HTTP 等。它们不等于模型语义准确率。
- **168 / 168 原始场景缓存比较通过**：三个模型各 28 场景 × 2 种复用条件，和全新计算对照。
- **72 / 72 新场景缓存比较通过**：Gemma 另测 36 场景 × 2 种复用条件，最大 logit 与分数差均为 0；不代表没有语义错误。
- **浏览器记录完成 16 个场景检查**：包含真实 DOM 点击与执行回执；拒绝场景允许两种拒绝状态，标准与上面的严格质量集不同。

<details>
<summary><strong>我们具体怎么测？点击展开</strong></summary>

1. 从零编写原始 **16 条开发 + 28 条测试**，再增加独立的 **12 条开发 + 36 条测试**；冻结哈希，不导出任何生产数据。首版只验证英文输入。
2. 原始提示词在开发集调整后冻结。本次三个模型沿用同一提示词和阈值，没有根据新测试结果调整。
3. 比较直接候选评分、单编号、业务 ID JSON 和编号 JSON 四种方法；保留原始失败输出。
4. 分别记录进程启动、权重已加载但 KV 冷、同页面新话语、页面更新后首次决策。计时等待 MLX 计算完成。
5. 单独评估模型选择与执行器行为。执行器拦住错误，仍然是模型选错。

同一批 28 条场景在多个缓存条件下重复运行，不算更多独立样本。历史一编号基线达到 26/28，严格质量略高于直接返回结果；直接评分并未在所有质量和延迟指标上占优。补充 JSON 格式实验受早期测试发现启发，单独标注，不能当作完全未见的测试结果。

完整的 p50/p95、拒绝率、可执行覆盖率、内存、模型版本、输入规模、复现命令，以及原始缓存失败记录都在[中文测试说明](docs/testing.zh-CN.md)和[完整报告](docs/results.zh-CN.md)。不同代码版本的时间不能拼起来计算提速倍数。

</details>

<a name="replay"></a>
<a name="live-demo"></a>

<details>
<summary><strong>开发者集成示例：浏览器动作与执行回执</strong></summary>

虚构桌面 Morrow Studio 用于说明如何把允许动作连接到真实 DOM 控件、验证状态版本并查看执行回执。

[本地安装与集成说明](docs/http-and-demo.zh-CN.md) · [八次请求的完整录制](docs/assets/live-demo-direct/README.zh-CN.md) · [原始浏览器记录](docs/assets/browser-demo-direct/browser-transcript.json)

[静态证据回放](docs/demo/index.html?lang=zh-CN)读取保存的结果，不需要模型或服务。克隆仓库后在本地打开 `docs/demo/index.html`；GitHub 会将 HTML 文件显示为源码。输入新请求需要本地 MLX 服务。这个集成示例只操作虚构应用中预先允许的按钮，不负责任意网站导航。

</details>

## 它如何做出选择

```text
应用状态 + 自然语言 + 允许动作
              │
        官方 MLX-LM 模型
              │
       最后位置的候选 logits
              │
   selected(id) / no_match / abstain
              │
    应用校验版本 → 执行 → 回执
```

候选映射为经过 tokenizer 验证的单 token 编码，再映射回业务 ID。直接评分读取因果模型的下一 token logits，不生成 JSON 续写；没有重写官方量化输出头。混合缓存只保留完整、可复用的前缀边界。

实现细节见[架构](docs/architecture.zh-CN.md)与[官方框架审查](docs/framework-audit.zh-CN.md)。首版范围是**英文、单轮、单步选择**，实验性的文本＋图片决策使用独立的 [MLX-VLM API](docs/multimodal.zh-CN.md)。不包含通用聊天、多步规划、任意参数生成、训练或 GPU 批处理。**中文文档不表示中文模型能力已经验证。**

<a name="related-projects"></a>

## 相关决策项目：AnyJev、OpenDecider 与 Laya

正在评估 **AnyJev**、**OpenDecider**、**Laya** 或 **Laya-MLX** 的开发者，也可能需要本地动作选择、分类或语义路由。**MLX Direct Decider 聚焦 Apple Silicon 上已有兼容的因果 MLX-LM 检查点，无需 fine-tuning 或单独的决策模型权重。** 应用定义候选项，通过原模型输出头为其评分。AnyJev 也提供无需训练的路径，各自的实现和分数含义有所不同。

| 项目 | 文档中的实现方向 | 与 MLX Direct Decider 的技术联系 |
| :--- | :--- | :--- |
| [AnyJev](https://github.com/nokia-applied-research/AnyJev) | 无需训练的 `Decider` 包含候选轮换与标签先验校正；另有自蒸馏 `Tacit` 模型路线。 | 同样复用已有 LLM；MLX Direct Decider 提供原生 MLX 评分及应用执行保护，未包含 AnyJev 的 L0 校正。 |
| [OpenDecider](https://github.com/manjunathshiva/opendecider) | 提供带类型输出及服务客户端的开放权重 System 1 决策模型。 | 同类决策模型方案；MLX Direct Decider 使用自己的请求和执行约定。 |
| [Laya](https://github.com/NandhaKishorM/laya) · Convai Innovations | 基于编码器，回答选择、有序评分和是非问题。 | 相关的带类型决策方法；MLX Direct Decider 从因果语言模型的选项 token logits 评分。 |
| [Laya-MLX](https://github.com/mizorewww/laya-mlx) | 在 Apple Silicon 上运行 Laya 编码器和决策头的独立 MLX 实现。 | 共享 MLX 平台和本地决策场景，模型架构及 API 各自独立。 |

[AnyJev、OpenDecider 与 Laya 对照](docs/decision-models.zh-CN.md)列出了来源快照、集成边界与评测注意点。这些是相关项目，不代表已内置相应后端或存在厂商合作。

<a name="faq"></a>

## 常见问题

### MLX Direct Decider 如何使用 MLX、MLX-LM 和 MLX-VLM？

MLX 提供数组与计算框架，MLX-LM 运行文本模型，可选的 MLX-VLM 后端运行文本／图片模型。Metal 执行由 MLX 负责。图片决策使用全新上下文，现有文本后端保留前缀复用。MLX Direct Decider 在此基础上提供应用定义的候选项、候选评分、拒绝结果和版本化执行。直接决策读取经过验证的单 token 选项编码对应的下一 token logits，不生成 JSON 续写。实现细节见[架构](docs/architecture.zh-CN.md)。

完整模型仍然处理输入。读取最后位置的 logits，不代表只运行最后一层或替换输出头。

### MLX Direct Decider 可以用于本地语义路由吗？

可以。把当前允许的路由或动作描述为带稳定业务 ID 的候选项，提供应用状态和用户请求，然后读取选中的 ID。动作由宿主应用定义和执行；不合适或含糊的请求可以返回 `no_match` 或 `abstain`。接入方法见 [Python API](docs/python-api.zh-CN.md)。

### MLX Direct Decider 和 AnyJev 有什么区别？

两者都可以复用已有模型，无需额外训练。MLX Direct Decider 聚焦 Apple Silicon 上的原生 MLX-LM 与实验性 MLX-VLM 推理，提供动态动作 ID、文本前缀复用和版本化执行。AnyJev 的 `Decider` 增加候选轮换与标签先验校正。本库使用直接候选评分，分数不是经过校准的正确概率。详见 [MLX Direct Decider 与 AnyJev 对照](docs/decision-models.zh-CN.md#mlx-direct-decider-vs-anyjev)。

### MLX Direct Decider 与 AnyJev、OpenDecider、Laya-MLX 有什么区别？

MLX Direct Decider 为已有兼容的因果 MLX-LM 检查点增加有限候选动作选择，无需微调或专用决策权重。AnyJev 的 `Decider` 同样无需训练地复用已有 LLM，并校正候选顺序和标签先验。OpenDecider 与 Laya 发布训练好的决策模型系列，Laya-MLX 运行 Laya 权重；它们的用户可以直接加载已发布检查点，不必亲自训练。详见[决策模型对照](docs/decision-models.zh-CN.md)。

### 安装 MLX Direct Decider 后，所有 MacBook 模型都能直接使用吗？

使用 `--model` 或 `MLX_DIRECT_MODEL` 指向兼容的本地 MLX-LM 检查点即可接入。加载接口跨模型架构复用，但每个检查点仍需满足架构实现、分词器、聊天模板、logits 访问和内存要求。安装软件包不会自动接管其他应用，也不能直接加载 Ollama 的 GGUF 模型。[兼容性说明](docs/models.zh-CN.md)区分了完整评测、小规模运行检查和尚未实测的模型家族。

### 已经评测了哪些 MLX 模型？

[首页模型表](#models)列出全部 11 个本地实测检查点，覆盖 Qwen、GLM、Gemma、NVIDIA Nemotron、Llama、Mistral 与 SmolVLM，标明文本／图片路径及加载失败。[文本兼容性指南](docs/models.zh-CN.md)和[多模态指南](docs/multimodal.zh-CN.md)保留精确版本、完整结果与局限；完整评测和小规模冒烟检查分别标注。

### 需要 fine-tuning、专用决策模型权重或云端 API 吗？

不需要。复用已有兼容的 MLX-LM 检查点，或用于实验性图片决策的 MLX-VLM 检查点，在 Apple Silicon Mac 上本地运行即可。无需云端 API、额外训练、微调、新决策头或专用决策检查点。已有兼容基础权重时可以直接复用；否则需要先单独获取基础权重，再把本地目录传给引擎。软件包不负责下载权重。

### 如何安装 MLX Direct Decider？

创建原生 ARM Python 3.11+ 环境，运行 `python -m pip install "mlx-direct-decider[mlx] @ git+https://github.com/ItsOdeLeo/MLX-Direct-Decider@v0.2.0"`；如需图片决策，把 `[mlx]` 换成 `[mlx,vlm]`。将 `MLX_DIRECT_MODEL` 设为本地检查点目录。[快速开始](#quick-start)提供可运行示例。0.2 是实验版。如需从源码开发，可克隆 [ItsOdeLeo/MLX-Direct-Decider](https://github.com/ItsOdeLeo/MLX-Direct-Decider)，在仓库目录运行 `python -m pip install -e '.[mlx]'`。

### 本地 MLX 动作选择有多快？

延迟取决于检查点、硬件、输入和可复用前缀。[实测结果](#results)区分了权重已加载且页面前缀可复用、启动、KV 冷和页面更新等情况。热缓存时间不是固定延迟保证，也不能证明无人监督执行的可靠性。

### MLX Direct Decider 是官方 JEV 实现或通用聊天助手吗？

MLX Direct Decider 是位于 [ItsOdeLeo/MLX-Direct-Decider](https://github.com/ItsOdeLeo/MLX-Direct-Decider) 的独立 JEV 启发项目。它使用已有 MLX-LM 模型，不使用 JEV 权重，也不复现未公开的 JEV 训练方法。0.2 支持英文、单轮、单步选择，通用聊天和多步规划不在范围内。实验性的文本／图片候选选择见[多模态指南](docs/multimodal.zh-CN.md)。

## 继续阅读

| 文档 | 内容 |
| :--- | :--- |
| [中文测试说明](docs/testing.zh-CN.md) | 怎么测、测到了什么、失败在哪里、如何复现 |
| [Python API](docs/python-api.zh-CN.md) | enum / boolean、返回字段、版本化执行 |
| [支持模型](docs/models.zh-CN.md) | 检查点、量化、依赖、许可证与限制 |
| [AnyJev、OpenDecider 与 Laya 对照](docs/decision-models.zh-CN.md) | 同类决策项目、MLX 运行时与集成边界 |
| [评测方法](docs/evaluation.zh-CN.md) · [完整结果](docs/results.zh-CN.md) | 全部基线、原始记录、版本与失败 |
| [本地 HTTP](docs/http-and-demo.zh-CN.md) | CLI、接口和真实浏览器操作 |
| [贡献指南](CONTRIBUTING.zh-CN.md) · [发布说明](docs/releasing.zh-CN.md) | 开发、构建、发布边界 |

<details>
<summary><strong>运行开发检查</strong></summary>

```sh
python -m pip install -e '.[dev]'
python scripts/check_docs.py
python -m pytest -m 'not model'
python -m ruff check src tests benchmarks scripts examples
```

真实模型测试需要显式指定本地权重，顺序运行模型任务。GitHub CI 在 Python 3.11 和 3.13 上执行核心检查并构建发行包；本地 MLX 推理单独验证。

</details>

<a name="connect"></a>

## 联系作者 · 支持项目

欢迎分享使用场景、复现结果和改进建议。你也可以通过提交 issue、修正文档或提供新的公开测试场景支持项目。

<p align="center"><strong>欧的Leo</strong></p>

<p align="center"><a href="https://x.com/ItsOdeLeo">X / Twitter · @ItsOdeLeo ↗</a> · <a href="https://xhslink.com/m/18bjTTf180W">小红书 · 欧的Leo ↗</a></p>

<p align="center">小红书号：<code>ItsOdeLeo</code></p>

<p align="center"><sub>欢迎 Star、分享或提交 PR，一起改进本地语义决策。</sub></p>

## 许可证与来源

[MIT](LICENSE.zh-CN.md)。项目独立维护，受 [TypeSafe AI 的 JEV](https://docs.typesafe.ai/) 启发，使用官方 [MLX-LM](https://github.com/ml-explore/mlx-lm)。没有官方合作或背书，不使用 JEV 权重，也不声称复现未公开的 RLCD。

模型和依赖保留各自许可证；参考项目与归属见 [NOTICE](NOTICE.zh-CN.md)。已有实测材料保留更名前的名称、源文件路径和哈希，详见[结果来源说明](docs/results.zh-CN.md)。候选名称的检索记录见[命名说明](docs/naming.zh-CN.md)。

<p align="center"><sub>Local decisions. Visible evidence. Open source.</sub></p>
