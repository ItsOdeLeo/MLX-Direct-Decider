# 变更记录

[返回项目首页](README.zh-CN.md)

## 0.2.0 — 2026-11-12

首个正式版本，发布在 GitHub Releases。安装方式：`pip install "mlx-direct-decider[mlx] @ git+https://github.com/ItsOdeLeo/MLX-Direct-Decider@v0.2.0"`。

- `MLXLMBackend`、`from_loaded_model` 和 `MLXDecisionEngine` 新增可选参数
  `prompt_layout="state_first"`：状态放在问题前面，共享「系统提示 + 状态」快照，
  对同一状态提多个问题时状态只预填充一次。结果的 `cache` 信息新增 `layout`、
  `shared_tokens` 两个字段和 `shared_state` 作用域。默认的 `question_first` 提示与缓存
  行为不变。见 [Python API](docs/python-api.zh-CN.md#对同一状态连续提多个问题)。

- 项目及仓库更名为 MLX Direct Decider。发行包与 CLI 为 `mlx-direct-decider`，
  导入名为 `mlx_direct_decider`；保留 `mlx_zeroeffort_decider` / `mlx-zeroeffort-decider`
  及 `jev_mlx` / `jev-mlx` 兼容入口。
  当前页面和界面使用新名称，benchmark 证据、录像及早期发布记录保留历史名称。
  不涉及 fine-tuning 或模型行为变化。

- 可选的 MLX-VLM 文本／图片决策：使用原模型输出头，无需微调；按次接收本地图片并计算全新上下文，提供 Python API 与 CLI。保留 Qwen3-VL、SmolVLM 的冒烟结果与错误，明确 Gemma VLM 的加载限制。
- 专属 Idefics3 PIL 处理器适配，以及针对 Mistral SentencePiece 边界行为的上下文候选 token 校验；保留现有文本缓存策略。
- 双语[运行组件与多模态指南](docs/multimodal.zh-CN.md)、合成图片示例、可复现检查命令及检查点证据。

## 0.1.0 — 已准备的发布版本

JEV MLX 的首个实验实现。准备源码版本与公开发布到仓库或包平台分别记录。

- 新增原创方块游戏 JEV MLX Blocks：本地模型选择落点，支持手动游戏、过期结果保护与完整记录下载。六项规则测试及十四项模拟响应的浏览器检查通过。一次 Qwen 开发运行完成 20 次落块、消除四行；同时保留首次拒绝、全部决策、原始录像和明确标注 4 倍速的 GIF，见[游戏报告](docs/blocks.zh-CN.md)。这不是冻结的游戏能力基准。
- 英文默认的 README 和独立中文首页，两种语言均有完整文档；语言入口位于标题与导航区下方，X 排在小红书前面，保留原创标题图及作者提供的社交主页链接与名片。展示名称为 JEV MLX，包与 API 名称保持 `jev-mlx` / `jev_mlx`。
- 增加开发者集成演示，在集成文档中提供八次真实 Qwen 请求的录制与原速 GIF。保留完整视频、全部响应与执行回执、源码哈希及录像恢复记录；提供录制和转码脚本。这组展示流程与质量 benchmark 分开。
- 离线中英文浏览器录制回放，保留全部 16 次决策及独立的状态变化检查。无需模型或服务即可运行，并明确区别于实时推理。
- 项目当前对外名称为 JEV MLX，曾使用 JEVKit MLX 和 MLXJ。仓库为 [CoderInPajamas/JEV-MLX](https://github.com/CoderInPajamas/JEV-MLX)。发行包和 CLI 为 `jev-mlx`，模块为 `jev_mlx`；当前模型变量为 `JEV_MLX_MODEL` 和 `JEV_TEST_MODEL`。原始 benchmark 元数据、源码哈希、截图与录像保留历史名称和内容，见[结果](docs/results.zh-CN.md)。
- 可安装的 Python 包，支持动态 enum 和有类型的 boolean 请求、不匹配与弃权、logits、候选范围内分数、原始 margin、模型身份、实际完成计算后的计时，以及缓存元数据。
- 通过官方 MLX-LM 加载模型并使用原始输出头对最后位置评分，候选使用 tokenizer 验证过的单 token 编号，本地加载权重。
- 官方前缀缓存快照、资源上限、预热、状态更新及串行推理。
- 带版本的会话，保护过期结果并提供一次性执行授权，包括拒绝伪造、修改和重放的结果。
- CLI 和 localhost HTTP 服务，提供虚构的 Morrow Studio 桌面、可筛选排序的课程库、模拟播放器，以及模型选择的真实 DOM 按钮点击和独立执行回执。
- 核心测试、显式启用的真实模型测试、虚构开发集与冻结测试集、数值对照工具、单编号与 JSON 基线、进程冷启动测量及完整失败记录。
- 验证了官方 MLX-LM 中的 Gemma 4 MoE 文本推理，保留原始集的四方法、各缓存条件证据，单独的进程冷启动，以及 1,741-token 旋转缓存集成检查。模型限制与错误选择完整记录在 [Gemma 报告](docs/gemma4-results.zh-CN.md)。
- 增加单独冻结的六领域虚构扩展集，含 12 个开发场景、36 个测试场景。Gemma、Qwen、GLM 均完成同页新话语条件下的四方法对比。直接评分严格正确数分别为 31/36、30/36、21/36；enum 动作误选分别为 1/30、1/30、8/30。布尔答案独立统计。[扩展结果](docs/extended-results.zh-CN.md)保留全部失败，并说明这些结果为什么不支持通用的无人确认动作执行。
- 支持指定场景目录、模型加载前拒绝重复的实验调度项、显式冷启动调度元数据，以及按请求类型拆分的证据报告和审计。最新核心测试有 161 项通过。独立 Gemma 扩展缓存阶段完成 108 次决策和 72 次通过的数值对照；实验计划中的全部 1,096 次正式测量均已完成。
- 重新构建 wheel，在源码仓库外的新环境安装其 MLX 依赖，核对 14 个运行文件与源码及 wheel 中字节一致。一次真实 Gemma CLI 调用为公开示例选择了 `player.pause`。这是安装冒烟验证，不是额外的质量 benchmark，见[发行包验证](benchmarks/results/extended-release-checks/distribution.json)。
- 架构、官方框架审查、模型支持、评测协议、实测结果、贡献指南和发布说明。

实际运行见[结果](docs/results.zh-CN.md)，各检查点的验证范围见[模型支持](docs/models.zh-CN.md)。本版本不宣称任意模型兼容、置信度已校准、固定延迟、零错误，或复现 JEV 权重与训练。GPU 批处理、任意网站自动化、训练、通用聊天和多步规划仍在范围之外。
