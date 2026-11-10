# JevBench 公开题集：74.0%，无需额外 fine-tuning

[返回项目首页](../README.zh-CN.md) · [机器可读报告](../benchmarks/results/jevbench-20261015/report.json)

2026 年 10 月 15 日，MLX Direct Decider 使用原有的
`mlx-community/Qwen3.5-9B-OptiQ-4bit` 检查点，在 Apple M2 Max、64 GiB
统一内存的 Mac 上完成旧版 JevBench 全部公开题：**171/231 正确，准确率 74.0%**。
本次使用独立的 `qa_question_slot` 问答评测配置，**没有额外 fine-tuning、LoRA、
训练决策头或新增专用决策权重**，量化检查点保持不变。

这是**已观察过的公开题集上的本地提示实验**，尚非官方榜单成绩，也不是未见测试集
上的结果。[当前官方方法](https://benchmarkheaven.com/jev-models/methodology)使用包含
封闭题的 1,500 道决策题，并采用 Capability／Composite 等指标；这里的 231 题
原始准确率不能直接换算为当前官方排名。

## 结果与逐题分歧

| 旧版公开题分组 | 原应用提示 | 本次问答配置 |
| :--- | ---: | ---: |
| Easy | 47/48（97.9%） | **48/48（100.0%）** |
| Original | 55/72（76.4%） | **63/72（87.5%）** |
| Hard | 55/111（49.5%） | **60/111（54.1%）** |
| **总计** | **157/231（68.0%）** | **171/231（74.0%）** |

新配置修正 **23 道**原错题，新增 **9 道**错误，净增 **14 道正确答案**，提高
**6.1 个百分点**。[完整报告](../benchmarks/results/jevbench-20261015/report.json)
保留全部 47 次预测变化，包括新旧都答错但答案不同的情况。全部 231 题都计入分母；
4 次原生 abstain、1 次 no-match 和 1 道超过 4,096 token 的题都按错误计算。
没有基础设施故障，没有筛选题目或重试失败请求。

排除模型加载和 4 道合成适配器检查后，串行进程内延迟为 **p50 1.214 秒／p95
13.750 秒**；原配置为 1.234 秒／11.283 秒。准确率提高了，p95 延迟也变长了。
这些耗时混合了新题与可用的前缀复用，不是同页面请求的固定延迟保证；本地算力也
不等于零成本。

## 改了什么，以及实验局限

实际问题放进原生 `QUESTION` 栏，用户话语改为固定说明。超过该栏 2,048 字符上限
时，完整问题仍留在 `USER_UTTERANCE`，由固定说明指向它，绝不截断。
[冻结的问答提示](../benchmarks/jevbench/profiles.py)把候选项视为答案，区分信息问答
与动作命令，同时保留 false、零值、明确的未知答案、no-match 和 abstain。
输入状态、候选顺序与描述、原生单 token logits 评分、条件 softmax、阈值和量化
权重都没有改变，也没有拟合温度。分数不是经过校准的正确概率。

这是独立的串行评测适配器，**应用动作选择的默认提示保持原配置**。实验提示上下文
会暂时修改模块全局变量，因此不能与其他原生决策调用在同一进程内并发运行。

首次只调整系统提示的实验，开发样例从 19/24 提高到 22/24，验证样例仍为 30/36。
第二轮选择的方案，开发样例从 26/36 提高到 32/36，验证样例从 27/36 提高到 28/36。
这些合成开发／验证样例共用模板，验证只多对一题，不能据此声称广泛泛化。
旧版公开题的基线已经看过；选定提示后只做一次公开题复测，没有根据复测结果继续
挑选或修改提示。因此仍需要新的、未观察过的题集来验证实际泛化。

## 复现与证据

模型固定为 [OptiQ 检查点修订
76b3310](https://huggingface.co/mlx-community/Qwen3.5-9B-OptiQ-4bit/tree/76b3310ab7aa52a34303c66fc928b6d7239c860c)，
混合 4/8-bit affine、group size 64；实测核心代码为
[`7b1d941`](https://github.com/ItsOdeLeo/MLX-Direct-Decider/tree/7b1d9413a590156570e3f109adcfdaf4ec085f19)。
Python 3.13.2、MLX 0.31.2、MLX-LM 0.31.3、Transformers 5.9.0、Tokenizers 0.22.2。
公开适配器从实测代码提取，只为可移植性调整本地路径与元数据说明，保留有效提示和
请求映射。数据、配置、分词器及公开文件的哈希见报告与
[文件清单](../benchmarks/results/jevbench-20261015/manifest.json)。

在已安装项目依赖、已下载固定版本模型的 Apple Silicon Mac 上运行：

```sh
git clone https://github.com/fstandhartinger/jevbench.git runs/jevbench-upstream
git -C runs/jevbench-upstream checkout --detach 609a664e81f46e03731600bd7bd9f1b7a61257c5
export MLX_DIRECT_MODEL=/absolute/path/to/Qwen3.5-9B-OptiQ-4bit
PYTHONPATH="runs/jevbench-upstream:src:." python -m benchmarks.jevbench.run \
  --model "$MLX_DIRECT_MODEL" \
  --jevbench-root runs/jevbench-upstream \
  --profile qa_question_slot \
  --output runs/jevbench-qa-reproduction
```

用 `--profile released` 和另一个输出目录可复现原应用提示配置。适配器保留所有
拒答和失败；公开的是逐题实测字段，旧的 runtime 描述已去除，以免误指向原提示。
原始响应仍在本地归档，未把其他未提交的开发改动混入本次结果。

[新配置逐题结果](../benchmarks/results/jevbench-20261015/qa-results.jsonl) ·
[原配置逐题结果](../benchmarks/results/jevbench-20261015/baseline-results.jsonl) ·
[完整报告与实验选择记录](../benchmarks/results/jevbench-20261015/report.json)
