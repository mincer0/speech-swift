# MiniCPM-o 原生双工移植：优化与质量报告

日期：2026-09-21 ~ 09-22
代码：`runtime/speech-swift-minicpm-native`，分支 `feature/minicpm-o-native`
对照基准：Comni（官方 llama.cpp-omni + Q4_K_M）

---

## 0. 一句话结论

**整 unit 耗时 1072ms → 596ms（-44%），首次低于实时并留出 1.68× 余量；
文本质量从"重复字/乱码/吞字"到连贯可听；所有剩余性能与听感现象都已用
官方实现或实测数据定性 —— 移植层面的优化到此见底。**

---

## 1. 起点与终点

### 1.1 性能弧线（`wall_clock_ms`，19 个发声 unit 的场景）

| 阶段 | wall_clock | audio_encoder | 关键改动 |
|---|---|---|---|
| 起点 | 1072 | 379.6 | — |
| ① 音频前端交回 MLX（Linear+Conv） | 762 | 113.4 | `MINICPM_AUDIO_FAST` |
| ② 同上 + GELU/LayerNorm | 739.8 | 91.6 | 同上扩围 |
| ③ flow ODE 步数 10→8 | 646.2 | 91.6 | `FlowConfiguration.odeSteps` |
| ④ flow 循环改 `asyncEval` | — | 91.6 | `DiTFlow.generate` |
| **⑤ 注意力去 MPSGraph** | **596** | **23.8** | **AUDIO_FAST 扩围至注意力** |
| 官方 llama.cpp-omni 对照 | 224（SPEAK 块） | 42.2 | — |

**实时余量：0.93× → 1.68×。**

### 1.2 首响应延迟（P7 悬案结案）

| 来源 | 首音频延迟 |
|---|---|
| P7 交接文档（优化前） | 6.1–6.4s（最差 12.89s） |
| **优化后实测** | **0.55–1.0s**（最好 0.55s，中位 ~0.85s） |

人类对话轮换间隔约 0.5–1s → **已达标，该争议可以结案**。

---

## 2. 落地的改动（按收益排序）

### 2.1 音频编码器注意力去掉 MPSGraph（-60ms，收益最大）

**问题**：`MINICPM_AUDIO_FAST` 当初覆盖了 Linear / Conv / GELU / LayerNorm，
**漏掉了注意力**，而它默认走 `MiniCPMAudioBF16Attention` —— 每次调用执行一个
MPSGraph。24 层 × 每 unit 一次，开销巨大。

**发现方式**：不是靠猜，而是提问 —— *"为什么我们的 MLX 编码器（83.5ms）
比 CoreML 的纯 CPU 路径（22.8ms）还慢 3.7 倍？"*

**改动**：`useMPSGraphAttention` 增加 `&& !audioFastMathOverride`。
`MINICPM_AUDIO_FAST=1` 现在覆盖 Linear + Conv + GELU + LayerNorm + **Attention** 五项。

**实测（3 次重复）**：`audio_encoder_ms` 83.5 → **23.6 / 23.9 / 23.8**（±0.2ms）；
`wall_clock` 646–700 → **599.0 / 603.0 / 587.4**。

**23.8ms 已追平 CoreML 纯 CPU（22.8ms），超过官方 llama.cpp（42ms）。**
理解能力验证通过（`duplex_understand_test.py` 命中 宇航员/火星/土豆/科幻）。

### 2.2 音频前端交回 MLX 内建（`MINICPM_AUDIO_FAST`）

原实现为数值 parity 写了四个手写 Metal kernel（Linear / Conv1d / GELU / LayerNorm），
外加一个 MPSGraph GELU。全部替换为 MLX 内建算子后：**379.6 → 91.6ms（-76%）**。

### 2.3 flow-matching ODE 步数 10 → 8

`FlowConfiguration.odeSteps`。`flow_ms` 169.7 → 131.3（与"2 步 × 17ms"理论值吻合）。
**代价**：音质需人工试听确认（本次未听到明显劣化，但请保持警觉）。

### 2.4 flow ODE 循环逐步阻塞同步 → `asyncEval`

`DiTFlow.generate` 原来每步 `eval(value)`，把 8 步严格串行化。
改 `asyncEval`（仍物化结果、仍限图规模，但不阻塞）。
**`flow_ms` 131.3 → 125.0（3 次 ±0.2ms）**，`token2wav_ms` 166.4 → ~160.5。

---

## 3. 质量修复（听感问题）

| # | 问题 | 根因 | 修法 |
|---|---|---|---|
| 1 | 文本重复字、垃圾 token | 采样器 repetition-penalty 符号错误 | 修正符号 |
| 2 | 句中"拆状态"、吞音节 | 把"解码器自停"当成了 turn 边界 | `finalChunk = endOfTurn` |
| 3 | 密集 unit 下文本截断 | 每 unit 文本预算 28 字，实际只能说出 ~6 字 | 预算降至 6 字 |
| 4 | 词尾吞音 | 未遵守官方 26/25 look-ahead 契约 | 保留隐藏 code、按提交前缀推进 |
| 5 | listen 段在客户端时间轴上留洞 | 引擎的 1 秒静音载荷被丢弃 | 下发静音块 |
| 6 | **模型自行开场 / 说"听不清" / 长故事讲不完** | **系统提示词缺官方双工纪律段** | 补全 `DuplexAdapter` 提示词 |

第 6 项一次解释三个症状：

> 对话必须始终由用户发起，**绝对不要主动开场或追问**。只有听到用户清晰说话时才回答；
> 回答完后保持倾听。……**噪声、静音和扬声器中你自己的声音都必须继续倾听。**

（补的是"模型不知道要无视自己的回声"这一条 —— 扬声器外放时它把自己念的话当成了用户输入。）

---

## 4. 被数据否决的方案（**本节最重要：防止重犯**）

| 假设 | 实测结果 |
|---|---|
| 关 `MINICPM_STAGE_TIMING` 借懒调度提速 | 715.9 vs 646.2ms，**无收益** |
| `listen_prob_scale=0.5` 抑制 listen | A/B 否定：1.0 反而更长更连贯 |
| 抑制自然 `chunk_eos` 逼模型多说 | 78 unit 逐字统计，文本量**不变**，且产出乱码 |
| base64 f32 传输（128KB/块）拖慢浏览器 | 每秒仅几毫秒解码量，不成立 |
| 客户端响应窗口掐输入导致断续 | 输入节奏实测 1.03s 稳定，不节流 |
| 块内 ~35% 静音是移植缺陷 | 官方填充率 72% vs 我们 60–70%，**模型天性** |
| token2wav/TTS 藏着手写标量 kernel | 排查干净：无自定义 kernel、无 MPSGraph |
| 变长块（早停）能省计算 | **早停会产生整秒近静音**（RMS<0.01），文本根本没念出来 |
| 变长块（去填充）有用 | 实测 17/17 块仍是 1.00s，**净收益 0** |
| LLM 4-bit 量化 | 用户否决（质量风险 vs ~150ms） |
| 语音组件 8-bit 量化 | 误差 0.54%/0.77%，但收益按带宽重估**只有 ~50ms**；且本仓库已有先例（TTS 8-bit 变体因"可听劣化"被撤下） |
| TTS 26 步循环上 `MLX.compile` | **受阻**：decoder 就地改写 KV cache，非纯函数，需整条路径函数式重构 |
| 音频编码器注意力换融合 SDPA | **慢 17%**（28.6 vs 23.6ms）：50 位置 × 16 头 × d=64 规模太小，flash-attention 分块/重缩放纯属开销 |
| **上 ANE（CoreML）** | 转换与验证都成功（10.4ms，真实语音 cos 0.999955），但注意力修复后**只剩 13.4ms（2.2%）空间**，代价是 583MB 依赖 + 9.6s 冷启动 + Swift 集成 → **性价比不成立** |
| LLM 的 exact-BF16 parity kernel 在拖后腿 | **全部门控在 `quantBits == 0`**，8-bit 模型下全关 |

---

## 5. 方法论（可复用）

1. **"为什么我们比一个更弱的执行路径还慢？"是最有效的诊断提问。**
   本轮最好的两次收获（MPSGraph GELU、MPSGraph Attention）都源于此，合计约 110ms。
   而"这个模块有什么可以优化的"这类提问，命中率低得多。

2. **`wall_clock` 单次测量不可作对比依据**（运行间方差 ±30ms）；
   **阶段级指标（如 `flow_ms`）方差可小到 ±0.2ms**，才是可靠信号。
   任何 A/B 至少要 3 次重复。

3. **真实数据 vs 分布外数据**：ANE 验证在随机噪声上只有 cos 0.9983（看着"可能有问题"），
   在真实语音上是 **0.999955**（0.45% 误差）。**分布外输入会放大表观误差**，
   判据必须用真实数据。

4. **先测量再实现**：量化方案在动手前先算了误差，结果直接把预期收益从 120ms 修正到
   50ms 并发现质量风险 —— 省下了一整轮无用功。

---

## 6. 工具（`tools/`）

| 工具 | 用途 |
|---|---|
| `duplex_ab_listen.py` | 直连 `/v1/realtime`，脚本化场景 + 分阶段耗时聚合（性能自验证） |
| `duplex_understand_test.py` | 喂已知语音（macOS `say`），验证理解能力是否退化 |
| `mps_fill_probe.py` | 驱动 Python 双工运行时测块内填充率 |
| `quantization_error_probe.py` | mlx 离线量化误差评估（不写文件） |
| `scripts/export_minicpm_o_audio_coreml.py` | 音频编码器导出 CoreML/ANE（含四个工具链障碍的解法） |
| `scripts/verify_minicpm_o_audio_coreml.py` | 冷启动 + 真实语音数值验证 |

**环境注意**：
- 本机沙箱的 `SAFE_DELETE_BULK_CONFIRM_REQUIRED` 会在 import 阶段杀掉 Python 进程
  （librosa 清 `__pycache__`），**CoreML 转换类脚本必须免沙箱运行**；
  `PYTHONDONTWRITEBYTECODE=1` 无效。
- 与 Comni 同跑会抢 GPU（`llm_decode` 曾被抬到 1082ms），**性能对比必须错开**。
- Swift 里多行可选链后接 `==` 的 `else if` 无法编译，需先算成局部变量。

---

## 7. 为何到此为止

剩余三个大阶段的现状：

| 阶段 | 耗时 | 定性 |
|---|---|---|
| `llm_decode` | 250–263ms | **真实的 8-bit 权重带宽成本**。审计确认：8-bit 路径下所有 exact-BF16 parity kernel 均已关闭，单 token 解码已走融合 SDPA。 |
| `semantic_tts` | ~180ms | 26 步自回归 + 每步一次主机同步（采样必需）；无隐藏 kernel；`MLX.compile` 受限于 KV cache 非纯函数 |
| `token2wav` | ~160ms | flow 125（已减步数、去逐步同步）+ HiFT 35；模块干净 |

**继续优化的三条路都已被评估并否决**：
- 换精度（4-bit / 语音组件 8-bit）→ 用户否决 / 收益仅 50ms 且项目内已有劣化先例
- 换执行单元（ANE）→ 注意力修复后仅剩 13.4ms 空间，代价过高
- 重构（TTS 函数式化、多 unit 流水线）→ 动已数值验证的路径，风险与收益不成比例

**结论：在"不改精度、不换执行栈、不重构已验证路径"的约束下，移植层面的优化已经见底。**

---

## 8. 若未来要再推进一步

按性价比排序，供决策：

| 方向 | 收益 | 前提 |
|---|---|---|
| 多 unit 流水线（双缓冲） | wall_clock → ~400ms 级 | 接受重构引擎主循环的回归风险 |
| 音频编码器上 ANE | 13.4ms（2.2%） | 除非同时做流水线改造，否则意义有限 |
| 语音组件 8-bit 量化 | ~50ms | 必须有听感 A/B 机制与回退方案 |
| LLM 4-bit | ~150ms | 用户已否决；若重启必须先过理解能力 A/B |

**建议：除非出现明确的产品需求（更低首响应延迟、更大模型、更长上下文），
否则维持现状。** 当前状态（596ms / 1.68× 实时余量 / 首响应 <1s / 质量达标）已满足实时双工对话的需要。
