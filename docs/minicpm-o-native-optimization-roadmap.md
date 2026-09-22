# MiniCPM-o 原生双工：未来优化路线图

更新：2026-09-21
适用：`runtime/speech-swift-minicpm-native`（分支 `feature/minicpm-o-native`）

---

## 当前基线

| 指标 | 起点（本轮优化前） | 现在 | 官方 llama.cpp-omni（Q4_K_M） |
|---|---|---|---|
| 音频编码 | 379.6ms | **91.6ms**（-76%） | 42.2ms |
| LLM 解码 | ~282ms | 282ms | ~86ms |
| TTS + token2wav | 389ms | 389ms | ~96ms |
| **整 unit（wall_clock）** | **1072ms** | **646ms**（-40%） | **224ms（SPEAK 块）** |
| 实时余量 | 0.93×（欠实时） | **1.55×** | 4.4× |

质量侧已修复：重复字/垃圾 token（采样器符号错误）、句中拆状态（`finalChunk`）、
首块整秒静音、密集 unit 截断（文本预算 28→6 字）、listen 时间轴空洞（静音块下发）、
轮次纪律（系统提示词补全官方双工纪律段）。

对照数据源：Comni 会话 `20260921_010800_adx_mua2kzwy`
（`/Applications/Comni.app/Contents/Resources/apps/server/data/sessions/`）。

---

## 待办项（按建议顺序）

### 1. 静音块发生率统计 —— 低成本，先做

**现象**：偶发"文本非空但波形近乎静音"的块（历史样本：unit 13「当然」0.08s、
unit 69「比」0.00s —— 25 个 code 全部提交，波形却静音）。

**做法**：给 `tools/duplex_understand_test.py` / `duplex_ab_listen.py` 加一个
"文本非空但有效语音 < 0.25s"计数器，自跑 5 轮取发生率，再与
unit 文本长度 / turn 内位置 / chunk 序号做相关性。

**产出**：发生率基线 + 触发模式。若与短 condition 强相关，则指向
flow 解码器缺少"开始发声"锚点，修法是把该 unit 文本并入下一 unit 重合成。

### 2. 音频编码器上 ANE —— 中大工程，收益 ~90ms 并释放 GPU

**背景**：ANE（Apple Neural Engine）是 M 系列芯片里与 CPU/GPU 并列的第三块
专用神经网络加速器，**与 GPU 相互独立、可同时工作**。

**现状**：音频编码器跑在 GPU 上（MLX），92ms 纯加在关键路径上 ——
它与 LLM/TTS/token2wav 抢同一块芯片：

```
现在：GPU 串行 [编码 92][LLM 282][TTS+token2wav 389] = 763ms
ANE 后：ANE [编码 ~20] ∥ GPU [LLM 282][TTS+token2wav 389] ≈ 671ms
```

**做法**：
1. 用 `coremltools` 把 `MiniCPMAudioModel`（24 层 / hidden 1024）导出为
   Core ML（fp16），编译成 `.mlmodelc`
2. Swift 侧加一个执行分支，仿照既有的 `vision_backend: coreml|metal` 开关
3. 用 `tools/duplex_understand_test.py` 验证理解能力不退化（ANE fp16 会带来
   数值漂移 —— 这是必须守住的关口）

**有利条件**：
- 音频块是**固定形状**（每 unit 恒定 1 秒 → 固定帧数），正是 ANE 最喜欢的形态，
  无需动态 shape 适配
- 官方生态有先例：Comni 注册表里的
  `vision/coreml_minicpmo45_vit_all_f16.mlmodelc` 就是 Vision 编码器的 ANE 版

**风险**：数值漂移伤理解（探针可自动把关）；模型变更后需重新转换；多一个后端要维护。

### 3. semantic_tts 批量化 —— 高难度，收益不确定

**现状**：TTS 语义解码器自回归生成 26 个 code，183ms / 26 ≈ **7ms/code**。
每步都必须：前向 → 采样 → **同步回 CPU 读 token** → 反馈为下一步输入。
逐 token 的同步是自回归的固有属性。

**可能的方向**（均未验证）：
- **投机解码**：小草稿模型猜多个 code，一次批验证 —— 需要额外模型，复杂度高
- **减少同步开销**：MLX Swift 不支持 GPU 侧控制流，CPU 往返难以完全消除
- **`interleavedGenerate`**：`MiniCPMTTSSemantic` 里已有这套流式基础设施但从未启用，
  值得先读一遍设计意图再决定

**建议**：除非 wall_clock 需要压进 500ms 以内，否则暂缓。

### 4. 变长音频块 —— ❌ 已证否：不可行，建议关闭

**核心证据（仓库内已有实测记录，见 `MiniCPMTTSSemantic.swift:1366` 注释）**：
让 TTS 语义解码器提前结束（`minNewTokens = 0`）曾被尝试过 ——
首块因此只生成 **5–13 个 code 而不是 25 个**，结果**整 1.00 秒块渲染成近静音**
（实测 RMS < 0.01，样本 "天气非" / "好的，" / "嗯，那"），
**文本完全没有被念出来**。这就是"变长块"最刺眼的反例：它不产生"更短的音频"，
而是产生"没有语音的音频"。

**原因**：flow 解码器的流式窗口依赖官方 26/25 look-ahead 契约（含 160ms 交叠 +
3 个 look-ahead code）；码数不足时窗口无法产出有效语音。这解释了为什么官方也把
中间块固定为 26 codes。

**官方行为复核**：Comni 会话里 24 个 AI 音频块，短块（<0.8s）只有 3 个，
**全部是 turn 的收尾块**，中间块一律 1.00s。即官方中间块同样是固定长度。

**我们与官方的唯一差异**：turn 开头的短块会被我们**前插静音填充到 1.00s**，
官方不填充。已实现 `MINICPM_DUPLEX_VARIABLE_BLOCKS=1` 跳过该填充，
但实测 **17/17 块仍为 1.00s**（首块在实践中总是填满窗口）→ 净收益 ≈ 0，默认关闭。

**结论：块内静音是模型的韵律输出，不是我们插入的填充。**
把静音裁掉只是把它从"块内"挪到"块间"，听感等价；而要真正缩短静音，
只能让模型的语音覆盖整个 unit —— 这一点已被两次独立实验否定
（见下文"已被数据否定的假设"）。

**唯一理论上能闭合停顿的路径**（不推荐，记录备查）：客户端把助手音频
**略快于 1× 播放**（约 1.2× 时域压缩，现代语音压缩算法基本无感），
使 0.65s 的语音覆盖 1.0s 的槽位。代价：需要引入时域压缩 DSP（当前 Swift 栈里没有），
且官方不做，属于产品级决策而非移植修复。

**本项原路线图的错误更正**：曾写"短 unit 提前停止解码可省下真实计算时间"——
不成立：中间块的 26 codes 是官方契约；而早停会产生静音而非短音频。

---

## 第二轮方案（含调研与实测，2026-09-21）

### 0. 先结掉一桩悬案：首音频延迟已达标 ✅

P7 交接文档记录的首音频延迟是 **6.1–6.4s（最差 12.89s）**。本轮优化后重测
（文本 turn 场景，纯响应延迟）：

| 会话 | 首音频 | 首文本 |
|---|---|---|
| sess_29081E37DB16 | **0.55s** | 6.21s |
| sess_5AA2079F31CD | 0.73s | 2.12s |
| sess_870BEF97D7DD | 0.81s | 2.26s |
| sess_CAFC7D2AE996 | 0.85s | 2.29s |
| sess_ABD075833B50 | 1.01s | 7.46s |

**典型 0.55–1.0s，比 P7 记录快约 7 倍**（浏览器会话的 2.55s 含用户说话时间，不是纯响应）。
人类对话轮换间隔约 0.5–1s，**P7 的阈值争议可以按"已达标"结案**。

### 1. TTS 采样循环上 `MLX.compile` —— ⚠️ 路径受阻，改为交付 asyncEval 版本

**原始调研**（保留备查）：MLX 的 `MLX.compile` 把多算子融合成单个 Metal kernel，
消除逐算子启动开销，对"循环内反复调用的同形状函数"收益最大（WWDC 2025 + MLX 文档）。
本仓库先例：`SourceSeparation/OpenUnmixModel.swift:261`。

**受阻原因（有代码证据）**：`MiniCPMTTSSemantic.generateChunk` 的 26 步循环里，
每步的 `decoder(current, caches: session.layerCaches, offset:)` **会就地改写 KV cache**
（非纯函数），而 `MLX.compile` 要求被编译函数无副作用。要套上 compile 必须把
decoder / layer / session 全部改成"传参进、返回新 cache"的函数式风格 ——
那会动到整条已通过数值验证的 TTS 路径，**风险与收益不成比例**。

**改用的落点：flow ODE 循环里的逐步阻塞同步**（`DiTFlow.swift:433`）。
原代码在 `odeSteps` 每次迭代末尾 `eval(value)`，把 8 步严格串行化：
主机必须等第 N 步完成才提交第 N+1 步，**启动延迟与同步成本被付了 8 次**
（生成每个音频块都要付一遍）。改为 `asyncEval(value)` —— 仍然物化结果、
仍然限制图规模与峰值内存，但**不阻塞**，CPU 可以先行、GPU 持续有活干。
这正是 `HiggsTTSModel` / `MossMLXRuntime` 在本仓库已采用的模式。

**实测（3 次重复，每次 19 个发声 unit）**：

| 指标 | 改前 | 改后（3 次） |
|---|---|---|
| `flow_ms` | 131.3 | **125.2 / 125.1 / 125.0**（±0.2ms） |
| `token2wav_ms` | 166.4 | 160.9 / 160.9 / 160.2 |
| `wall_clock_ms` | 646.2（单次） | 700.4 / 688.9 / 694.4 |

**诚实结论**：对**本阶段**的收益是确定且可复现的（flow **-6ms**、token2wav **-6ms**，
方差极小）；但 `wall_clock` 在 646–700ms 之间波动（`llm_decode` 与 `semantic_tts`
在多次运行里的自然方差就有 ±30ms），**因此不能声称整体提速** —— 那 48ms 的差异
主要来自其他阶段的运行间波动，不是本次改动。

**改动价值**：语义等价、零风险、可复现的 -6ms，保留。

### 2. TTS / token2wav 从 F16 量化到 8-bit —— 已实测误差，**建议搁置**

**实测（`tools/quantization_error_probe.py`，mlx 离线量化后反量化，8-bit / group=64）**：

| 组件 | 2-D 权重 | 权重相对误差 | 线性层输出误差 | 量化后 |
|---|---|---|---|---|
| tts | 639 MB | 平均 0.542% / 最大 0.649% | 平均 0.543% / 最大 0.645% | 170 MB |
| token2wav | 1201 MB | 平均 0.550% / 最大 0.769% | 平均 0.557% / 最大 1.019% | 261 MB |

**收益重新估算（按带宽而非体积）**：
- TTS：26 步 × (639→170 MB) ≈ 12.2 GB 少读 → M2 Max ~300 GB/s 实用带宽 → **~40ms**
- flow：8 步 × (flow 权重 ~450→120 MB) ≈ 2.6 GB → **~10ms**
- HiFT：单次 × (83→22 MB) → **~1ms**
- **合计 ≈ 50ms**（wall_clock 646 → ~600ms，7.7%），**远低于最初估的 120ms**

**风险（项目内已有前车之鉴）**：`Qwen3TTS/Configuration.swift:276-282` 明确记录
*"int4 decommissioned for TTS"*、*"the 8-bit bundle was retired
(synthesis quantization audibly degrades speech)"* —— 本项目已经因为可听劣化撤下过 TTS 的 8-bit 变体。

**成本**：Swift 侧无量化支持（`MiniCPMTTSSemantic` / `MiniCPMToken2Wav` 全是裸 `Linear`），
需要改 3 个模块的模型构造 + 权重加载 + 配置块 —— 属于"容易静默出错"的改动面。

**结论：50ms 收益 / 中高成本 / 真实音质风险 → 建议搁置**，除非未来需要最后那几十毫秒。
（注意：本项目 LLM 4-bit 也是因同类风险被用户否决的，标准应保持一致。）

### 3. `asyncEval` 让 CPU 侧工作与 GPU 重叠 ⭐

**调研**：MLX Swift 有 `asyncEval()` —— 提交后立即返回，调用线程可以继续。
**本仓库其他模型已在用**（`HiggsTTSModel.swift:195`、`MossMLXRuntime.swift:556`），
但 MiniCPM 热路径全部用阻塞式 `eval()`。

**适用点**：引擎在阶段边界（尤其 `STAGE_TIMING` 下）做阻塞同步；之后 CPU 还要做
base64 编码、JSON 组装、WebSocket 发送。改成 `asyncEval` 可让这些 CPU 工作与
GPU 计算重叠。
**预期**：数十 ms 量级（保守）。**成本**：低。**风险**：需小心与 `evalLock` 的交互。

### 4. 音频编码器上 ANE / CoreML —— ✅ 转换已跑通，实测 **10.5ms（比 MLX 快 8 倍）**

**实测结果（`scripts/export_minicpm_o_audio_coreml.py`，1 秒音频块 / 100 帧梅尔 → 50 个位置）**：

| 执行路径 | 延迟 | 与 torch 的余弦相似度 | max\|diff\| |
|---|---|---|---|
| torch CPU（参考） | 97–102 ms | — | — |
| CoreML `CPU_ONLY` | 22.8 ms | 0.9922（min 0.9736） | 4.83% of peak |
| **CoreML `CPU_AND_NE`（ANE）** | **10.5 ms** | **0.9983（min 0.9866）** | 2.82% of peak |
| 我们现在的 MLX/GPU 实现 | 83.5–91.6 ms | — | — |
| 官方 llama.cpp-omni（同一 GPU 阶段） | 42.2 ms | — | — |

**→ ANE 版本比我们现在的 MLX 实现快约 8 倍，比官方 llama.cpp 还快 4 倍。**
更值得注意的是：**连 CoreML 的纯 CPU 路径（22.8ms）都比我们的 MLX GPU 路径快 3.7 倍** ——
这说明我们的 MLX 编码器实现仍有很大优化空间（官方 GPU 42ms 已是最好的 GPU 对照）。

**过程中解决的四个环境/工具链障碍**（都已固化进脚本）：
1. `coremltools 6.3.0` 配 `protobuf 7.x` 无法导入 → **升级到 coremltools 9.0**
2. `torchvision::nms does not exist` → 复用本仓库 `minicpm_duplex/runtime.py` 的
   `ensure_torchvision_compat()` schema 垫片
3. `modeling_minicpmo.py` 的相对导入（且目录名 `MiniCPM-o-4_5` 不是合法包名）
   → 建符号链接临时包，避免加载整个 17GB checkpoint
4. `torch.jit.trace` 传 lambda 返回 `ScriptFunction`（coremltools 拒收）
   → 用 `nn.Module` 包装；并显式传 `source="pytorch"`

**权重加载完美**：`loaded 367 tensors; missing=0 unexpected=0` —— 官方 `apm.*`
张量与 `MiniCPMWhisperEncoder` 逐一对齐；输出形状 `(1, 50, 1024)` 与流水线
一直观测到的"每 unit 50 个位置"完全一致。

**产物**：`models/MiniCPM-o-4_5-audio-ane/MiniCPM-o-4_5-audio-encoder.mlpackage`（583 MB，fp16）

**✅ 数值验证通过与冷启动实测（`scripts/verify_minicpm_o_audio_coreml.py`）**：

真实语音验证（`tools/fixtures/understand_test.wav` 的梅尔，非随机噪声）：

| 指标 | 结果 |
|---|---|
| 余弦相似度 | **平均 0.999955，最低 0.999715** |
| 相对误差 | max\|diff\|/max\|ref\| = **0.45%** |
| cos < 0.99 的位置 | **0/50** |

**结论：在真实语音上，ANE 输出与 fp32 参考几乎无法区分。** 注意随机噪声输入下
cos 只有 0.9983 —— 说明先前用噪声测出的误差被放大了（分布外输入），**真实语音
才是有效判据**。

冷启动：

| 项 | 耗时 |
|---|---|
| 首次加载 mlpackage（含 ANE 编译） | **9.6 s** |
| 二次加载（同进程） | 9.6 s |
| 首次推理 | 27.0 ms |
| **稳态推理** | **10.4 ms**（20 次平均，与导出脚本的 10.5ms 一致） |

**冷启动 9.6s 可接受**：服务本身启动约 40s，加上后约 50s；且 Swift 侧编译产物会缓存
到应用容器（`~/Library/Caches`），真实部署大概率只在模型变更后付一次，而非常驻每次启动
（此结论在 Python 路径实测，Swift 侧需复核）。

**→ 该方案已具备落地条件**（数值验证通过 + 启动成本有界）。剩余工作：Swift 执行分支、
`audio_projection_layer` 投影接入、以及让 ANE 与 GPU 真正并行的流水线改造。

### 5. 客户端自适应播放速率（抖动缓冲）⭐⭐

**调研**：macOS 自带 **`AVAudioUnitTimePitch`**（`rate` 1/32→32、音高独立、`overlap` 3–32
可调，macOS 10.10+），**不需要第三方 DSP**。但它属于 AVFoundation，**浏览器客户端用不上**
（Web Audio 不提供时域压缩；`<audio>` 元素的 `preservesPitch` + `playbackRate` 可用，
但我们的客户端是 PCM 排播，不是媒体元素）。

**两条落法**：
- 服务端：用 `AVAudioUnitTimePitch` 离线模式对 PCM 做微压缩（1.0→1.05×）再下发
- 客户端：JS 里实现一个小 WSOLA（约 100 行），或改用媒体元素播放

**诚实评估**：它只能平滑**欠载（underrun）导致**的硬切，**不能消除模型自身的
韵律停顿**（见第 4 项变长块的结论）。而当前 wall_clock 已低于实时，欠载已基本消失
→ **收益存疑，优先级低**。

### 6. 多 unit 流水线（双缓冲）⭐⭐

让 unit N 的 TTS+vocoder 与 unit N+1 的音频编码+LLM 解码重叠。
GPU 仍会串行化，但依赖等待与 CPU 段可以重叠。
**预期**：中等（理论 wall_clock → max(前半段, 后半段) ≈ 400ms），**成本高**
（要重构引擎主循环），风险是打乱现有已验证的时序。**建议缓行。**

---

## 本轮代码审计：两处"疑似大鱼"已排除

| 审计对象 | 结论 |
|---|---|
| `MiniCPMLLM` 的注意力是否走了慢路径 | ✅ **已是融合路径**：单 token 解码（热路径）走 `SDPA.attendAndMerge`（MLXFast）；MPSGraph 的 `MiniCPMExactBF16SDPA` 只在多 token prefill / 上下文重建时使用 |
| `MiniCPMToken2Wav` / `MiniCPMTTSSemantic` 是否有手写标量 kernel | ✅ 干净（详见上文第 4 项） |

→ `llm_decode` 的 282ms 是**真实的 8-bit 权重带宽成本**，不是实现缺陷。

---

## 搁置项

### LLM 4-bit 量化（用户决定保持 8-bit）

数据备查：官方 Q4_K_M 的 LLM 解码 ≈9.5ms/token，我们 8-bit ≈34ms/token，
预期 `llm_decode` 282 → 100–150ms。转换脚本 `scripts/convert_minicpm_o_llm_to_mlx.py`
现成；折中方案是只压 FFN/MoE，attention 与 `lm_head` 保 8-bit。
重启此项时必须先跑 `tools/duplex_understand_test.py` 的质量 A/B。

---

## 已被数据否定的假设（勿重试）

| 假设 | 结果 |
|---|---|
| 关 `MINICPM_STAGE_TIMING` 可借懒调度提速 | 实测 715.9ms vs 646.2ms，**无收益** |
| `listen_prob_scale=0.5` 抑制 listen | A/B 否定：1.0 反而更长更连贯、listen 占比更低 |
| 抑制自然 `chunk_eos` 逼模型多说 | 无效（78 unit 逐字统计，文本量不变），且产出乱码 |
| base64 f32 传输（128KB/块）拖慢浏览器 | 每秒仅几毫秒解码量，不成立 |
| 客户端响应窗口掐输入导致断续 | `FileResponseDrain`/`LiveInputTimeline` 均不节流，实测输入节奏 1.03s 稳定 |
| 块内 ~35% 静音是移植缺陷 | 官方填充率 72% vs 我们 60–70%，**模型天性** |
| token2wav/TTS 藏着手写标量 kernel | 排查干净：无自定义 kernel/MPSGraph，生产路径全是 MLX 内建 |

---

## 工具

| 工具 | 用途 |
|---|---|
| `tools/duplex_ab_listen.py` | 直连 `/v1/realtime`，脚本化场景 + 分阶段耗时聚合，性能改动自验证 |
| `tools/duplex_understand_test.py` | 喂已知语音（`say` 合成），验证理解能力是否退化 |
| `tools/mps_fill_probe.py` | 直接驱动 Python 双工运行时测填充率（仅隔离编排层） |

**测量注意事项**：Comni 运行时会抢 GPU（`llm_decode` 曾被抬到 1082ms），
所有性能对比必须错开；听感相关结论需 Mac 真机验证。
