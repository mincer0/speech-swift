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

### 4. 变长音频块 —— ✅ 已实测：几乎不触发，建议关闭

**结论先行**：本项已被实测否定。官方短块（0.32–0.76s）**全部出现在 turn 边界**
（Comni 会话里 3 个短块全是 turn 的收尾块），**中间块同样是固定 26 codes / 1.00s**
—— 这是官方 26/25 look-ahead 契约的一部分，不是我们的差异。

我们与官方的实际差异只有一处：turn 开头的短块（EOS 立即放行的首块）会被我们
**前插静音填充到 1.00s**，官方不填充。已实现
`MINICPM_DUPLEX_VARIABLE_BLOCKS=1` 跳过该填充，但实测 **17/17 块仍是 1.00s** ——
因为首块在实践中几乎总是填满整个窗口。

**净收益 ≈ 0，默认关闭。** 若未来发现某类会话频繁产生短首块，再打开验证。

**原路线图的错误更正**：本项曾写"短 unit 提前停止解码可省下真实计算时间"——
不成立，中间块的 26 codes 是官方契约，省不了。

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
