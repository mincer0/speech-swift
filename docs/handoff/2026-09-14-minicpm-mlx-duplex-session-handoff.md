# MiniCPM MLX 双工语音路线 — 会话交接文档

> 日期：2026-09-14（Asia/Shanghai）
> 工作树：`/Users/mincer/项目/s2s/runtime/speech-swift-minicpm-native`，分支
> `feature/minicpm-o-native`，基线 `fe5f7ba` + 本会话 7 个提交（见第 2 节）。
> 上游背景：`/Users/mincer/项目/s2s/HANDOFF_MINICPM_MLX.md`；前序修复细节见
> `docs/handoff/2026-09-12-minicpm-mlx-duplex-audio-fix.md`（第 1–15 节，本文档为其收束）。

## 1. 一页状态

| 项 | 状态 |
| --- | --- |
| 服务 | `127.0.0.1:7861`，`backend=swift_mlx`，`status=ready`（launchd label `com.mincer.minicpm-duplex`） |
| 对话主链路 | ✅ 正常：长故事 90s 完整输出、RTF 0.61、零到达空洞、插话可用 |
| 语音可懂度 | ⚠️ 部分修复：think 泄漏、纯协议 unit 垃圾音频、启动标记缺失已修；**unit 接缝处仍偶发单字重复/缺失**（用户听感"有些不像中文"） |
| 根因（已定位未落地） | 官方每个 unit 的 prefill 携带真实音频 embeddings；我们的静音 continuation 不喂音频 → 模型失去时间轴听觉上下文 → 接缝质量退化 |
| 修复阻塞点 | 恢复真实音频上下文需要音频编码器桶式补零（已验证 16ms/块，但偏离密封 parity 0.25 > 0.05）→ 需重建 golden 基线（见第 5 节决策） |
| 环境警告 | GPU 训练任务并行时推理 RTF 0.7 → 14（20 倍劣化），任何语音验收必须在无训练负载下进行 |

## 2. 本会话提交（全部在 feature/minicpm-o-native，未推送）

| commit | 内容 |
| --- | --- |
| `18f1262` | 断续主修复：continuation 驱动 + `emitOutputs` 提取 + 播放缓冲 |
| `1617818` | 静音包服务端 promotion + 播放缓冲上限（替换错误的包计数预算） |
| `9cb6368` | `<think>`/`</think>` 采样禁令 + TTS 条件纯文本化（后者见 `240e400` 部分回退） |
| `54a7748` | 前端稳定模式门控（用户说话期间禁止插话）+ lengthPenalty 默认 1.2 |
| `57fe8cd` | 音频编码器分阶段基准 + 重编译瓶颈调查 + 桶式补零尝试（已回退） |
| `240e400` | 紧急回归修复：恢复钉死的 TTS 条件收集 + vocoder 文本门控 |
| `f453195` | 官方源码逐 token 对照结论 + 后续路径 |

注意：`240e400` 回退了 `9cb6368` 中的"TTS 条件只收文本"部分（该改动剥掉了
`<|tts_bos|>` 启动标记导致短答全静音），保留 think 禁令；`index != 0` 收集
规则为钉死 parity 行为，勿再改。

## 3. 已修复的根因链（按用户报告顺序）

### 3.1 输出断断续续（v1/v2，已解决）

服务端"一个输入包 → 一个 unit"串行循环把输出速率锁死在客户端包节奏上
（实测 1s 音频 / 1.78s 到达、17 次播放缓冲清空）。修复三层：

1. `MiniCPMDemoBackend.continueResponse()`：response 打开且队列空时用全零
   continuation tick（~1ms，跳过音频编码器）自驱动下一 unit；
2. `Server.swift` worker 的 `driveContinuation`：队列非空立即让出（真实输入
   优先），epoch 变化立即停止；
3. 播放缓冲上限：`responseAudioSentSeconds - 距首块墙钟 ≥ 3s` 即节流
   （播放进度基准，barge-in 丢弃上限即缓冲量）。

验证：回答内部零到达空洞；浏览器 E2E（文件模式注入音频）Ahead 3392ms、
Shift/Drift 全 0。

### 3.2 静音包未走快路径（v2，已解决）

前端只在自己 modelState='speaking' 时给零包加 `continuation_tick`，大量
静音包按真实音频编码（350–440ms/包）→ 服务周期 ≥ 包间隔 → 队列永不清空
→ 驱动永不触发。修复：服务端 `append` 对全零双工音频直接 promotion（引擎
侧前置校验保持不变，listening 阶段 turn 已结束仍走真实编码）。

### 3.3 未知语音 / 不像中文（v3，部分解决）

- `<think>` 泄漏：Qwen3 推理标记在本词表非 special，双工采样无禁令 →
  已在引擎 init 解析并加入每步 forbidden ✓
- 纯协议 unit 垃圾音频：`[speak]`-only 条件跑满 25 码 → 修复为解码文本
  非空才跑 vocoder ✓
- `<|tts_bos|>` 启动标记被剥（v3 修复的回归，v7 修复）：短答全静音 →
  已恢复钉死收集规则 ✓

### 3.4 长输出中断（v4/v5，已解决）

lengthPenalty（仅作用于 turn_eos，不影响 listen/barge-in）1.1 → 1.2，
配合插话门控。注意：前端 spinbutton 默认 1.05 会覆盖服务端默认——
`audio_duplex.html` 的 `duplexLengthPenalty` 默认值需要同步改（**未完成**，
见第 6 节）。

## 4. 当前残留问题与根因（下一会话的主任务）

### 4.1 unit 接缝断字（用户听感"根本不像中文"的来源）

每个 unit 的 LLM 文本 1–9 字符，接缝处丢字/重复（"小刺猬猬"、
"阳光明[媚的]早晨"、"叫朵|它呀"）。TTS 念的就是这个文本。

根因链（已实证）：
1. 官方 `streaming_prefill`（pinned commit `ba7fa9c`，
   `MiniCPMO45/modeling_minicpmo_unified.py:4544`）对 AUDIO/OMNI 每个 unit
   **必须携带真实音频 embeddings**（`has_audio` 必需，无 continuation
   旁路）——官方上下文是连续音频时间线；
2. 我们的 continuation promotion 喂 `[<unit>]` 无音频 → 模型失去时间轴
   听觉上下文 → unit 开头采样 `<|listen|>`（被 coerce）、句中 yield、
   边界断字；
3. 协议层本身无 bug：官方 `streaming_generate` 循环与我们的
   generate 逐项对照一致（coerce、收集 `j != 0`、28 字符上限、终止符
   延迟合并、forbidden 列表）。

### 4.2 音频编码器每形状重编译（~400ms/块）

流式 KV 逐块变长 → 注意力张量形状每块不同 → 钉死的 MPSGraph BF16
注意力（`MiniCPMAudioBF16Attention`）每块重新编译 plan。基准
（`testStageTimingBenchmark`）：mel 6ms、同形状 15.7ms、新形状 400ms+
（native 变体 ~950ms）；另有每 call 懒建图地板 ~125ms。

桶式补零 + 加性掩码（-10000 有限值，exp(-10000-max) 精确下溢为 +0）已
实现并验证 ~16ms/块，但 24 层 BF16 递归把归约分段变化放大到 encoder
states max 0.25 / mean 0.0147（密封容差 0.05/0.01）→ 已按纪律回退，
golden gate 恢复 0 失败。

### 4.3 已否决/已排除的方向

- BF16 LLM：本机 RTF 3.1、首音 30s，不可用；且边界断字与精度无关。
- MPS 对照路线：WebSocket 对无 Origin 头探针返回 403（其自有准入策略），
  探针对照需加 Origin 或改用浏览器；`backend=mps` 启动 ~40s。
- "NaN/1e38 音频"：分析脚本伪影（blob 为 16-bit PCM 按 float32 误读），
  音频本身是真实语音电平。**任何波形分析必须先读 `getsampwidth()`。**

## 5. 下一步路线（决策点，按优先级）

**目标：恢复官方的连续音频时间线，修掉接缝断字。**

1. **重建 golden 基线并接受实现级偏差**（推荐首选）：
   - 上游 demo 已重新克隆在 `/tmp/MiniCPM-o-Demo`（pinned `ba7fa9c`，会话
     结束后可能被系统清理，需重克隆）；
   - 官方 oracle 脚本在 `MiniCPMO45/modeling_minicpmo_unified.py` +
     `core/processors/`；用 `scripts/export_minicpm_o_audio_golden.py`
     流程重建 fixture（桶式补零开启状态）；
   - 重密封后接缝断字预期大幅改善（模型获得连续音频时间线）。
2. **前端 Length Penalty 默认值同步**：`audio_duplex.html`
   `duplexLengthPenalty` 与 `audio-duplex-app.js:1077` 的 fallback
   `1.05 → 1.2`（当前前端值会覆盖服务端默认，使抑制失效）。
3. **混合音频上下文**（若不接受第 1 步的偏差重密封）：每 N 块插入一次
   真实静音编码（成本 +400ms/N），其余走 continuation——上下文部分恢复。
4. **编码器每形状重编译**：MPSGraph 路径的掩码占位符受 BF16 加法精度
   提升限制（placeholder add 会升 FP32，需 FP32 相加后回投 BF16——已验证
   可行但数值仍偏离）；MLX 侧 `compile(shapeless:)` 需逐层验证。此为
   性能项，非正确性阻塞项。
5. **稳健性**：denormal 级输入样本经 mel log 产生 NaN 并沿编码器→LLM→
   TTS 传播（诊断实测）；建议 mel 能量下限 clamp（小改动）。

## 6. 验证工具与命令（均已验证可用）

```bash
# 服务（launchd；停止用 ./minicpm_duplex_service.sh stop）
cd /Users/mincer/项目/s2s
MINICPM_DUPLEX_BACKEND=mlx MINICPM_DUPLEX_PORT=7861 ./minicpm_duplex_service.sh start

# 故事探针（文本质量/RTF/完成度）
cd runtime/speech-swift-minicpm-native
.venv-minicpmo 路径: /Users/mincer/项目/s2s/.venv-minicpmo/bin/python
scripts/probe_minicpm_native_duplex.py --url "ws://127.0.0.1:7861/v1/realtime?mode=audio" \
  --question "请给我讲一个至少三百字的长故事…" --turn-timeout 150

# 音频编码器分阶段基准（需模型路径 env，golden 测试需 GOLDEN_DIR）
MINICPM_AUDIO_MLX_PATH=/Users/mincer/项目/s2s/models/MiniCPM-o-4_5-native-mlx-v1/MiniCPM-o-4_5-audio-mlx \
MINICPM_AUDIO_GOLDEN_DIR=/Users/mincer/项目/s2s/test-output/minicpm-audio-golden-v2 \
  swift test --filter "MiniCPMAudioTests" --disable-sandbox

# 断续分析方法（无现成脚本，方法如下）：按 stream.jsonl 的 audio delta
# 时间戳 + blob WAV 时长逐 turn 模拟播放（播放时钟 1x，缓冲跨 turn 归零），
# stall = max(0, 到达间隔 - 前块时长)。**必须读 getsampwidth()**（blob 为
# 16-bit PCM，按 float32 误读会产生假 NaN）。
```

## 7. 环境与安全规则（延续主 handoff）

- 训练任务（如 `training.rhythmnet_v24_tempo_robust`）并行时推理劣化
  20 倍，语音验收必须在无训练负载下进行；不要未经确认停止用户训练进程。
- 不执行 `git reset --hard` / 清理未跟踪文件；工作树大量 WIP 未提交，
  以 `git status --short` 为准。
- 不推送远端；重型 build/test 串行执行。
- `/tmp/MiniCPM-o-Demo` 与 `/tmp/*.wav/b64` 是临时产物，会话后可能消失。
- 服务日志：`.runtime/minicpm-duplex/server.{stdout,stderr}.log`；会话数据：
  `data-native/sessions/<id>/stream.jsonl` + `blob/*.wav`（16-bit PCM）。
