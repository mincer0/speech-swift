# MiniCPM MLX 双工路线语音输出断续修复 — 交接文档

> 日期：2026-09-12（Asia/Shanghai）
> 工作树：`/Users/mincer/项目/s2s/runtime/speech-swift-minicpm-native`（分支 `feature/minicpm-o-native`）
> 范围：Swift 原生 MLX 双工服务（`MINICPM_DUPLEX_BACKEND=mlx` 路线）的音频输出连续性、复读防护、探针窗口归属与构建环境
> 上游背景：`HANDOFF_MINICPM_MLX.md`（主集成目录）第 15/16 节

## 1. 问题与定量证据

用户报告：MLX 路线输出语音断断续续。

对真实会话录音流（`data-native/sessions/<id>/stream.jsonl`）做播放模拟分析
（脚本思路见第 5 节）：修复前每个约 1.0 秒的音频块平均 1.78 秒才到达
（中位数），单次回答内部存在 0.8–5.8 秒空窗，5 轮对话共 17 次播放缓冲
清空（underrun）——这就是"断断续续"。

## 2. 根因

传输 worker 是"一个 `input.append` → prefill → 一个 generate unit（约 1 秒
音频）"的串行循环：**输出节奏被客户端发包节奏锁死**。

- 生成本身并不慢：阶段计时显示每个 unit 的 LLM 解码 0.16–0.43s、语义 TTS
  0.03–0.37s、Token2Wav 0.17–0.39s，合计 RTF 0.5–0.9（快于实时）。
- 但客户端每 1.0 秒才送来一个 1 秒包（浏览器），探针更慢（等 ack 后再等满
  1 秒 ≈ 1.6–1.7 秒/包）。生成跑不满，输出速率 ≈ min(生成速率, 包到达率)。
- 浏览器 1.0 秒喂包恰好贴着实时零余量：任何单 unit 尖峰（首块 forceFlush、
  semantic TTS / Token2Wav 尖峰、首次 Metal 编译）都会让播放器断流。
- 浏览器 `AudioPlayer.endTurn()` 不清空已排程音频（自然句尾依赖它），所以
  不能靠"无限跑 ahead + 打断时靠客户端丢弃"解决，领先距离必须有界。

## 3. 修复内容（按文件）

### 3.1 `Sources/MiniCPMDemoBackend/Backend.swift`

- 把 `append()` 内联的"engine 输出 → wire 事件"转换提取为
  `emitOutputs(_:inputID:requestedResponseID:wallClockStart:)`（throwing），
  `append` 与新的 `continueResponse` 共用；response 打开/关闭/新开时同步
  维护 `activeResponseID`、`pendingFinalize` 与领先计数。
- 新增 `continueResponse() -> [MiniCPMDemoEvent]?`：当 `state == .active`、
  双工模式、`activeResponseID != nil`、`!pendingFinalize` 且领先预算未用完
  时，用全零 16k 采样 continuation tick（`continuation_tick=true`、
  `speech_active=false`，引擎跳过音频编码器，prefill 约 1ms）自驱动生成
  下一个 unit；input_id 使用 `cont_%08llu` 前缀。引擎拒绝/turn 结束/预算
  用尽返回 nil。
- 新增 `continuationLeadUnits` 计数与 `maxContinuationLeadUnits = 3`：
  合成 unit 计入，**真实 input.append 回补 1**（`append` 入口处
  `max(0, count-1)`）。该值即 barge-in 时可能需要丢弃的陈旧音频上限
  （≤3 秒），也是吸收生成尖峰的播放缓冲。
- 所有 `activeResponseID = nil` 的生命周期点同步清零领先计数
  （cancel/reset/close/error/listen 关闭/新 response 开启）。
- 新增私有 `encodeFloat32(_:)` 用于合成 tick 的 PCM 载荷。

### 3.2 `Sources/MiniCPMDemoBackend/Server.swift`

生成 worker 增加 `driveContinuation(for:)`，在每个真实输入处理完
（emit + commit）之后调用：

- 循环条件：`generationEpoch.isCurrent(queued.epoch)`——barge-in/reset
  立即退出；
- **队列非空立即 return 回主 FIFO 循环**：真实输入永远优先，直播用户
  音频按序 prefill，不积累落后；
- 每 unit 走与主路径一致的 emit（epoch 逐事件校验，陈旧事件在
  `emitMiniCPMDemoEvents` 处丢弃）→ `commitPendingResponse()`；
- 错误事件写线后抛出，行为与主路径一致（关闭 FIFO，由下一输入触发
  transport teardown）。

### 3.3 `Sources/MiniCPMDuplexRuntime/MiniCPMNativeDuplexEngine.swift`

新增 `MiniCPMResponseRepeatGuard`（8-bit 解码器的退化复读循环防护）：

- 检测规则移植自参考 Python 双工协调器（`minicpm_duplex/server.py` 的
  `_repeated_phrase_suffix`）：归一化（字母/数字，含 CJK）后，≤12 字短语
  连续重复 4 次（单字符循环如"哈哈"排除），并扩展**句子级规则**：13–24
  字短语连续重复 3 次（one-second unit 会把循环切在任意位置，长句循环
  只有 4 次阈值会漏）。
- 另加 120 unit 回答上限（对齐上游 streaming answer 安全上限）。
- 触发后的行为：在 `generate()` 入口处消费 trip，复用既有
  `forceListenOverride` 路径——`closeTurnForForcedListen`（KV 喂 turn_eos）
  + 强制采样 `<|listen|>` + 释放 TTS/Token2Wav turn 状态，response 正常
  关闭，协议边界干净。
- 接线点：`generate()` 末尾对 spoken unit 追加文本并检查；`isListen`、
  `endOfTurn`、`prepare`、`reset`、`interrupt`（经
  `repairInterruptedTurn`）、`rollback`（被丢弃的 unit 文本不得计入）
  处 reset。
- 存量文本仅保留尾部 600 字符（检测窗口 + 余量）。

### 3.4 测试（新增 7 个）

- `Tests/MiniCPMDemoBackendTests/MiniCPMDemoBackendTests.swift`：
  - `testContinueResponseStreamsNextUnitFromSyntheticContinuationTick`
    （合成 tick 的输入形态、同 response 归属、cont_ 前缀、关闭后拒驱）；
  - `testContinueResponseStopsAtLeadBudgetAndRefillsOnRealInput`
    （预算 3 + 真实输入回补 1）；
  - `testContinueResponseIsRefusedWithoutAnOpenResponse`；
  - 新增 `ScriptedDuplexEngine` 桩。
- `Tests/MiniCPMDuplexRuntimeTests/MiniCPMDuplexRuntimeTests.swift`：
  短语 4 次阈值、句子级 3 次阈值、单字符循环/短文本豁免、120 unit 上限。

### 3.5 `scripts/probe_minicpm_native_duplex.py`

全双工修复后，旧 response 的 `turn_end` 可能落进下一个问题窗口（模型
听到新问题才收口旧回答），回答尾部与 `cont_*` 单元也会在窗口之间到达。
探针窗口归属修正：

- end 边界到达时若**当前窗口尚无任何模型输出**（text/audio 都没有），
  视为前一 response 的收口，不完成当前窗口，继续等本窗口自己的回答；
- 当前无窗口时到达的 text/audio：归属最近完成的 turn（续说/尾部），不再
  误报 unsolicited；启动期（尚无任何 turn）的 unsolicited 与复读检测
  全部保留；
- `last_completed_turn` 跟踪，response.done / listen_end / 超时完成路径
  均登记。

## 4. 设计权衡记录

- **领先预算为什么是 3 而不是无限/120**：浏览器 `endTurn()` 不清空已排程
  音频；无限跑 ahead 时打断要冲掉的陈旧音频无界（长回答可累积十几秒）。
  3 秒足够吸收实测尖峰（首块 forceFlush ≈ 1.2s、TTS/T2W 尖峰 ≤0.4s），
  同时把打断排空压到 ≤3 秒。真实输入按 1/s 回补，稳态缓冲 ≈ 3 秒。
- **探针 ack 语义不受影响**：driver 单元的 delta 带 `cont_*` id，不会误
  ack 探针的真实输入；unsolicited 判定只在无 turn 上下文时触发。
- **continuation tick 与既有语义一致**：探针在回答期本来就发
  `continuation_tick` 静音包（prefill_ms≈1.1），P1–P4 全部在这些语义上
  通过；驱动只是把同样的输入在队列空缺时自动补上。

## 5. 回归证据（2026-09-12）

- normal 探针 5/5，`unsolicited=0`、`repeated=0`；回答内部空窗
  **0.000s**（修复前 17 次清空、最差 5.8s）：
  - `sess_A7448859DDF6`（lead-budget 构建）18 块音频 5 个回答；
  - 最终烟雾 `sess_…（/tmp/mlx-final-check-*）` 5/5。
- story 探针 3/3（复读 guard 生效后；修复前一次运行复现 48 包超时、
  无边界、复读循环）。
- barge-in 探针 3/3：第一轮"好的，我"后打断，第二轮"7加8等于15。"、
  第三轮"9减4等于5。"，无旧输出越过 cancel。
- Swift：`MiniCPMDuplexRuntimeTests` 33/33、`MiniCPMDemoBackendTests`
  53/53、`MiniCPMDemoServiceTests` 5/5。
- 前端 static 15/15；`scripts/tests` 审计 62/62；主集成目录 `tests/`
  122 通过（6 个既有 skip）。
- 播放模拟分析方法：按 `stream.jsonl` 的 audio delta 到达时间戳与 blob
  WAV 时长，逐 turn 计算 `max(gap − 前块时长)` 作为回答内部 stall。

## 6. 构建环境修复与注意

- 默认 `.build` release scratch 曾报 `missing required module
  '_NumericsShims'`：Xcode 升级后残留过期 clang module cache。删除
  `.build/arm64-apple-macosx/release` 后全新构建即恢复（本次已在默认
  scratch 完成干净重建；临时 scratch `.build-rel-fix` 亦可）。
- **注意 `.build/release` 是指向 `.build/arm64-apple-macosx/release` 的
  符号链接**：清理该目录会把 `scripts/build_mlx_metallib.sh` 放置的
  `mlx.metallib` 一并删除，运行时报 `MLX error: Failed to load the
  default metallib`。重建方法：
  `./scripts/build_mlx_metallib.sh release`（产物落在二进制旁边，MLX
  按 binary 同目录 → SwiftPM bundle → METAL_PATH 顺序查找）。
- 部署：release 二进制位于 `.build/release/minicpm-mlx-server`（launcher
  默认路径，native link 自检通过，无 Python/PyTorch 链接）。

## 7. 当前运行状态

- 服务：`127.0.0.1:7861`，`backend=swift_mlx`，`status=ready`，经
  `MINICPM_DUPLEX_BACKEND=mlx MINICPM_DUPLEX_PORT=7861
  ./minicpm_duplex_service.sh start` 启动（launchd label
  `com.mincer.minicpm-duplex`）。`7860` 为无关 LTX UI（本轮已不在监听）。
- 日志：`.runtime/minicpm-duplex/server.{stdout,stderr}.log`。

## 8. 已知边界与遗留

- 首音延迟（6–13s）与 8-bit 量化导致的文字含混为既有问题，本次未改。
- 回答最开头仍可能出现一次 ≤0.13 秒的间隙（首块尖峰发生在缓冲建立前），
  播放器启动延迟（~200ms 起）可掩盖。
- turn 边界（提问期）的等待时间由输入节奏决定，不属于本次断续问题。
- 浏览器端 `endTurn` 保留 ≤1 秒句尾缓冲属设计行为；模型主动收口时客户端
  可能多播 ≤3 秒（领先预算上限）——如需更急的打断可后续在客户端
  interrupt 路径加 `audioPlayer.stopAll()`。
- P5 真实麦克风/扬声器 AEC（E4）、P7 产品阈值、P8 全量 runner 仍未验收，
  见主 handoff。

## 9. 安全规则（延续主 handoff）

- 本提交仅包含本次会话改动与本文档；工作树中其余未跟踪 WIP 保持原状。
- 不执行 `git reset --hard` / `git checkout --` / 清理未跟踪文件；
  不 push 到远端，除非用户明确要求。
- 重型 build/test 与模型服务严格串行；不要仅凭历史 PID 停进程。

## 10. 追加修复（2026-09-13）：浏览器真实会话仍断续的第二轮根因

第一轮修复后探针全绿，但用户真实浏览器测试仍断续（会话
`sess_D4B8F11F1B4A`）。取证发现两个探针路径掩盖不了的问题：

1. **浏览器静音包不走 continuation**。前端只在自身
   `modelState === 'speaking'` 启发式成立时给零包加 `continuation_tick`
   （`audio-duplex-app.js` 的 `zeroFilled && modelState === 'speaking'`），
   turn 边界之后、以及任何启发式失配的窗口里，静音包按真实零音频处理，
   每包付出 350–440ms 的完整音频编码器 prefill → 服务端周期 ≥1.0s，
   永远追不上 1.0s 的包节奏 → 队列常驻非空。
2. **驱动循环的"队列非空即让出"门禁因此从未触发**（整个会话 0 个
   `cont_*` 单元），修复形同虚设；且"领先预算按合成/真实包计数"的机制
   本身约束不住客户端缓冲增长（每包回补 1 个预算时，2 unit/包 的产出率
   会让缓冲无界增长）。用模拟浏览器 wire 行为的客户端完整复现。

第二轮修复：

- **服务端零包自动 promotion**（`MiniCPMDemoBackend.append`）：双工模式下
  解码后的输入若为纯零 Float32 音频（无 text/frames），由服务端直接注入
  `continuation_tick`，完全不信任客户端 flag；引擎侧全部前置校验
  （turn 打开、全零、无 frames/text）保持不变，listening 阶段
  （turn 已结束）的静音仍走真实编码路径以维持模型的时间感。
  `normalizeObject` 会把 base64 PCM 解码成数字数组，检测两种表示都支持。
- **播放缓冲上限取代包计数预算**（`maxBufferedAudioSeconds = 3`）：
  每个 open response 记录首块音频的发出时刻与已发送音频秒数；
  `已发送 − 距首块墙钟时间 ≥ 3s` 即停止驱动。浏览器以 1x 实时播放，
  该差值就是客户端未播放缓冲——barge-in 丢弃上限与尖峰吸收缓冲二合一，
  且无论 unit 由真实包还是驱动产生都成立。另保留
  `maxConsecutiveSyntheticUnits = 4` 连发上限，覆盖不产音频的纯文本
  response，防止缓冲上限永不触发。
- worker 的"队列非空让出"保留：真实包优先被模型听见，压低 barge-in
  决策延迟；静音包 promotion 后服务端周期 ~0.73–0.79s < 1.0s，队列会
  排空，驱动在空窗内自动补位。

回归（2026-09-13）：`MiniCPMDemoBackendTests` 55/55（新增
`testContinueResponseStopsAtPlaybackCushionBound`、
`testContinueResponseStopsAtSyntheticStreakCapAndRealInputResumes`、
`testSilentDuplexPacketsArePromotedToContinuationTicks`）；
`MiniCPMDuplexRuntimeTests` 33/33。模拟浏览器客户端（1.0s 包节奏、
浏览器 continuation 启发式、不等 ack）：回答阶段每 1.0s 音频块
0.73–0.79s 到达、缓冲填充至 ~3s 后正确节流；探针 normal 5/5（回答内部
stall 0.000s）、barge-in 3/3。

剩余已知断续来源（非本轮范围）：每次回答/分句首块的听→说转换开销
（~1.3–3.4s，对应已知首音延迟问题）；用户持续说话期间真实语音 prefill
（~0.4s/包）造成的轻微赤字（≤0.3s/块，播放器启动缓冲可部分掩盖）。

## 11. 追加修复（2026-09-13）：语音怪异与长输出中断的第三轮根因

断续修复后用户实测：语音含未知语音、中文不准、长输出无法完成。取证
（长故事探针 + `MINICPM_DUPLEX_LOGIT_TRACE_TOPK` 逐步采样 trace + 会话
audio/text delta 对账）定位两个确定性问题：

1. **`<think>` 泄漏**：Qwen3 骨架的推理标记在本词表里不是 special token，
   双工采样没有禁令；用户会话文本里出现字面 `<think>`，TTS 直接念出垃圾
   （"未知语音"）。
2. **协议 token 混入 TTS 条件**：旧收集规则是 `index != 0`——假设 unit 的
   第 0 步一定是 `<|speak|>`/`<|tts_bos|>` 引导 token。trace 显示模型在
   unit 开头经常连续采样多个 `<|listen|>`（被 coerce 成 tts_bos）或
   `[listen, speak]`，于是协议 token 落在第 ≥1 步、被原样收进
   `buildCondition`。只含协议 token 的 unit（generated 非空但无文本）会用
   退化条件跑满 25 个语音码 → 一整秒无法辨认的语音（用户会话
   `client_in_00000005` 即此类）；夹在文本中的协议 token embedding 则污染
   相邻发音。另有一条独立确认：unit 第 0 步采样出纯文本时（约 2/40），
   index!=0 规则会把它丢掉——这正是"叫朵|它呀""这|美丽的森林"类开头丢字。

BF16 对照实验：模型根目录换挂 `MiniCPM-o-4_5-llm-mlx-bf16`（parity 已
验证的 bundle），RTF 3.1、首音 30s——本机不可用，且文本乱在 BF16 下同样
出现 → 乱字与量化无关，是跨单元上下文构造 + 协议 token 处理问题。

修复：

- `MiniCPMNativeDuplexEngine`：init 时解析 `<think>`/`</think>` 并加入每步
  forbidden 列表（`reasoningForbiddenTokenIDs`）。
- `MiniCPMNativeDuplexProtocol.consume`：收集规则改为 **只收集非协议
  token**（listen/speak/ttsBOS/turnEOS/chunkEOS/chunkTTSEOS/eos/unitEnd
  一律不进 TTS 条件与文本序列），不再依赖 step 下标启发式。协议 token
  仍照常进 KV（shouldFeed 行为不变，KV/logit parity 不受影响）；
  纯协议 unit 现在 generated 为空 → 不跑 TTS、不发垃圾音频。

回归：Swift 88/88（更新 2 个固定旧行为的协议 trace 测试）、前端 15/15；
长故事探针：**81.28s 完整长故事**（修复前 5–15s 即断）、RTF 0.663、
think 泄漏 0；normal 5/5、barge-in 3/3、回答内播放模拟最差 stall ≤1.1s
（模型在句子间主动让话的 turn 边界，属全双工自然行为）。

遗留边界：8-bit 模型在 unit 边界偶发单字重复/缺失（"小刺猬猬"、
"阳光明[媚的]早晨"）——这是量化模型跨单元采样的质量天花板；BF16 在本机
RTF 3.1 不可用，group-32 重量化需要重做 golden parity（上游 demo 检出
`/tmp/MiniCPM-o-Demo` 已被系统清理），留待后续决策。

## 12. 追加修复（2026-09-13 晚）：句中 yield 碎片化的门控与惩罚

用户再次实测后取证（sess_C7154416E736），根因链完整闭合：

模型在用户说话期间频繁句中 yield（turn_eos）→ 语义 TTS turn 重置
（token2wav_decoder_positions 回到 302）→ 续句片段成为新 TTS turn 的
悬空开头 → `minNewTokens=0` 允许其立即采样 EOS → 0 语音码 → 发出
"前导静音 + 尾端残音"的 padding chunk（该 chunk 的
`audio_padding_samples` 缺失是因为 endOfTurn unit 不加 transport
padding，前导静音来自 vocoder 自身）→ 用户听到 1 秒级空洞与残缺词。
波形取证：文本 unit 的音频含 96% 静音（RMS 0.0007–0.0017）。

修复（沿用 Python 稳定模式的既定产品语义）：

- **前端稳定模式门控**（live-input-timeline.js）：用户**正在说话**时
  `responseWindowActive=false`（首个 listen/speak 决策走 greedy → 模型
  确定性听写，不再插话碎片）；采样窗口只在**用户停顿后**的 3-chunk
  窗口内打开。此前窗口在语音期间保持打开是模型不断插话的直接原因。
  相关前端静态测试已更新为新语义。
- **服务端 lengthPenalty 默认 1.1 → 1.2**（仅 turn_eos 的 logit 惩罚，
  不影响 listen/barge-in 与 chunk 边界）：进一步抑制句中 yield。
  注意：1.45 的校准实验作废——训练进程（rhythmnet_v24_tempo_robust，
  20:03 启动）与推理抢 GPU，实测 RTF 10–14（正常 0.7–0.8），该轮数据
  全部受污染；1.2 未在干净环境下重新校准。

诊断性发现（不修，记录）：音频编码器对 denormal 级输入（≤1e-38）会产生
NaN 并沿 LLM/TTS/vocoder 链路传播（诊断探针的注入字节模式触发）；
浏览器零填充静音走 continuation promotion 不经过编码器，真实麦克风
room tone（~1e-4）是否触发待验证。建议后续在 mel 能量下限处做 clamp。

遗留：GPU 与训练任务共享时，实时性必然劣化 15–20 倍——这不是服务端
可修复的问题；验收测试必须在无训练负载下进行。

## 13. 音频编码器性能专项（2026-09-13 晚）：定位、修复尝试与结论

用户要求排查 MPS→MLX 移植的编码器 bug 与优化空间。分阶段基准
（`E2EMiniCPMAudioTests/testStageTimingBenchmark`，env-gated）结论：

- mel 提取（CPU vDSP 双精度稠密 DFT）：**6ms**/秒音频——非瓶颈。
- `acceptAudio` 流式路径、**同形状重复调用**：**15.7ms**——非瓶颈。
- 长 streaming 会话第 2 块起：**400–430ms/块**（MPSGraph 变体）/ ~950ms
  （native 变体），与 KV 增长量无关、恒定——**每个新 KV 张量形状触发
  一次 MPSGraph plan 重编译/MLX 内核重特化**。流式 cache 逐块变长，形状
  永不重复，故每块都付一次编译税。
- 每 call 固定地板 ~125ms 为 Swift 侧懒图构建（帧数=16 的 forward
  graph-build-only ≈ 124ms，GPU 实算仅 ~2.6ms）。
- 机器噪声警告：GPU 训练任务并行时绝对数值不可比（RTF 0.7→14）；同一次
  运行内的相对比较才有效。

修复尝试（桶式 KV 补零 + 加性掩码，掩码值从 -inf 改为有限 -10000 以绕开
MPSGraph 的 BF16 加法精度提升）：**数值上不可行**。24 层 BF16 递归编码器
把 softmax 归约分段的 1-ULP 变化放大到 encoder states max 0.25 /
mean 0.0147（密封 gate 容差 0.05/0.01），136 项断言失败。该现象与
MiniCPMAudioBF16Attention.swift 头注所述"layer 7 放大"一致。重封存该
gate 需要官方 oracle（pinned upstream `/tmp/MiniCPM-o-Demo` 已被系统
清理，需重克隆后重建 golden fixture）。

已回退全部桶式/掩码改动，`E2EMiniCPMAudioGoldenTests` 重新通过
（6 tests, 0 failures）——生产音频路径保持钉死的比特等价状态。

生产影响评估：桶式优化只对"真实静音/真实语音 packet"的编码成本有意义
（每包 ~400ms → ~16ms）；当前生产路径的静音包已走 continuation
promotion（~1ms），真实语音包的 0.4s 编码是模型前向的固有成本。后续
若要落地：①重克隆 pinned demo 并重建 golden；②桶式补零 + 掩码按本文档
实现（BF16Attention 掩码占位符走 FP32 加法后回 BF16）；③以相同 gate
重新密封。另有稳健性待办：denormal 级输入样本会经 mel 的 log 产生 NaN
并沿编码器→LLM→TTS 链路传播（诊断探针实测），建议在 mel 能量下限处
clamp。

## 14. 紧急回归修复（2026-09-13 深夜）：短答全静音

用户报告"语音输出没声了"。会话取证（sess_B9D7AF32D722）+ 波形分析发现
第 10 节的"文本-only 收集"引入回归：语义 TTS 条件被剥掉了
`<|tts_bos|>`/`<|speak|>` 启动标记——它们是 TTS 解码器的语音启动信号
（P3-A 密封条件的一部分）。中段 unit 靠 `minNewTokens=26` 强制出码掩盖
了问题；而真实对话的短答几乎每个 unit 都 yield（endOfTurn=true →
`minNewTokens=0`）→ TTS 立即 EOS → 0 语音码 → 单元整段静音。

修复：
- 收集规则恢复钉死的 `index != 0`（P3-A parity 条件原样），think 标记
  禁令保留（二者正交）。
- 纯协议 unit 的垃圾音频改在 TTS 门控处拦截：解码文本为空（收集序列
  全为协议标记）时跳过 vocoder，不发音频 delta。
- 协议 trace 测试恢复钉死断言。

重要方法论教训：排查当晚的"NaN/1e38 垃圾音频"是**分析脚本伪影**——
blob WAV 为 16-bit PCM，脚本按 float32 误读。修正采样宽度解读后：
所有历史会话的音频都是真实语音电平（RMS 0.02–0.13），v7 构建的模拟
浏览器会话 31 块全部为连续真实语音（RMS 0.06–0.13）、TTS 码连续推进
（25→650）。23:19 会话（v5）本身就有 91 块完整长故事音频。

## 15. 官方源码对照（2026-09-13 深夜，/tmp/MiniCPM-o-Demo 已重克隆至 pinned commit）

逐 token 对照 `MiniCPMO45/modeling_minicpmo_unified.py` 的
`streaming_generate` 循环与我们的 `MiniCPMNativeDuplexEngine.generate`：

- listen→tts_bos 强制转换（turn 未结束时）：一致 ✓
- 收集规则 `j != 0`（含被 coerce 的 tts_bos、speak、turn_eos；index-0 的
  文本 token 被丢弃不收集）：一致 ✓（已恢复钉死规则）
- 28 字符上限在 j!=0 时拒绝越界 token 并强制 chunk_eos：一致 ✓
- chunk_eos 不可被采样（forbidden），20 步上限强制注入：一致 ✓
- 终止符延迟与 </unit> 合并 feed：一致 ✓

**确认的结构性差异**：官方 `streaming_prefill` 对 AUDIO/OMNI 模式的每个
unit 都要求并携带真实音频 embeddings（`has_audio` 必需，无 continuation
旁路）——即官方上下文是 [audio_emb][<unit>][generate] 连续音频时间线；
我们的 continuation promotion 则喂 [<unit>] 无音频。这解释了模型在
unit 开头采样 <|listen|>（被 coerce）、句中 yield、边界断字的倾向：
模型对"上一秒发生了什么"失去听觉上下文。

落地真实音频上下文的前置条件与路径：
1. 音频编码器桶式补零（本文档第 13 节已实现并验证 ~16ms/块，但与官方
   golden 的 1-ULP 放大偏差 0.25 超出密封容差）——需要先重建 golden
   基线并产品化决策"接受实现级偏差换上下文正确性"；
2. 或优化编码器每形状重编译（MPSGraph plan 桶化需掩码占位符，受
   BF16 加法精度提升限制；MLX 侧可用 compile(shapeless:) 但需逐层验证）；
3. 或混合策略：每 N 块插入一次真实静音编码，折衷成本与上下文。

对照实验遗留：官方 PyTorch/MPS 路线（backend=mps）对无 Origin 头的
WebSocket 探针返回 403，探针对照实验需加 Origin 头或改用浏览器。
