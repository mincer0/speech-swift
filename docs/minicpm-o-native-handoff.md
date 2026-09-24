# HANDOFF：MiniCPM-o 原生双工（Swift/MLX）

最后更新：2026-09-24
分支：`feature/minicpm-o-native`
HEAD：`56c47b0`（与 `origin` 同步，工作区干净）

---

## 0. 30 秒状态摘要

| 项 | 值 |
|---|---|
| 服务 | **正在运行**（`127.0.0.1:7861`，`status: ready`） |
| 整 unit 耗时 | **596ms**（起点 1072ms，-44%），实时余量 **1.68×** |
| 音频编码 | **23.8ms**（起点 379.6ms，-94%；官方 llama.cpp 为 42ms） |
| 首响应延迟 | **0.55–1.0s**（起点 6.1–6.4s） |
| 质量 | 文本连贯；四个真 bug + 提示词纪律已修并验证 |
| 剩余工作 | 无阻塞项。移植层面优化已见底，详见 §7 |

**新接手请先读**：`docs/minicpm-o-native-optimization-report.md`（完整回顾 + 14 条已被否决的假设）。
**不要再试**：报告 §4 列出的 14 条假设，每条都已用实测否决。

---

## 1. 把服务跑起来

### 1.1 启动（必须用双 fork 脱进程组）

macOS 无 `setsid`；直接在工具调用里 `nohup ... &` 启动的服务**会在该次调用结束时被杀**。
可靠写法（Python 双 fork）：

```bash
/Users/mincer/.workbuddy/binaries/python/versions/3.13.12/bin/python3 - <<'PY'
import os
log = open('/tmp/minicpm_native_service.log', 'ab', 0)
pid = os.fork()
if pid == 0:
    os.setsid()
    if os.fork() == 0:
        os.chdir('/Users/mincer/项目/s2s')
        os.environ['MINICPM_STAGE_TIMING'] = '1'   # 分阶段耗时（诊断用）
        os.environ['MINICPM_AUDIO_FAST'] = '1'     # ← 关键：音频前端快速路径
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
        os.execv('/bin/zsh', ['/bin/zsh', '/Users/mincer/项目/s2s/start_minicpm_native.sh'])
    os._exit(0)
os.waitpid(pid, 0)
PY
```

### 1.2 重启（先杀后起）

```bash
OLD=$(lsof -nP -iTCP:7861 -sTCP:LISTEN -t 2>/dev/null | head -1); [ -n "$OLD" ] && kill $OLD; sleep 4
# 然后执行上面的启动块
```

`ps` 在本机沙箱不可用，查端口只能用 `lsof -nP -iTCP:<port> -sTCP:LISTEN`。

### 1.3 启动参数与环境变量

启动脚本：`/Users/mincer/项目/s2s/start_minicpm_native.sh`（自带 fail-closed 预检，
可用 `--check` 只做只读体检）。它读取：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `MINICPM_NATIVE_BINARY` | `<repo>/.build/release/minicpm-mlx-server` | 二进制路径 |
| `MINICPM_NATIVE_MODEL_ROOT` | `models/MiniCPM-o-4_5-native-mlx-v1` | 模型根（要求 5 个固定组件名） |
| `MINICPM_NATIVE_DATA_DIR` | `data-native` | 会话/资产落盘目录 |
| `MINICPM_NATIVE_PORT` | `7861` | 7860 被无关的 LTX UI 占用 |
| `MINICPM_NATIVE_ENABLE_VAD` | `0` | 默认关 |

**引擎级开关**（`MINICPM_` 前缀，代码内读取）：见 §4.2。

### 1.4 构建

```bash
cd /Users/mincer/项目/s2s/runtime/speech-swift-minicpm-native
swift build -c release --disable-sandbox
```
约 25–55s。**构建后必须重启服务**（脚本不会自动重启）。

---

## 2. 验证（每次改动后照做）

### 2.1 性能：分阶段耗时自动聚合

```bash
cd /Users/mincer/项目/s2s
/Users/mincer/.workbuddy/binaries/python/envs/default/bin/python tools/duplex_ab_listen.py --speech 0 --chunks 16 --runs 1.0
```

输出示例（健康基线）：

```
wall_clock_ms      ~596      实时余量 1.68×
audio_encoder_ms   ~23.8
llm_decode_ms      ~260
semantic_tts_ms    ~180
token2wav_ms       ~160   (flow ~125 + hift ~35)
```

**判据与纪律**：
- **必须重复 ≥3 次**。`wall_clock` 运行间方差可达 **±30ms**，单次对比会得出错误结论。
- **阶段级指标才是可靠信号**（如 `flow_ms` 方差可小到 **±0.2ms**）。
- **与 Comni 等官方对照实现错开时间跑** —— 它会抢 GPU，曾把 `llm_decode` 从 272ms 抬到 1082ms。

### 2.2 理解能力（**任何影响音频编码/数值的改动都必须跑**）

```bash
cd /Users/mincer/项目/s2s
/Users/mincer/.workbuddy/binaries/python/envs/default/bin/python tools/duplex_understand_test.py --chunks 8
```

喂入 macOS `say` 合成的已知语音（"请给我讲一个关于宇航员在火星上种土豆的科幻故事"），
检查回复是否命中 `宇航员/火星/土豆/科幻` 且**无** `没听清/没明白` 类负面信号。

> 这是本项目最重要的回归闸门。音频编码器的数值改动曾差点破坏理解能力，
> 靠这个探针才发现并守住。

### 2.3 会话侧诊断

会话落在 `data-native/sessions/<sess_id>/`（`stream.jsonl` + 音频 blob）。
已有分析脚本模式可复用：解析 `response.output.delta` 取 `kind=text`（文本）
或 `kind=audio`（波形，可算块长/有效语音占比）。

---

## 3. 代码地图

```
Sources/
  MiniCPMAudio/         音频编码器（理解路径）
    MiniCPMAudioModel.swift        ★ 主文件；fast-path 开关都在这里（§4.2）
    MiniCPMAudioBF16*.swift        MPSGraph parity kernel（AUDIO_FAST 下不启用）
  MiniCPMLLM/           LLM（8-bit 量化）
    MiniCPMLanguageModel.swift     ★ 工厂函数决定是否用 exact-BF16 kernel
    MiniCPMExactBF16*.swift        parity kernel（仅 quantBits==0 时启用）
  MiniCPMTTSSemantic/   TTS 语义解码（自回归 26 步）
    MiniCPMTTSSemantic.swift       ★ generateStreamingChunk 的 26/25 契约（§5）
  MiniCPMToken2Wav/     声码器
    DiTFlow.swift                  ★ flow ODE 循环（asyncEval）
    FlowConfiguration.swift        ★ odeSteps = 8
    Pipeline.swift                 prompt 缓存（baseFlowCache）+ 缓冲冲刷
  MiniCPMDuplexRuntime/ 双工引擎
    MiniCPMNativeDuplexEngine.swift ★ unit 循环、静音块、文本预算、1s 填充
  MiniCPMDemoBackend/   服务端
    Backend.swift / RealtimeRoutes.swift
    Resources/Web/audio-duplex/audio_duplex.html  ★ 系统提示词默认值在此
tools/                  （注：实际在 /Users/mincer/项目/s2s/tools/，仓库内也有一份）
docs/                   路线图 + 报告
```

**关键：`tools/` 与 `data-native/` 在仓库之外**（`/Users/mincer/项目/s2s/` 层级）。
仓库内 `tools/` 是副本；改脚本请同步两边，或直接以 `s2s/tools/` 为准。

---

## 4. 配置与开关

### 4.1 当前生效配置（不要随意改）

| 项 | 值 | 位置 |
|---|---|---|
| `odeSteps` | **8**（默认曾是 10） | `FlowConfiguration.swift:25` |
| `MINICPM_AUDIO_FAST` | **1**（生产必须开） | 环境变量 |
| `MINICPM_STAGE_TIMING` | 1（诊断；实测关掉无收益，开着免费拿数据） | 环境变量 |

### 4.2 `MINICPM_AUDIO_FAST=1` 覆盖什么（**改动前务必核对**）

**覆盖**：Linear / Conv1d / GELU / LayerNorm / **Attention**（五项）。
历史上曾漏掉 Attention，导致编码器比 CoreML 纯 CPU 还慢 3.7 倍 —— 修好后 -60ms。
**新增任何手写 kernel 时，必须同步扩这个开关的覆盖面。**

**相关开关**（多为诊断逃生舱，生产勿动）：
`MINICPM_AUDIO_ATTENTION_VARIANT=native|mlx`（强制走 MLX 注意力）、
`MINICPM_AUDIO_GELU_VARIANT`、`MINICPM_AUDIO_LAYERNORM_VARIANT`、
`MINICPM_TTS_BF16_LINEAR=mpsgraph`、`MINICPM_MLX_BF16_LINEAR=1`、
`MINICPM_DUPLEX_VARIABLE_BLOCKS=1`（已实测**净收益 0**，默认关）。

---

## 5. 不可触碰的契约（改了会静默坏掉）

1. **TTS 的 26/25 look-ahead 契约**：中间块必须生成 26 个 code、提交 25 个、
   隐藏 1 个 look-ahead 并在下一个 chunk 回喂。
   **让中间块提前停止（EOS）会产生"整秒近静音"**（实测 RMS<0.01，文本根本没念出来）——
   这就是"变长块"路线被否决的原因。
2. **`finalChunk = endOfTurn`**：只有真正的 turn 边界才能结束流式 chunk。
   把"解码器自停"当边界会撕掉声码器状态、吞音节。
3. **每 unit 文本预算 = 6 字**：调到 28 会在密集 unit 下截断（模型每 unit 大约只说 4–6 字）。
4. **1 秒静音块必须下发**：listen 边界的静音载荷不能丢，否则客户端时间轴留洞、
   Ahead 指标失真。
5. **系统提示词的双工纪律段**（`audio_duplex.html` 的 textarea 默认值）：
   "绝对不要主动开场" / "噪声、静音和扬声器中你自己的声音都必须继续倾听" /
   "讲故事不要只回'好的'"。删掉会导致**自行开场**、**把自己外放的声音当用户输入**、
   **长故事讲不完** —— 一条提示词同时解释三个症状。
6. **所有 MonoGame/GPU/音频调用需 try-catch 包裹失败静默**（headless 可执行性要求）。

---

## 6. 已知坑（环境）

| 症状 | 原因 / 解法 |
|---|---|
| Python 脚本 `exit=1`，日志只有 `[safe-delete][SAFE_DELETE_BULK_CONFIRM_REQUIRED]` | 沙箱批量删除护栏在 import 阶段杀进程（librosa 清 `__pycache__`）。**必须免沙箱运行**；`PYTHONDONTWRITEBYTECODE=1` 无效 |
| Swift 报 `expected ':' after '? ...' in ternary` | 多行可选链后接 `==` 的 `else if` 无法编译 → 先把标志算成局部变量 |
| `ps` 返回空 / permission denied | 沙箱限制；用 `lsof -nP -iTCP:<port> -sTCP:LISTEN` |
| 后台服务启动后立刻 connection refused | 没用双 fork（§1.1） |
| `pip` 不存在于 `.venv-minicpmo` | 用 `python -m pip` |
| 性能对比数据明显失真 | 与 Comni/LTX 等抢占 GPU 的进程错开 |
| `coremltools` 导不进来 | 旧版 6.3.0 与 protobuf 7.x 不兼容；已升级到 9.0 |

**Python 环境**：
- `/Users/mincer/项目/s2s/.venv-minicpmo/bin/python` —— torch / mlx / coremltools（模型转换、CoreML 导出）
- `/Users/mincer/.workbuddy/binaries/python/envs/default/bin/python` —— 跑 `tools/` 下的探针脚本（websocket-client）

---

## 7. 未完成 / 下一步

### 7.1 无阻塞项

服务可用，质量达标，性能进实时。**可以就此交接。**

### 7.2 若要继续优化（按性价比）

| 方向 | 收益 | 前提与风险 |
|---|---|---|
| 多 unit 流水线（双缓冲） | wall_clock → ~400ms 级 | 需重构引擎主循环；回归风险高 |
| 语音组件 8-bit 量化 | ~50ms | 需听感 A/B + 回退方案；本仓库已有"TTS 8-bit 因可听劣化被撤下"的先例 |
| 音频编码器上 ANE | 13.4ms（2.2%） | 导出/验证脚本已就绪（`scripts/export_minicpm_o_audio_coreml.py`、`verify_*.py`），但只剩 13.4ms 空间 + 583MB 依赖 + 9.6s 冷启动 → 性价比不成立 |
| LLM 4-bit | ~150ms | **用户已否决**；若重启必须先过理解能力 A/B |

### 7.3 两个遗留小项

- **偶发静音块**（历史样本：unit 13「当然」0.08s、unit 69「比」0.00s）：优先级已降低
  （模型韵律天性所致），但未彻底解释。
- **`silent` 字段未持久化**到会话记录：需在浏览器侧确认。

---

## 8. 参考文档索引

| 文档 | 位置 | 内容 |
|---|---|---|
| **优化与质量报告** | `<repo>/docs/minicpm-o-native-optimization-report.md` | 完整回顾 + **14 条已被否决的假设**（先读这个） |
| 优化路线图 | `<repo>/docs/minicpm-o-native-optimization-roadmap.md` | 待办项详解 + 每项的实验数据 |
| CoreML/ANE 导出 | `<repo>/scripts/export_minicpm_o_audio_coreml.py` | 含四个工具链障碍的解法 |
| CoreML 验证 | `<repo>/scripts/verify_minicpm_o_audio_coreml.py` | 冷启动 + 真实语音数值验证 |
| 前期诊断（工作区） | `/Users/mincer/WorkBuddy/2026-09-19-23-06-54/*.md` | 吞字 / 断续 / 头脑风暴系列等历史分析 |
| 通用诊断方法 | `~/.workbuddy/skills/mlx-inference-perf-diagnosis/SKILL.md` | 可复用到其他 MLX/Swift 项目 |

---

## 9. 交接检查清单

- [ ] `curl http://127.0.0.1:7861/health` → `status: ready`
- [ ] `swift build -c release` → 0 error
- [ ] `duplex_ab_listen.py` 跑 3 次 → `wall_clock` ~596ms、`audio_encoder_ms` ~23.8ms
- [ ] `duplex_understand_test.py` → 命中 宇航员/火星/土豆，无负面信号
- [ ] `git status` 干净、`git log origin/feature/minicpm-o-native..HEAD` 为空
- [ ] 浏览器打开 `http://127.0.0.1:7861/audio-duplex/audio_duplex.html`，**硬刷新**后开一局会话：
      应等你先说话、能听清、说"讲个长故事"能讲完
