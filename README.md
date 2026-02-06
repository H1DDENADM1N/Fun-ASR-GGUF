# Fun-ASR-GGUF

将 [Fun-ASR-Nano](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) 模型转换为可以在本地高效运行的格式，实现**准确、快速的离线语音识别**，可直接转录长音频生成 SRT。主要依赖了 [llama.cpp](https://github.com/ggml-org/llama.cpp) 对 LLM Decoder 的加速推理。

### 核心特性

- ✅ **纯本地运行** - 无需网络，数据不外传
- ✅ **速度快** - 混合推理架构，支持 GPU 加速 (CUDA, Vulkan, Metal)
- ✅ **准确率高** - Encoder 保持 FP32 精度
- ✅ **内存占用小** - CTC Decoder 和 LLM Decoder 使用 INT8 量化
- ✅ **上下文增强** - 可提供上下文信息，进一步提升识别准确率
- ✅ **支持热词** - 通过 CTC 预识别，基于音素提取热词，提高专业领域识别准确率，实时监控热词文件
- ✅ **时间戳精确** - 字符级时间戳对齐
- ✅ **较长音频支持** - 单次可识别长达 60 秒的音频，超长音频自动分段，智能对齐
- ✅ **SRT 导出** - 支持自动生成美化后的 SRT 字幕

[转录速度视频演示（RTX5050）](https://github.com/user-attachments/assets/3bbd3f77-6aa9-4387-a53d-7859d1b7f5a5)

## 快速开始

### 1. 安装依赖

导出模型需要：

```bash
uv sync  # 先排除其他 group 干扰
uv sync --group gpu --group export-model


# pip install torch torchaudio transformers onnxruntime modelscope onnxscript sentencepiece
```

推理需要：

```bash
uv sync  # 先排除其他 group 干扰
uv sync --group gpu  # onnxruntime-gpu
# uv sync --group cpu  # onnxruntime


# pip install -r requirements.txt
# pip install onnx onnxruntime numpy pydub gguf watchdog rich pypinyin srt
```

>  `pydub` 用于音频格式转换，需要系统安装 [ffmpeg](https://ffmpeg.org/download.html)


从 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 下载预编译二进制文件，将动态库放入 `fun_asr_gguf/bin/` 文件夹：

| 平台        | 下载文件                             |
| ----------- | ------------------------------------ |
| **Windows** | `llama-bXXXX-bin-win-vulkan-x64.zip` |
| **Linux**   | `llama-bXXXX-bin-ubuntu-x64.zip`     |
| **macOS**   | `llama-bXXXX-bin-macos-arm64.zip`    |

> Linux 和 macOS 尚未经过完整测试，欢迎反馈

### 2. 下载模型（可选，如已有导出模型可跳过）

下载原始模型

```bash
uv sync  # 先排除其他 group 干扰
uv sync --group gpu --group export-model
# pip install modelscope
modelscope download --model FunAudioLLM/Fun-ASR-Nano-2512 --local_dir ./Fun-ASR-Nano-2512
```

导出模型

```bash
# 导出 Encoder (FP32) + CTC Decoder (INT8)
uv run 01-Export-Encoder-Adaptor-CTC.py
# python 01-Export-Encoder-Adaptor-CTC.py

# 将 onnx 量化为 fp16 和 int8
python 02-Quantize-ONNX.py

# 导出 LLM Decoder (INT8)
uv run 02-Export-Decoder-GGUF.py
# python 02-Export-Decoder-GGUF.py
```

### 3. 运行识别

```python
from fun_asr_gguf import create_asr_engine

# 创建并初始化引擎 (推荐使用单例或长期持有实例)
engine = create_asr_engine(
    encoder_onnx_path="model/Fun-ASR-Nano-Encoder-Adaptor.fp32.onnx",
    ctc_onnx_path="model/Fun-ASR-Nano-CTC.int8.onnx",
    decoder_gguf_path="model/Fun-ASR-Nano-Decoder.q8_0.gguf",
    tokens_path="model/tokens.txt",
    hotwords_path="hot.txt", # 可选：热词文件路径，支持运行期间实时修改
    similar_threshold=0.6,   # 可选：热词模糊匹配阈值，默认 0.6
    max_hotwords=10,         # 可选：最多提供给 LLM 的热词数量，默认 10
)
engine.initialize()

result = engine.transcribe("audio.mp3", language="中文")
print(result.text)
```

就这么简单！

> 单段音频长度在60秒内可准确识别，过长会自动分段

---

## 工作原理

```
音频输入
    ↓
┌─────────────────────────────────────────────┐
│  Encoder (ONNX, FP32)        → 音频特征      │  保持最高精度
│  CTC Decoder (ONNX, INT8)    → 粗识别结果    │  快速预识别
└─────────────────────────────────────────────┘
    ↓              ↓              ↓
  音频特征      时间戳         热词候选
    ↓              ↓              ↓
┌─────────────────────────────────────────────┐
│  构建 Prompt (Prefix + 音频 + Suffix)        │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  LLM Decoder (GGUF, INT8, llama.cpp)        │  支持 Vulkan GPU
│  ↓                                          │
│  生成最终识别文本                             │
└─────────────────────────────────────────────┘
    ↓
  时间戳对齐 → 输出结果
```

### 为什么这样做？

1. **CTC 预识别** 提供两样东西：
   - 粗识别结果（用于筛选热词，提供给LLM）
   - 时间戳信息（为LLM的文本输出赋予时间戳）

2. **混合架构** 各取所长：
   - ONNX Runtime：运行 Encoder 和 CTC，稳定可靠
   - llama.cpp：运行 LLM，支持 GPU 加速(Vulkan, Metal)，速度超快

3. **精度策略**：
   - **Encoder 用 FP32** - 保证识别准确率（INT8 影响大）
   - **CTC 和 LLM 用 INT8** - 节省内存，速度提升，准确率影响小

### 内存占用

 - **Encoder**（约 200M 参数）
   - FP16：~400MB 内存（DML加速）
   - INT8：~200MB 内存（CPU加载）
 - **CTC Decoder**：内存占用可忽略（几十 MB）
 - **LLM Decoder**（约 600M 参数）
   - INT8：~1.2GB 内存或显存（显卡内会用fp16进行计算，但比原生fp16要快）
   - FP16：~1.2GB 内存或显存
   - 另外 4096 的上下文也会占用 400M 显存

 **总内存占用**（推荐 FP16 Encoder + INT8 Decoder）：约 **1.8GB**（CPU 内存或 GPU 显存）

---

## 使用示例

### 基础用法

```python
from fun_asr_gguf import create_asr_engine

engine = create_asr_engine(
    encoder_onnx_path="model/Fun-ASR-Nano-Encoder-Adaptor.fp32.onnx",
    ctc_onnx_path="model/Fun-ASR-Nano-CTC.int8.onnx",
    decoder_gguf_path="model/Fun-ASR-Nano-Decoder.q8_0.gguf",
    tokens_path="model/tokens.txt",
    hotwords_path="hot.txt",  # 可选：热词文件路径
    similar_threshold=0.6,    # 可选：热词匹配阈值
    max_hotwords=10,          # 可选：最多召回热词数
)
engine.initialize()


result = engine.transcribe(
    "input.mp3", 
    language="中文", 
    context="这是睡前消息的音频，主持人叫督工", 
    verbose=True,       # 打印细节
    segment_size=60.0,  # 分片60秒
    overlap=4.0,        # 片间重叠4秒
    start_second=0.0,   # 从第零秒开始
    duration=300.0,     # 转录到第300秒结束
    srt=True            # 输出 SRT 字幕文件
)
print(result.text)           # 识别文本
print(result.segments)       # 带时间戳的分段
print(result.timings)        # 各阶段耗时
```


### 热词配置

创建 `hot.txt` 文件（每行一个热词）：

```text
督工
静静
深度学习
神经网络
```

**特性：**
1. **实时更新**：识别程序运行期间，你可以随时修改 `hot.txt` 并保存，程序会通过 `watchdog` 自动更新内存中的热词库，无需重启。
2. **模糊召回**：热词可以有几千条、上万条，程序会根据 CTC 的粗识别结果进行音素级别的模糊匹配，找出相似度在 `similar_threshold` 以上的热词，并取前 `max_hotwords` 条（默认10条）作为上下文提供给 LLM Decoder 从而得到更准确的输出。

---

## 性能参考

以下是在小新Pro16GT（U9-258H + RTX5050）笔记本上的效果，60秒的睡前消息音频，转录用时1.89秒。

需要注意的是，**LLM Decoder 所需时间取决于吐出文字的数量，不适合用 RTF 描述**，睡前消息音频的文字密度非常高，短短60秒就有350个字，但这段音频的速度可以作为下限参考，即 RTF 最慢也不会慢过 0.03

在文字密度更低的音频上，识别速度还能更快。

```
[统计]
  音频长度:  60.00s
  Decoder输入:  24689 tokens/s (总: 204, prefix:73, audio:126, suffix:5)
  Decoder输出:    219 tokens/s (总: 253)

[转录耗时]
  - 音频编码：   359ms
  - CTC解码：     67ms (Infer: 36ms, Dec: 1ms, HW: 30ms)
  - LLM读取：      8ms
  - LLM生成：   1153ms
  - 总耗时：    1.89s

✓ 字幕已导出至: input.srt

------------------------------ 完整转录文本 ------------------------------
大家好，2026年1月11日星期日，欢迎收看1004期《睡前消息》。请静静介绍话题。去年10月19日967 
期节目说到委内瑞拉问题，我们回顾一下你当时的评论。无论是从集结的兵力来看，还是从动机来看 
，特朗普政府并不打算对委内瑞拉政权发动全面的进攻，最多是发动象征性的轰炸进行政治投机。在 
诺贝尔和平奖发给了委内瑞拉反对派之后，美国军队进攻的概率进一步降低。现在美国突袭委内瑞拉 
，抓走了总统马杜罗，督工你怎么看待两个月之前的判断？当初的判断不变，美国对于委内瑞拉的突 
袭性质依然是政治投机，不能算是地面战争。入侵的美国军队总数是一两百，站在委内瑞拉领土上的 
时间不超过一个小时，算是地面战争或者全面进攻，实在有点勉强。当然，美国东用总力量并不小，
150架先进飞机加上经年累月部署的情报网络，这放在东亚或者欧洲也不是一只很小的力量。用到美国 
的西半球主场压倒委内瑞拉的军队那是必然的。
--------------------------------------------------------------------------
```

同一段音频，纯 CPU 推理速度：

```
[统计]
  音频长度:  60.00s
  Decoder输入:    313 tokens/s (总: 203, prefix:72, audio:126, suffix:5)
  Decoder输出:     48 tokens/s (总: 256)

[转录耗时]
  - 音频编码：   727ms
  - CTC解码：     95ms (Infer: 65ms, Dec: 1ms, HW: 29ms)
  - LLM读取：    649ms
  - LLM生成：   5334ms
  - 总耗时：    7.19s
```

长音频推理输出示例：

```
[1] 加载音频...
    音频长度: 300.00s
    检测到长音频，开启分段识别模式...

[转录耗时]
  - 音频编码：  1690ms
  - CTC解码：    961ms (Infer: 210ms, Dec: 6ms, HW: 744ms)
  - LLM读取：     56ms
  - LLM生成：   5870ms
  - 总耗时：    9.90s

✓ 字幕已导出至: input500.srt

```
---

## 性能参考 2
以下是在 Redmi Book Pro 14 2022 (AMD Ryzen 7 6800H with Radeon Graphics) 笔记本上的效果，60秒的睡前消息音频，GPU 转录用时约 5.9 秒，RTF 0.098；CPU 转录用时约 9.12 秒，RTF 0.152。
需要注意的是，**LLM Decoder 所需时间取决于吐出文字的数量，不适合用 RTF 描述**，睡前消息音频的文字密度非常高，短短60秒就有252个字，但这段音频的速度可以作为下限参考，即 RTF 最慢也不会慢过 0.15。
在文字密度更低的音频上，识别速度还能更快。
### GPU srt 字幕转录 (AMD Radeon Graphics Rembrandt)
```
[统计]
  音频长度:  60.00s
  Decoder输入:   5864 tokens/s (总: 204, prefix:73, audio:126, suffix:5)
  Decoder输出:     70 tokens/s (总: 252)

[转录耗时]
  - 音频编码：  1628ms
  - CTC解码：    254ms
  - LLM读取：     35ms
  - LLM生成：   3594ms
  - 总耗时：    5.90s

✓ 字幕已导出至: input.srt
```
### CPU srt 字幕转录 (AMD Ryzen 7 6800H, 4675 MHz)
```
[统计]
  音频长度:  60.00s
  Decoder输入:    367 tokens/s (总: 203, prefix:72, audio:126, suffix:5)
  Decoder输出:     41 tokens/s (总: 255)

[转录耗时]
  - 音频编码：  1603ms
  - CTC解码：    248ms
  - LLM读取：    553ms
  - LLM生成：   6224ms
  - 总耗时：    9.12s

✓ 字幕已导出至: input.srt
```

---

## 常见问题

**Q: Encoder 和 CTC 如何选择量化精度？**  
- Encoder-Adapter 以及 CTC Decoder 都是在用 onnxruntime 加速，它既支持 CPU，也支持通过 DirectML 使用显卡加速。CPU 跑 int8 最快，GPU 跑 fp16 最快。


**Q: 支持哪些语言？**  
- Fun-ASR-Nano-2512 支持中文、英文、日文。
- Fun-ASR-MLT-Nano-2512 还支持更多语言（粤语、韩文、越南语等）。

**Q: 如何提高识别准确率？**  
- 配置 `hot.txt` 添加领域热词
- 使用 `context` 参数提供上下文信息

**Q: 输出全是「!!!!!!!!!!」怎么办？**  
- Intel 集成显卡的 fp16 矩阵求和计算没有使用 fp32 做临时变量，溢出为 NaN，产生了如此问题，解决办法是通过设置环境变量禁用 fp32 ，或禁用 vulkan。取决于 CPU 更快，还是集显的 fp32 更快。

```python
os.environ["VK_ICD_FILENAMES"] = "none"       # 禁止 Vulkan
os.environ["GGML_VK_VISIBLE_DEVICES"] = "0"   # 禁止 Vulkan 用独显（强制用集显）
os.environ["GGML_VK_DISABLE_F16"] = "1"       # 禁止 VulkanFP16 计算（Intel集显fp16有溢出问题）
```



---

## 文件说明

### 项目结构

- `01-Export-Encoder-Adaptor-CTC.py` - 导出 Encoder 和 CTC Decoder
- `02-Export-Decoder-GGUF.py` - 导出 LLM Decoder
- `03-Inference.py` - 完整的使用示例
- `fun_asr_gguf/`
  - `core/` - 核心逻辑（资源管理、解码、编排）
  - `bin/` - 存放 DLL 和二进制可执行文件
  - `asr_engine.py` - Facade 入口类
  - `srt_utils.py` - 字幕生成工具
  - `text_merge.py` - 滑动窗口文本合并算法

### 导出的模型

```
model/
├── Fun-ASR-Nano-Encoder-Adaptor.fp32.onnx   # 音频编码器 (FP32)
├── Fun-ASR-Nano-CTC.int8.onnx               # CTC 解码器 (INT8)
├── Fun-ASR-Nano-Decoder.q8_0.gguf           # LLM 解码器 (INT8)
└── tokens.txt                               # CTC Token 映射
```

---

## 技术细节

### 架构设计

- **Encoder**：ONNX 格式，FP32，提取音频特征
- **CTC Decoder**：ONNX 格式，INT8 量化，用于时间戳和热词候选
- **LLM Decoder**：GGUF 格式，INT8 量化，llama.cpp 推理


---

## 致谢

- [Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR) - 原始模型
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - GGUF 推理引擎
