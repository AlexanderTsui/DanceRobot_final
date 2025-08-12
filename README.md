# 舞蹈机器人主控系统

本项目是一个基于Python的舞蹈机器人主控系统，集成了视觉识别、语音控制和音乐匹配三大核心功能，通过一个稳健的状态机来协调机器人的各种行为。

## ✨ 主要特性

- **多模态交互**: 支持通过语音指令和视觉姿态两种方式对机器人进行实时控制。
- **智能舞蹈模式**: 能在特定模式下，通过麦克风采集环境音乐，并从预设的曲库中识别出对应的歌曲，将歌曲编号发送给下位机。
- **状态机驱动**: 采用事件驱动结合状态轮询的设计模式，确保了系统在高负载下的稳定性和响应性。
- **模块化设计**: 各个核心功能（视觉、通信、音乐识别）高度解耦，易于维护、测试和扩展。
- **健壮的通信**: 与下位机（ECU）的通信包含完整的重试和确认机制，保证了指令的可靠送达。

## ⚙️ 系统架构

系统由一个主控单元（`main.py`）调度三大核心模块，并通过串口与硬件进行通信。

```
舞蹈机器人主控系统
├── 主控单元 (main.py)
│   ├── 状态管理器 (current_mode)
│   └── 异步任务处理器 (threading, queue)
├── 通信模块 (serial_communication.py)
├── 视觉识别模块 (vision_recognize.py)
└── 音乐识别模块 (music_recognizer.py)
```

## 🚀 快速上手

### 1. 环境准备

**硬件要求:**
- 一台性能足够的开发主机（如 树莓派4B 或更高配置的PC/笔记本电脑）
- 一个USB摄像头 (支持 640x480 分辨率)
- 两个可用的串口设备 (或USB转串口模块)
- 一个麦克风

**软件依赖:**
- Python 3.7+
- 所需的Python包已在 `requirements.txt` 中列出。

### 2. 安装步骤

1.  **克隆项目**
    ```bash
    git clone <your-repository-url>
    cd DanceRobot3
    ```

2.  **安装Python依赖**
    建议使用虚拟环境以避免包冲突。
    ```bash
    # (可选) 创建并激活虚拟环境
    python3 -m venv venv
    source venv/bin/activate 
    
    # 安装所有必需的库
    pip install -r requirements.txt
    ```

3.  **生成音乐指纹**
    在第一次运行主程序之前，必须先为参考音乐生成指纹库。
    ```bash
    python3 src/modules/music_recognize/fingerprint_generator.py
    ```
    这条命令会处理 `data/reference_music/` 目录下的所有音频，并将生成的指纹库保存在 `data/fingerprints/` 中。

### 3. 配置

在运行前，您可能需要根据实际的硬件连接情况，修改 `src/main.py` 文件顶部的串口配置：

```python
# src/main.py
ECU_SERIAL_PORT = '/dev/ttyUSB0'  # 修改为你的电控单元（ECU）串口
VOICE_SERIAL_PORT = '/dev/ttyUSB1' # 修改为你的语音模块串口
```

## 🎮 如何使用

完成安装和配置后，直接运行主程序即可启动系统：

```bash
python3 src/main.py
```

程序启动后会进入待机（IDLE）模式，等待来自ECU的指令进行模式切换。

### 测试独立模块

如果需要单独测试某个功能模块，可以运行对应的脚本：
- **视觉模块测试**: `python3 src/modules/vision/vision_recognize.py`
- **音乐识别模块测试**: `python3 src/modules/music_recognize/music_recognizer.py`

## 📂 项目结构

```
.
├── data/
│   ├── fingerprints/         # 存放生成的音乐指纹
│   ├── recordings/           # 存放临时的录音文件
│   └── reference_music/      # 存放用于匹配的参考音乐
├── src/
│   ├── main.py               # 程序主入口，包含状态机和主循环
│   └── modules/
│       ├── communicate/      # 封装了带重试的串口通信逻辑
│       ├── music_recognize/  # 音乐识别模块，我们工作的主要成果
│       └── vision/           # 视觉姿态识别模块
├── requirements.txt          # Python依赖列表
└── README.md                 # 本文档
```

## 🤖 工作流程

系统启动后进入一个高速轮询的主循环，其核心逻辑由一个状态机驱动。

### 状态机

系统通过来自ECU的指令在不同模式间切换：

| 收到指令 | 切换模式 | 描述 |
|:---:|:---|:---|
| `"1"` | 语音识别 (VOICE) | 监听语音模块，并将识别结果转发给ECU |
| `"2"` | 视觉识别 (VISION) | 启动摄像头，识别特定姿态并发送指令 |
| `"3"` | 舞蹈模式 (DANCE) | 异步启动音乐识别，并将歌曲ID发送给ECU |

### DANCE模式详解

为了避免阻塞主循环，DANCE模式采用异步多线程处理：
1.  **启动**: 当ECU指令切换到DANCE模式时，主线程会立即启动一个**后台线程**来执行耗时10秒的音乐识别任务，自身则继续高频轮询。
2.  **等待**: 后台线程完成识别后，会将结果（歌曲ID）放入一个线程安全的队列中。
3.  **获取**: 主线程在DANCE模式下轮询时，会以非阻塞方式检查队列。
4.  **处理**: 一旦从队列中获取到结果，主线程就将其发送给ECU，并自动切换回待机模式，完成一次完整的识别流程。

这种设计确保了即时在进行音乐识别，主系统依然能对其他指令保持高响应性。

