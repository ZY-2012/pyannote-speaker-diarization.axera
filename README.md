# pyannote-speaker-diarization.AXERA

`pyannote/speaker-diarization-community-1`（cc-by-4.0）说话人日志模型在 AX650N 上的部署工程：
输入 16 kHz 单声道会议音频，输出 RTTM 说话人时序标签。

- [x] Python 板端推理（axengine）
- [x] C++ 板端推理（RTF 0.057）
- [x] 模型转换（导出 → 对分 → 校准 → Pulsar2 量化）

量化模型与 C++ 可执行文件发布在 HuggingFace：
[HY-2012/pyannote-speaker-diarization.axera](https://huggingface.co/HY-2012/pyannote-speaker-diarization.axera)，
本仓库仅含全部代码。

## 支持平台

- AX650N（NPU3）

## 模型

- 模型：`pyannote/speaker-diarization-community-1`（分割 + 嵌入 + PLDA/VBx 聚类，cc-by-4.0）
- 量化：8 个 axmodel，U16 激活 / S8 权重；InstanceNorm / TSTP 池化在 U16 域数值崩坏 → 主机 FP32
- 部署架构详见 [板端评测记录.md](板端评测记录.md)

## 目录结构

```
├── [model_convert/](model_convert/)       # 模型转换（导出 → 校准 → 量化）
│   ├── export_sincnet_split.py    # 分割前端 3 段 ONNX + 主机参数
│   ├── export_lstm_cells.py       # 4 层 BiLSTM 96 步 cell
│   ├── export_models.py           # embedding 卷积栈 + fbank 参数
│   ├── generate_calibration.py    # 真实会议校准数据
│   ├── generate_cell_calib.py     # LSTM cell 校准数据
│   └── compile.sh                 # 一键编译 8 个 axmodel
├── [python/](python/)             # 板端 Python 推理（community1_sdk + example.py）
├── [cpp/](cpp/)                   # C++ 推理源码（见 cpp/README.md）
├── [benchmark/](benchmark/)       # 测试集下载/参考处理/评分脚本
├── [scripts/](scripts/)           # 主机参数导出（npz → bin，C++ SDK 用）
├── [samples/](samples/)           # 演示音频（2 人会议 120 s）
└── run_ax650.sh                   # 一键运行（板端，需先下载 HF 模型）
```

## 环境

```bash
conda create -n community1-diar python=3.12
conda activate community1-diar
cd pyannote-speaker-diarization.AXERA
pip install -r requirements.txt
```

板端推理另需：

```bash
pip install numpy scipy scikit-learn soundfile axengine
```

C++ 交叉编译工具链（aarch64 gcc 9.2 + AX650N BSP SDK）：见 [cpp/README.md](cpp/README.md)。

## 模型转换（三步）

1. **导出**：`export_sincnet_split.py` + `export_lstm_cells.py` + `export_models.py`（自动对分验证）
2. **校准**：`generate_calibration.py` + `generate_cell_calib.py`（真实会议数据）+ `pack_calib.sh`
3. **编译**：`compile.sh`（Pulsar2 Docker，8 个 axmodel）

详见 [model_convert/README.md](model_convert/README.md)。

## 板端部署

### Python 推理

```bash
# 下载模型（models/ 下 23 个文件，或 hf download）
pip install -r requirements.txt
bash run_ax650.sh                    # 自带样例 samples/sample_meeting.wav
bash run_ax650.sh in.wav out.rttm
```

### C++ 推理

```bash
# 本地交叉编译（见 cpp/README.md），产物 community1_diar
bash cpp/download_toolchains.sh
bash cpp/build_ax650.sh

# 板端运行（直接用 HuggingFace 仓库 bin/community1_diar_ax650）
export LD_LIBRARY_PATH=/soc/lib:${LD_LIBRARY_PATH:-}
./community1_diar --model-dir models --wav in.wav --out out.rttm --step 2.0
```

## RTF（AX650N 实测，samples/sample_meeting.wav 120 s）

| 推理路径 | 耗时 | RTF |
|------|------|------|
| C++（8 线程，step=2.0） | 6.9 s | 0.057 |
| Python（step=2.0） | 19 s | 0.16 |

> RTF = 推理耗时 / 音频时长（不含模型加载）；step=2.5 再快约 20%（+0~1.5 pp）。

## 指标（帧级 DER）

| 数据集 | community-1 板端 | 3D-Speaker 板端¹ | community-1 GPU |
|---|---|---|---|
| AMI dev12（no collar） | **20.06%** | 29.72% | 20.08% |
| AliMeeting eval（±0.125 s） | **20.88%** | 29.34% | 18.84% |
| AliMeeting eval（±0.25 s） | **17.26%** | 24.36% | 15.16% |
| VoxConverse test（±0.125 s） | **9.11%** | 9.27% | 8.49% |

¹ 3D-Speaker 板端 = FSMN VAD + CAM++ + 谱聚类量化管线，RTF 0.046~0.047。

测试集下载、参考处理与评分脚本见 [benchmark/README.md](benchmark/README.md)。

## 参考

- [pyannote-audio](https://github.com/pyannote/pyannote-audio)（MIT）
- [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1)（cc-by-4.0）
