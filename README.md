# pyannote-speaker-diarization.AXERA

`pyannote/speaker-diarization-community-1`（cc-by-4.0）说话人日志模型的 Axera AX650N
部署工程：输入 16 kHz 单声道会议音频，输出 RTTM 说话人时序标签。

- [x] Python 板端推理（axengine，`python/example.py`）
- [x] C++ 板端推理（`cpp/`，交叉编译，RTF 0.084）
- [x] 模型转换（导出 → 对分验证 → 真实会议校准 → Pulsar2 量化，`model_convert/`）

量化模型与 C++ 可执行文件发布在 HuggingFace：
[HY-2012/pyannote-speaker-diarization.axera](https://huggingface.co/HY-2012/pyannote-speaker-diarization.axera)，
本仓库仅含全部代码。

## 支持平台

- AX650N（NPU3）

## 指标（帧级 DER，板端实测）

| 数据集 | community-1 板端 | 3D-Speaker 板端¹ | community-1 GPU FP32 |
|---|---|---|---|
| AMI dev12（no collar） | **20.06%** | 29.72% | 20.08% |
| AliMeeting eval（±0.125 s） | **20.88%** | 29.34% | 18.84% |
| AliMeeting eval（±0.25 s） | **17.26%** | 24.36% | 15.16% |

¹ 3D-Speaker 板端 = FSMN VAD + CAM++ + 谱聚类量化管线（`3D-Speaker-Meeting-Summary`
demo 的分割部分），无重叠检测，RTF 0.046。

RTF（纯推理，不含模型加载，AX650N）：**0.084**（C++ 8 线程，step=2 s）/
0.16（Python，step=2 s）；step=2.5 s 再快约 20%（板端 +0~1.5 pp）。
完整评测与量化细节见 `板端评测记录.md`。

## 目录结构

```
├── model_convert/           # 模型转换（导出 → 校准 → Pulsar2 量化，8 个 axmodel）
│   ├── export_sincnet_split.py    # 分割前端 3 段 + 主机参数
│   ├── export_lstm_cells.py       # 4 层 BiLSTM 96 步 cell
│   ├── export_models.py           # embedding 卷积栈 + fbank 参数
│   ├── generate_calibration.py    # 真实会议校准数据
│   ├── generate_cell_calib.py     # LSTM cell 校准数据
│   └── compile.sh                 # 8 个 axmodel 一键编译（Pulsar2 Docker）
├── python/                  # 板端 Python 推理（community1_sdk + example.py）
├── cpp/                     # C++ 推理源码（交叉编译，与 Python 逐行对齐）
├── scripts/                 # 主机参数导出（npz → bin，C++ SDK 用）
├── samples/                 # 演示音频（2 人会议 120 s）
└── run_ax650.sh             # 一键运行（板端，需先下载 HF 模型）
```

## 模型转换

```bash
# 依赖：torch + pyannote.audio 4.0.7 + onnxruntime + Pulsar2 Docker
cd model_convert
python export_sincnet_split.py && python export_lstm_cells.py && python export_models.py
python generate_calibration.py && python generate_cell_calib.py && bash pack_calib.sh
bash compile.sh
```

上游权重 `pyannote/speaker-diarization-community-1` 需接受 HF 用户条款（cc-by-4.0）。

## C++ 推理

`cpp/` 与 Python 版逐行对齐（板端 C++ vs Python 整场 DER 差 0.56%）。交叉编译：

```bash
bash cpp/download_toolchains.sh   # gcc-arm-9.2 + AX650 BSP SDK（一次性）
bash cpp/build_ax650.sh           # 产物 install/ax650/community1_diar
```

详见 `cpp/README.md`。

## 参考

- [pyannote-audio](https://github.com/pyannote/pyannote-audio)（MIT）
- [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1)（cc-by-4.0）
