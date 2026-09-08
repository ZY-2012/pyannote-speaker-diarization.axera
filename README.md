# pyannote community-1 Speaker Diarization on AX650N

`pyannote/speaker-diarization-community-1`（cc-by-4.0）在 AX650N 上的量化部署，
一句话把会议音频变成说话人分段（RTTM）。

## 快速开始（板端）

```bash
pip install numpy scipy scikit-learn soundfile axengine   # 板端依赖
bash run_ax650.sh                        # 跑自带样例 samples/sample_meeting.wav
bash run_ax650.sh your_16k_mono.wav out.rttm
```

输出 RTTM 每行一个说话人片段：

```
SPEAKER your_audio 1   6.865  15.627 <NA> <NA> SPEAKER_00 <NA> <NA>
```

可选参数（环境变量或 example.py 参数）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--step` / `STEP` | 2.0 | 分割滑动步长（秒）。1s 更细但慢 2.8×；2s 几乎无损（推荐） |
| `--num-speakers` | 0 | 0=自动估计；已知人数可固定（如 4） |
| `--threshold`/`--fa`/`--fb` | 0.6/0.07/0.8 | PLDA 聚类参数，一般无需改 |

## 指标（板端实测，帧级 DER）

| 数据集 | 板端量化 | GPU FP32 | 3D-Speaker+overlap(GPU) |
|---|---|---|---|
| AMI dev12（no collar） | **20.06%** | 20.08% | 22.25% |
| AliMeeting eval（±0.125s） | 20.88% | 18.84% | 18.51% |
| AliMeeting eval（±0.25s） | 17.26% | 15.16% | 13.99% |

- RTF（纯推理，不含模型加载）：**0.16**（step=2s）/ 0.44（step=1s）
- 单场 30 分钟会议：约 5 分钟（step=2s）
- 完整评测与量化细节见 `板端评测记录.md`；对比分析见
  `../pyannote_community1_vs_3D-Speaker_对比报告.md`

## 目录

```
models/         8 个 axmodel + 8 个主机侧参数 npz（model_meta.json 有清单）
python/         板端 SDK（community1_sdk/ + example.py）
model_convert/  可复现的导出/量化脚本（Pulsar2）
samples/        自带测试音频（2 人会议 120s，AliMeeting 片段）
run_ax650.sh    一键推理入口
```

## 部署架构（哪些在 NPU、哪些在 CPU）

| 组件 | 载体 | 原因 |
|---|---|---|
| sincnet 首层 conv | NPU（主机 im2col + MatMul） | 单通道 Conv NPU 算错 |
| sincnet 后续 conv/maxpool | NPU ×2 模型 | — |
| InstanceNorm / TSTP 池化 | 主机 FP32 | U16 量化域数值崩坏 |
| 4 层 BiLSTM | NPU ×4（64 步展开 cell，FP32 状态跨块） | Pulsar2 不支持 LSTM 算子 |
| FFN / powerset 解码 | 主机 | 小矩阵 |
| fbank / PLDA / AHC / VBx | 主机 numpy/scipy | 纯算法 |

## 模型转换复现

```bash
cd model_convert
python export_sincnet_split.py      # 分割前端 3 段 + host_frontend.npz
python export_lstm_cells.py         # LSTM cell ×4
python export_models.py             # embedding 卷积栈 + host_emb_tail.npz + fbank 参数
python generate_calibration.py && bash pack_calib.sh
bash compile.sh                     # 8 个 axmodel（需 Pulsar2 Docker）
```

详见 `model_convert/` 各脚本注释与 `README.md`（本目录上级工程的
`pyannote_community1.AXERA/板端评测记录.md` 记录了全部 NPU 坑）。

## C++ 推理

`cpp/` 提供与 Python 对齐的 C++ 实现（板端实测 C++ vs Python DER 差 0.56%，
RTF 0.09，OMP 8 线程）。交叉编译工具链下载见 `cpp/download_toolchains.sh`
（gcc-arm-9.2 + AX650 BSP SDK），或参考 `cpp/README.md`。

## 模型下载

量化模型（axmodel + 参数）与 C++ 可执行文件在
[HuggingFace: HY-2012/pyannote-speaker-diarization.axera](https://huggingface.co/HY-2012/pyannote-speaker-diarization.axera)，
本仓库只放源码（模型转换 + Python/C++ 推理）。

## 参考

- [pyannote-audio](https://github.com/pyannote/pyannote-audio)（MIT）
- [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1)（cc-by-4.0）
