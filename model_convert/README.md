# 模型转换（Pulsar2 量化，四步复现）

① 导出 ONNX（自动对分验证）→ ② 生成校准数据 → ③ 打包 → ④ 量化编译。

```bash
# 依赖：torch + pyannote.audio 4.0.7 + onnxruntime + Pulsar2 Docker 镜像
export PULSAR2_IMAGE=docker-registry.aitsw.axera-tech.com/pulsar2:20260810-temp-09cadfa9

python export_sincnet_split.py     # 分割前端 3 段 ONNX + host_frontend.npz
python export_lstm_cells.py        # LSTM cell ×4 ONNX
python export_models.py            # embedding 卷积栈 ONNX + host_emb_tail.npz + fbank 参数
python generate_calibration.py     # 真实会议校准数据
bash pack_calib.sh
bash compile.sh                    # 8 个 axmodel → compile/
```

上游权重 `pyannote/speaker-diarization-community-1`（cc-by-4.0，需接受 HF 用户条款），
路径用 `COMMUNITY1_SNAPSHOT` 环境变量指定（默认 HF 缓存）。

量化架构与全部 NPU 坑见上级工程 `板端评测记录.md`：
C=1 卷积 NPU 算错 → im2col+MatMul；InstanceNorm/TSTP 在 U16 域崩坏 → 主机 FP32；
Pulsar2 不支持 LSTM → 64 步展开 cell + FP32 状态跨块（c 状态 clamp ±8）；
check=2 容差过松必须上板对拍。
