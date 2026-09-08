# C++ 推理（community1_diar，AX650N）

与 `python/` 版逐行对齐的 C++ 实现；板端实测 **C++ vs Python 整场会议 DER 差 0.56%**，
RTF 0.12（step=2s，OMP 4 线程）vs Python 0.16。

## 编译

### 依赖（一次性，见 download_toolchains.sh）

- 交叉编译器：gcc-arm-9.2 aarch64-none-linux-gnu
- AX650 BSP SDK（msp/out，提供 ax_engine_api.h / libax_*.a）

### 一键构建

```bash
bash build_ax650.sh
# 产物：install/ax650/community1_diar
```

环境变量覆盖默认路径：

```bash
TOOLCHAIN_ROOT=/path/to/gcc-arm-9.2 BSP_MSP_DIR=/path/to/ax650n_bsp_sdk/msp/out bash build_ax650.sh
```

## 运行（板端）

```bash
export LD_LIBRARY_PATH=/soc/lib:$LD_LIBRARY_PATH
./community1_diar --model-dir models --wav in.wav --out out.rttm \
                  --step 2.0 --num-speakers 0
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model-dir` | models | axmodel + params（*.bin）目录 |
| `--wav` | 必填 | 16 kHz 单声道 wav |
| `--out` | output.rttm | RTTM 输出 |
| `--step` | 2.0 | 分割滑动步长（秒） |
| `--num-speakers` | 0 | 0=自动估计 |

`--num-speakers` 固定人数时用内置 KMeans（与 Python 的 sklearn KMeans 随机初始化
不同，DER 可能有 ~1pp 差异）；自动估计模式（默认）与 Python 完全对齐。

## 实现说明

- 模型参数（fbank/InstanceNorm/FFN/PLDA 等）由 `../scripts/export_params_bin.py`
  从 npz 导出为 `models/*.bin`，C++ 运行时加载；PLDA 的 eig 分解在导出期预计算，
  C++ 侧零 Eigen 依赖。
- 前端对分工具：`tools/dump_features.cpp`（宿主 g++ 可直接编译），dump fbank 与
  Python 版对比（实测 cosine 0.9999999）。
- axengine 封装见 `src/EngineWrapper_axera.cpp`：显式 VNPU=DISABLE 初始化、
  `AX_SYS_MemAllocCached` + 每次 RunSync 前后 `MflushCache`/`MinvalidateCache`
  （漏掉会导致同输入多次运行结果不一致，见板端评测记录）。

## 性能（AX650N，step=2s，OMP_NUM_THREADS=4）

| | RTF | 备注 |
|---|---|---|
| C++ | 0.12 | 30 分钟会议约 6 分钟 |
| Python | 0.16 | 对照 |

分阶段（每 10s 窗口）：LSTM cell 40 次 NPU 调用 ~110ms（串行链延迟为主）、
FFN 15ms、fbank ~10ms、其余 ~50ms。
