# 测试集评测

README 指标表所用三个测试集的下载、参考处理与评分方法。评分口径：帧级 DER
（10 ms 网格 + 匈牙利映射，`score_der.py`），collar 为总宽度（0.25 = ±0.125 s）。

三个脚本的作用：

| 脚本 | 作用 |
|---|---|
| `run_board.sh` | 板端批量推理：遍历 wav 目录，逐个跑 C++ 二进制，输出 RTTM 目录（断点续跑，已存在的结果跳过） |
| `score_der.py` | 评分：ref/hyp 两个 RTTM 目录按文件名配对，逐场 + 加权输出 DER/miss/FA/conf |
| `prep_ami_ref.py` `prep_ali_ref.py` | 参考重建：官方标注（AMI words XML / AliMeeting TextGrid）转成评测用 RTTM，一次性处理，跑完即可删 |

完整评测 = 下载数据 → 重建参考（一次性）→ 板端推理出 RTTM → 评分。

### 参考重建脚本用法

```bash
# 查看参数
python benchmark/prep_ami_ref.py --help
python benchmark/prep_ali_ref.py --help

# AMI：--annot 指向官方 NXT words 标注目录，--out 输出参考 RTTM 目录
#   标注文件命名 {meeting}.{spk}.words.xml（如 ES2004a.A.words.xml）
python benchmark/prep_ami_ref.py --annot /path/to/annot_dir/words --out ref_dir --gap 0.3

# AliMeeting：--tg 指向 Eval_Ali_TextGrid 解压目录，--out 输出参考 RTTM 目录
#   TextGrid 命名 R8001_M8004.TextGrid（不带麦克风阵列后缀）
python benchmark/prep_ali_ref.py --tg /path/to/Eval_Ali_TextGrid --out ref_dir
```

## 1. AMI dev（16 场 Mix-Headset）

| 项 | 说明 |
|---|---|
| 下载 | [AMI 官网](https://groups.inf.ed.ac.uk/ami/corpus/)（需注册），下载 `MixHeadset` 音频与 NXT words 标注（`{meeting}.{spk}.words.xml`） |
| 参考 | `python prep_ami_ref.py --annot annot_dir/words --out ref_dir --gap 0.3`<br>把官方 NXT 逐词标注合并成说话人段 RTTM（词间空隙 ≤0.3 s 合并） |
| 音频 | Mix-Headset 16 kHz 单声道，无需处理 |
| 协议 | no collar，12 场（排除 ES2004b/IS1009b/TS3003b，TS3003d 无音频） |

```bash
# ① 板端推理：对每场音频跑 community1_diar，输出 RTTM 到 out_dir
bash benchmark/run_board.sh ami_audio_dir out_dir
# ② 评分：ref_dir 与 out_dir 按 meeting 名配对算 DER（collar=0 即 no collar）
python benchmark/score_der.py --ref ref_dir --hyp out_dir --collar 0.0
```

## 2. AliMeeting Eval（4 场远场）

| 项 | 说明 |
|---|---|
| 下载 | [OpenSLR 119](https://www.openslr.org/119/)：`Eval_Ali.tar.gz`（8 通道 16 kHz）与 `Eval_Ali_TextGrid.tar.gz` |
| 音频 | 取 ch0 降混为单声道（16 kHz，无需重采样） |
| 参考 | `python prep_ali_ref.py --tg Eval_Ali_TextGrid --out ref_dir`<br>把官方 TextGrid 说话人 tier 转成 RTTM（只保留有转写的区间） |
| 协议 | collar 0.25 / 0.5 两档；评测 4 场：R8001_M8004 / R8003_M8001 / R8007_M8010 / R8007_M8011 |

```bash
# ① 板端推理
bash benchmark/run_board.sh ali_audio_dir out_dir
# ② 评分：两档 collar 分别算（0.25 = ±0.125 s，0.5 = ±0.25 s）
python benchmark/score_der.py --ref ref_dir --hyp out_dir --collar 0.25
python benchmark/score_der.py --ref ref_dir --hyp out_dir --collar 0.5
```

## 3. VoxConverse（test 232 场）

| 项 | 说明 |
|---|---|
| 下载 | [官网 v0.3](https://www.robots.ox.ac.uk/~vgg/data/voxconverse/)：`voxconverse_test_wav.zip` + 仓库内 `test/*.rttm`（也可用 [HF 镜像](https://huggingface.co/datasets/diarizers-community/voxconverse) 的 parquet 抽取） |
| 参考 | 官方 `test/*.rttm`，无需处理 |
| 协议 | collar 0.25 |

```bash
# ① 板端推理
bash benchmark/run_board.sh vox_test_wav_dir out_dir
# ② 评分
python benchmark/score_der.py --ref vox_test_rttm_dir --hyp out_dir --collar 0.25
```
