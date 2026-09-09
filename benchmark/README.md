# 测试集评测

README 指标表所用三个测试集的下载、参考处理与评分方法。评分口径：帧级 DER
（10 ms 网格 + 匈牙利映射，`score_der.py`），collar 为总宽度（0.25 = ±0.125 s）。

## 1. AMI dev（16 场 Mix-Headset）

| 项 | 说明 |
|---|---|
| 下载 | [AMI 官网](https://groups.inf.ed.ac.uk/ami/corpus/)（需注册），下载 `MixHeadset` 音频与 NXT words 标注（`{meeting}.{spk}.words.xml`） |
| 参考 | `python prep_ami_ref.py --annot annot_dir/words --out ref_dir --gap 0.3` |
| 音频 | Mix-Headset 16 kHz 单声道，无需处理 |
| 协议 | no collar，12 场（排除 ES2004b/IS1009b/TS3003b，TS3003d 无音频） |

```bash
bash benchmark/run_board.sh ami_audio_dir out_dir          # 板端推理
python benchmark/score_der.py --ref ref_dir --hyp out_dir --collar 0.0
```

## 2. AliMeeting Eval（4 场远场）

| 项 | 说明 |
|---|---|
| 下载 | [OpenSLR 119](https://www.openslr.org/119/)：`Eval_Ali.tar.gz`（8 通道 16 kHz）与 `Eval_Ali_TextGrid.tar.gz` |
| 音频 | 取 ch0 降混为单声道（16 kHz，无需重采样） |
| 参考 | `python prep_ali_ref.py --tg Eval_Ali_TextGrid --out ref_dir` |
| 协议 | collar 0.25 / 0.5 两档；评测 4 场：R8001_M8004 / R8003_M8001 / R8007_M8010 / R8007_M8011 |

```bash
bash benchmark/run_board.sh ali_audio_dir out_dir
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
bash benchmark/run_board.sh vox_test_wav_dir out_dir
python benchmark/score_der.py --ref vox_test_rttm_dir --hyp out_dir --collar 0.25
```
