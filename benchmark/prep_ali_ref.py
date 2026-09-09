#!/usr/bin/env python3
"""AliMeeting 参考 RTTM 重建：官方 TextGrid 说话人 tier → RTTM（只保留有转写的区间）。

用法：
  python prep_ali_ref.py --tg Eval_Ali_TextGrid --out ref_dir

TextGrid 命名 R8001_M8004.TextGrid（不带麦克风阵列后缀）。
"""
import argparse
import re
from pathlib import Path

TG_PAT = re.compile(r'name = "([^"]+)"')


def parse_textgrid(path):
    tiers, cur = {}, None
    for line in path.read_text(errors='replace').splitlines():
        m = TG_PAT.search(line)
        if m and '_SPK' in m.group(1):
            cur = m.group(1)
            tiers[cur] = []
        elif cur:
            sline = line.strip()
            if sline.startswith('xmin = '):
                tiers[cur].append([float(sline[7:]), None, None])
            elif sline.startswith('xmax = '):
                tiers[cur][-1][1] = float(sline[7:])
            elif sline.startswith('text = '):
                tiers[cur][-1][2] = sline[7:].strip('"')
    return tiers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tg', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for tg_p in sorted(Path(args.tg).glob('*.TextGrid')):
        mid = '_'.join(tg_p.name.split('.')[0].split('_')[:2])  # R8001_M8004
        tiers = parse_textgrid(tg_p)
        lines = []
        for spk in sorted(tiers):
            for xmin, xmax, text in tiers[spk]:
                if not text or xmax <= xmin:
                    continue
                lines.append(f'SPEAKER {mid} 1 {xmin:.3f} {xmax - xmin:.3f} '
                             f'<NA> <NA> {mid}_{spk} <NA>')
        lines.sort(key=lambda l: float(l.split()[3]))
        (out / f'{mid}.rttm').write_text('\n'.join(lines) + '\n')
        print(f'{mid}: {len(lines)} segments')


if __name__ == '__main__':
    main()
