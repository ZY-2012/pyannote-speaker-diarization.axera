#!/usr/bin/env python3
"""AMI 参考 RTTM 重建：官方 NXT words 级标注 → RTTM。

段级标注含句内长停顿（参考会"变胖"）；words 级逐词时间戳更贴近真实语音。
相邻词间隔 <= --gap 的合并为一段。

用法：
  python prep_ami_ref.py --annot annot_dir/words --out ref_dir [--gap 0.3]

标注文件命名 {meeting}.{spk}.words.xml（如 ES2004a.A.words.xml）。
"""
import argparse
import re
from pathlib import Path

W_PAT = re.compile(r'starttime="([\d.]+)"\s+endtime="([\d.]+)"')


def words_of(path):
    out = []
    for line in path.read_text(encoding='latin-1', errors='replace').splitlines():
        m = W_PAT.search(line)
        if m:
            st, en = float(m.group(1)), float(m.group(2))
            if en > st:
                out.append((st, en))
    return sorted(out)


def merge(spans, gap):
    if not spans:
        return []
    out = [list(spans[0])]
    for st, en in spans[1:]:
        if st - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], en)
        else:
            out.append([st, en])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annot', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gap', type=float, default=0.3)
    args = ap.parse_args()
    annot = Path(args.annot)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meetings = sorted({p.name.split('.')[0] for p in annot.glob('*.words.xml')})
    for mid in meetings:
        xmls = sorted(annot.glob(f'{mid}.*.words.xml'))
        lines = []
        for x in xmls:
            spk = x.name.split('.')[1]
            for st, en in merge(words_of(x), args.gap):
                lines.append(f'SPEAKER {mid} 1 {st:.3f} {en - st:.3f} '
                             f'<NA> <NA> {mid}_{spk} <NA>')
        lines.sort(key=lambda l: float(l.split()[3]))
        (out / f'{mid}.rttm').write_text('\n'.join(lines) + '\n')
        print(f'{mid}: {len(xmls)} speakers, {len(lines)} segments')


if __name__ == '__main__':
    main()
