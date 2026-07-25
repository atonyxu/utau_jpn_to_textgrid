import argparse
import glob
import os
import shutil
import tempfile

from hifigan import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description='对数据集子文件夹内的 wav 进行变调, 并同步复制/重命名对应 TextGrid'
    )
    parser.add_argument('-i', '--input', required=True,
                        help='数据集子文件夹路径 (须包含 wav/ 和 TextGrid/)')
    parser.add_argument('-s', '--shift', nargs='+', type=float, required=True,
                        help='变调半音数, 如 +4 -3 +2')
    parser.add_argument('-g', '--glob', default='*.wav',
                        help='wav 文件 glob 模式 (相对于 wav/ 文件夹, 默认 *.wav)')
    parser.add_argument('-d', '--device', default='auto',
                        help='设备: auto / cuda / cpu (默认 auto)')
    return parser.parse_args()


def main():
    args = parse_args()

    wav_dir = os.path.join(args.input, 'wav')
    tg_dir = os.path.join(args.input, 'TextGrid')
    if not os.path.isdir(wav_dir):
        raise FileNotFoundError(f'wav 目录不存在: {wav_dir}')
    if not os.path.isdir(tg_dir):
        raise FileNotFoundError(f'TextGrid 目录不存在: {tg_dir}')

    # 扫描匹配的 wav 文件
    pattern = os.path.join(wav_dir, args.glob)
    matched = sorted(glob.glob(pattern))
    if not matched:
        print(f'未找到匹配 {args.glob} 的 wav 文件')
        return

    tags = [f'{s:+g}' for s in args.shift]
    print(f'匹配 {len(matched)} 个 wav 文件, 变调: {tags}')
    print(f'预计输出 {len(matched) * len(args.shift)} 个 wav + {len(matched) * len(args.shift)} 个 TextGrid\n')

    # 临时目录: 复制匹配的 wav 进去, 供 run_pipeline 处理
    tmp_dir = tempfile.mkdtemp(prefix='pitchshift_')
    try:
        for fp in matched:
            shutil.copy2(fp, tmp_dir)

        # 变调
        run_pipeline(
            input_path=tmp_dir,
            output_dir=tmp_dir,
            shift_semitones_list=args.shift,
            device=args.device,
            quiet=False,
        )

        # 移动变调后的 wav 到源目录
        moved = 0
        for f in os.listdir(tmp_dir):
            if not f.lower().endswith('.wav'):
                continue
            src = os.path.join(tmp_dir, f)
            dst = os.path.join(wav_dir, f)
            shutil.move(src, dst)
            moved += 1
        print(f'\n移动 {moved} 个 wav 到 {wav_dir}')

        # 复制 TextGrid
        tg_copied = 0
        tg_missing = 0
        for fp in matched:
            base = os.path.splitext(os.path.basename(fp))[0]
            src_tg = os.path.join(tg_dir, f'{base}.TextGrid')
            if not os.path.exists(src_tg):
                print(f'  [跳过] TextGrid 不存在: {src_tg}')
                tg_missing += 1
                continue
            for tag in tags:
                dst_tg = os.path.join(tg_dir, f'{base}_{tag}.TextGrid')
                shutil.copy2(src_tg, dst_tg)
                tg_copied += 1
        print(f'复制 {tg_copied} 个 TextGrid 到 {tg_dir}')
        if tg_missing:
            print(f'  ({tg_missing} 个 TextGrid 缺失, 已跳过)')

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print('\n完成!')


if __name__ == '__main__':
    main()
