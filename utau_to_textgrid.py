#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UTAU VCV (Japanese) 标注 -> Praat TextGrid 转换工具。

输入: 一个 UTAU 歌声音源目录 (内含若干音高子目录, 每个子目录有 oto.ini + wav 样本)
输出:
  <out>/wav/NNNN.wav          (重命名后的音频)
  <out>/TextGrid/NNNN.TextGrid(同名音素标注)

说明:
  - 从 oto.ini 可获得每个音节的 [辅音->元音] 分界点 (preutterance = left + preu)。
  - 辅音起始位置 oto.ini 没有直接给出, 这里通过音频短时能量检测得到:
      * 塞音/塞擦音 (k,t,p,b,d,g,ch,ts,j,z 等): 向前找闭塞段 (静音) 起点
      * 擦音 (s,sh,h,f 等): 找噪声能量上升点
      * 响音 (鼻音/流音/半元音 n,m,r,w,y 等): 取两元音间能量谷底
  - 假名 -> 罗马音 -> 音素 转换基于 jpn-phoneset 目录下的字典文件。

依赖: numpy, textgrid (Praat TextGrid 读写), wav 读取用标准库 wave
"""

import argparse
import os
import sys
import wave
from pathlib import Path

import numpy as np
import textgrid as tglib

HERE = Path(__file__).resolve().parent
PHONESET_DIR = HERE / "jpn-phoneset"


# =========================================================================
# 1. 假名 -> 罗马音 -> 音素 映射
# =========================================================================

# 小假名 (含浊音/半浊音小字)
SMALL_Y = {"ゃ": "ya", "ゅ": "yu", "ょ": "yo", "ャ": "ya", "ュ": "yu", "ョ": "yo"}
SMALL_V = {"ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
           "ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o"}


_PITCH_OFFSETS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_pitch_to_midi(dirname):
    """从音高目录名/后缀解析 MIDI 编号。
    支持形如 'A3', 'C4', 'D4-N', 'F#4', 'Gb3' 的目录名或 alias 后缀。
    返回 MIDI 整数 (C4=60, A4=69); 无法解析时返回 None。"""
    import re
    m = re.match(r"^([A-Ga-g])([#b]?)(\d+)", dirname)
    if not m:
        return None
    letter = m.group(1).upper()
    accidental = m.group(2)
    octave = int(m.group(3))
    offset = _PITCH_OFFSETS[letter]
    if accidental == "#":
        offset += 1
    elif accidental == "b":
        offset -= 1
    return 12 * (octave + 1) + offset


def detect_pitch_from_alias(alias):
    """从 VCV alias 的音高后缀提取 MIDI 编号。
    如 'a か_A3' -> 57, 'a か_D4-N' -> 62。
    无后缀或无法解析时返回 None。"""
    _, _, suffix = alias_parts(alias)
    if not suffix:
        return None
    return parse_pitch_to_midi(suffix)


def parse_prefix_map(path):
    """解析 UTAU prefix.map。返回 {subdir_name: midi} 映射。
    prefix.map 格式: 'B3\\t\\t_A3' 表示 B3 音高用 _A3 子目录。
    取每个子目录映射的最大音高作为代表音高 (通常是最接近目录名的音高)。"""
    import re
    if not Path(path).exists():
        return {}
    # 先收集每个子目录对应的所有音高
    dir_pitches = {}
    # prefix.map 是纯 ASCII (音名), 编码无所谓
    enc = "shift_jis"
    try:
        with open(path, "r", encoding=enc, errors="replace") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                note = parts[0].strip()
                subdir = parts[2].strip().lstrip("_")
                if not subdir or not note:
                    continue
                midi = parse_pitch_to_midi(note)
                if midi is None:
                    continue
                dir_pitches.setdefault(subdir, []).append(midi)
    except Exception:
        pass
    # 每个子目录的代表音高 = 映射区间的中位数
    result = {}
    for subdir, midis in dir_pitches.items():
        midis.sort()
        result[subdir] = midis[len(midis) // 2]
    return result


def detect_f0_midi(samples, sr, fmin=70.0, fmax=500.0):
    """用自相关法检测音频平均基频, 转换为 MIDI 编号。
    仅对浊音段(能量高于阈值)做检测, 取中位数 F0。
    返回 MIDI 整数; 检测失败返回 None。"""
    # 取能量包络, 找浊音段
    times, energy = energy_envelope(samples, sr)
    if len(energy) == 0:
        return None
    # 浊音阈值: 能量 > 最大能量的 20%
    e_max = energy.max()
    if e_max < 1e-6:
        return None
    voiced_mask = energy > e_max * 0.2
    if voiced_mask.sum() < 10:
        return None
    # 取浊音段的音频
    hop_ms = times[1] - times[0] if len(times) > 1 else 5.0
    hop_samples = int(hop_ms * sr / 1000.0)
    frame_len = int(0.04 * sr)  # 40ms 窗
    if frame_len >= len(samples):
        frame_len = len(samples) // 2
    f0_list = []
    lag_min = int(sr / fmax)
    lag_max = int(sr / fmin)
    lag_max = min(lag_max, frame_len - 1)
    if lag_max <= lag_min:
        return None
    # 在浊音段做自相关 F0 检测
    n_frames = (len(samples) - frame_len) // hop_samples + 1
    for i in range(0, n_frames, max(1, n_frames // 200)):  # 最多取 200 帧
        start = i * hop_samples
        end = start + frame_len
        if end > len(samples):
            break
        # 检查该帧是否在浊音段
        t_ms = start / sr * 1000.0
        idx = int(t_ms / hop_ms) if hop_ms > 0 else 0
        if idx >= len(voiced_mask) or not voiced_mask[idx]:
            continue
        frame = samples[start:end].astype(np.float64)
        frame = frame - frame.mean()
        e_frame = np.sqrt(np.mean(frame * frame))
        if e_frame < 1e-4:
            continue
        # 归一化自相关
        acf = np.correlate(frame, frame, mode='full')
        acf = acf[frame_len - 1:]  # 取右半
        acf = acf / (acf[0] + 1e-12)
        # 在 [lag_min, lag_max] 范围内找最大峰
        if lag_max >= len(acf):
            lag_max = len(acf) - 1
        seg = acf[lag_min:lag_max + 1]
        if len(seg) == 0:
            continue
        peak = np.argmax(seg) + lag_min
        if acf[peak] > 0.3:  # 峰值阈值
            f0 = sr / peak
            if fmin <= f0 <= fmax:
                f0_list.append(f0)
    if len(f0_list) < 5:
        return None
    f0_median = np.median(f0_list)
    # F0 -> MIDI: midi = 69 + 12 * log2(f0 / 440)
    midi = round(69 + 12 * np.log2(f0_median / 440.0))
    return int(midi)


def _load_simple_dict(path):
    """加载 形如 'あ a' 的 简单 字典文件。"""
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                d[parts[0]] = parts[1]
    return d


def _load_romaji_phones(path):
    """加载 romaji -> phones 字典 (如 'ka k a', 'kya ky a')。
    返回 {romaji: [phone, phone, ...]}。"""
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                romaji = parts[0]
                phones = parts[1:]
                d[romaji] = phones
    return d


# i 段假名 -> 拗音辅音 (ki->ky, shi->sh, chi->ch, ti->ty ...)
PALATAL_STEMS = {
    "ki": "ky", "gi": "gy", "ni": "ny", "hi": "hy", "mi": "my", "ri": "ry",
    "bi": "by", "pi": "py", "fi": "fy", "vi": "vy",
    "shi": "sh", "chi": "ch", "ji": "j",
    "ti": "ty", "di": "dy",
}


def _compute_compound_romaji(base_rom, small_phone):
    """根据基础罗马音 + 小假名代表的音, 推算复合罗马音。
    base_rom 例如 'ki','shi','ku','u','fu','tsu','chi','vu','te','to'
    small_phone ∈ {'a','i','u','e','o','ya','yu','yo'}"""
    if small_phone in ("ya", "yu", "yo"):
        sv = small_phone[1]  # a/u/o
        # i 段 + 小やゆよ -> 拗音
        if base_rom in PALATAL_STEMS:
            return PALATAL_STEMS[base_rom] + sv
        # て/で + 小やゆよ -> ty/dy
        if base_rom in ("te", "de"):
            return {"te": "ty", "de": "dy"}[base_rom] + sv
        # ふ(fu)/ヴ(vu) + 小やゆよ -> fy/vy
        if base_rom in ("fu", "vu"):
            return ("fy" if base_rom == "fu" else "vy") + sv
        if base_rom == "i":
            return "y" + sv
        if base_rom == "e":
            return "ye" if sv == "e" else None
        return None
    else:  # 小元音 a/i/u/e/o
        sv = small_phone
        # i 段 + 小元音 -> 拗音 (きぇ->kye, しぇ->she, て+ぃ->ti 除外)
        if base_rom in PALATAL_STEMS:
            return PALATAL_STEMS[base_rom] + sv
        if base_rom in ("te", "de"):
            # て+ぃ=ti, て+ぇ=tye, て+ぅ=tu, て+ぁ=ta, て+ぉ=to
            if sv == "e":
                return {"te": "tye", "de": "dye"}[base_rom]
            return {"te": "t", "de": "d"}[base_rom] + sv
        if base_rom in ("to", "do"):
            # と+ぅ=tu, ど+ぅ=du
            return {"to": "t", "do": "d"}[base_rom] + sv
        if base_rom == "u":
            return {"a": "wa", "i": "wi", "u": "wu", "e": "we", "o": "wo"}.get(sv)
        if base_rom == "i":
            return {"e": "ye"}.get(sv)
        if base_rom == "tsu":
            return "ts" + sv
        if base_rom == "fu":
            return "f" + sv
        if base_rom in ("ku", "gu"):
            return {"ku": "kw", "gu": "gw"}[base_rom] + sv
        if base_rom == "vu":
            return "v" + sv
        if base_rom == "su":
            return "s" + sv
        if base_rom == "zu":
            return "z" + sv
        if base_rom in ("ne", "he", "me", "re", "se", "ke", "ge", "be", "pe"):
            return base_rom[:-1] + sv
        return None


def build_kana_romaji_map(hira_dict, kata_dict):
    """构建 假名->罗马音 字典, 含复合拗音。"""
    result = {}
    result.update(hira_dict)
    result.update(kata_dict)

    bases = dict(hira_dict)
    bases.update(kata_dict)
    for base_kana, base_rom in list(bases.items()):
        for small_set in (SMALL_Y, SMALL_V):
            for small_kana, small_phone in small_set.items():
                compound = base_kana + small_kana
                if compound in result:
                    continue
                rom = _compute_compound_romaji(base_rom, small_phone)
                if rom is not None:
                    result[compound] = rom
    return result


# 音素分类 (用于辅音起始检测策略)
# 依据 jpn-phoneset/japanese-romaji-phones.txt 中辅音音素
STOP_LIKE = {"k", "g", "t", "d", "p", "b",
             "kw", "gw", "ky", "gy", "ty", "dy", "py", "by",
             "ch", "ts", "j", "z"}  # 塞音 + 塞擦音 (含闭塞段)
FRICATIVE = {"s", "sh", "h", "hy", "f", "fy"}  # 擦音 (连续噪声, 无闭塞)
SONORANT = {"n", "m", "ny", "my", "r", "ry", "w", "y", "v"}  # 响音 (类元音)


def consonant_category(consonant):
    if consonant in STOP_LIKE:
        return "stop"
    if consonant in FRICATIVE:
        return "fricative"
    if consonant in SONORANT:
        return "sonorant"
    return "other"


# 典型辅音时长 (ms), 用于约束搜索范围与回退默认值
CONSONANT_MAX_MS = {
    "stop": 220, "fricative": 220, "sonorant": 160, "other": 160,
}
CONSONANT_DEFAULT_MS = {
    "stop": 90, "fricative": 110, "sonorant": 70, "other": 80,
}


# 已知辅音音素前缀 (长前缀优先匹配), 用于回退推导
_CONSONANT_PREFIXES = sorted(
    ["cl", "ts", "sh", "ch", "ry", "ky", "py", "dy", "ty", "ny", "hy", "my",
     "gy", "by", "kw", "gw", "fy", "ng", "ngy", "dz",
     "k", "g", "t", "d", "s", "z", "h", "b", "p", "f", "m", "n", "r", "w",
     "v", "y", "j"],
    key=len, reverse=True,
)


def _derive_phones(romaji):
    """当罗马音不在字典中时, 尝试拆成 (辅音, 元音)。"""
    if not romaji or romaji[-1] not in "aeiou":
        return None
    vowel = romaji[-1]
    stem = romaji[:-1]
    if stem in _CONSONANT_PREFIXES:
        return (stem, vowel)
    return None


def kana_to_phones(kana, kana_romaji, romaji_phones):
    """假名 -> 音素列表。返回 (consonant_or_None, vowel)。
    例: 'か' -> ('k','a'); 'あ' -> (None,'a'); 'きゃ' -> ('ky','a')。"""
    if kana in ("ん", "ン"):
        return (None, "N")
    romaji = kana_romaji.get(kana)
    if romaji is None:
        return None
    phones = romaji_phones.get(romaji)
    if phones is None:
        # 回退: 按辅音前缀拆分 (如 'fyu' -> ('fy','u'))
        return _derive_phones(romaji)
    if len(phones) == 1:
        return (None, phones[0])
    if len(phones) >= 2:
        # 形如 'k a' -> 辅音 'k', 元音 'a'; 'cl k a' -> 促音, 取后两个
        if phones[0] == "cl":
            return ("cl", phones[2]) if len(phones) >= 3 else (None, phones[-1])
        return (phones[0], phones[-1])
    return None


# =========================================================================
# 2. oto.ini 解析
# =========================================================================

class OtoEntry:
    __slots__ = ("sample", "alias", "left", "fixed", "right", "preu", "ovl")

    def __init__(self, sample, alias, left, fixed, right, preu, ovl):
        self.sample = sample
        self.alias = alias
        self.left = left
        self.fixed = fixed
        self.right = right
        self.preu = preu
        self.ovl = ovl

    @property
    def vowel_start(self):
        """辅音->元音 分界点 (绝对, ms)。"""
        return self.left + self.preu

    @property
    def region_end(self):
        """该条目可用区域右边界 (绝对, ms)。"""
        if self.right < 0:
            return self.left - self.right  # left - (负值) = left + |right|
        # right 为正: 距音频末尾的距离, 无法在此确定, 用 left+preu+fixed 近似
        return self.left + self.preu + self.fixed


def detect_oto_encoding(oto_path):
    """探测 oto.ini 的编码。UTAU 常见编码: Shift-JIS / UTF-8 / EUC-JP。
    用 errors='replace' 解码全文, 统计替换字符 (U+FFFD) 数量, 选最少的。
    这比 strict 解码更鲁棒: 个别坏字节不会让整个编码判定失败。"""
    with open(oto_path, "rb") as f:
        raw = f.read()
    best_enc = "shift_jis"
    best_repl = None
    for enc in ("utf-8", "euc_jp", "shift_jis"):
        try:
            decoded = raw.decode(enc, errors="replace")
        except Exception:
            continue
        repl = decoded.count("\ufffd")
        # 优先选替换字符最少的; 相同时按 utf-8 > euc_jp > shift_jis 顺序
        if best_repl is None or repl < best_repl:
            best_repl = repl
            best_enc = enc
    return best_enc


def parse_oto(oto_path, encoding=None):
    """解析 oto.ini。编码自动探测 (Shift-JIS / UTF-8 / EUC-JP)。返回 OtoEntry 列表。"""
    if encoding is None:
        encoding = detect_oto_encoding(oto_path)
    entries = []
    with open(oto_path, "r", encoding=encoding, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            if "=" not in line:
                continue
            sample, rest = line.split("=", 1)
            sample = sample.strip()
            fields = rest.split(",")
            if len(fields) < 6:
                continue
            alias = fields[0].strip()
            try:
                left = float(fields[1])
                fixed = float(fields[2])
                right = float(fields[3])
                preu = float(fields[4])
                ovl = float(fields[5])
            except ValueError:
                continue
            entries.append(OtoEntry(sample, alias, left, fixed, right, preu, ovl))
    return entries


def alias_parts(alias):
    """拆分 VCV 别名 'a か_A3' -> ('a', 'か', 'A3')。
    前置音(可为 '-'), 当前假名, 音高后缀。"""
    # 去掉音高后缀 (最后一个 _ 之后)
    if "_" in alias:
        core, suffix = alias.rsplit("_", 1)
    else:
        core, suffix = alias, ""
    if " " in core:
        prev, kana = core.split(" ", 1)
    else:
        prev, kana = "-", core
    return prev, kana, suffix


# =========================================================================
# 3. WAV 读取 + 能量包络
# =========================================================================

def read_wav(path):
    """读取 wav, 返回 (samples_float64, sample_rate)。"""
    with wave.open(str(path), "rb") as w:
        nch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        data = (data - 128.0) / 128.0
    elif sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sw}")
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    return data, sr


def energy_envelope(samples, sr, frame_ms=10.0, hop_ms=5.0):
    """短时 RMS 能量。返回 (times_ms, energy)。"""
    frame = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    if frame < 2:
        frame = 2
    if hop < 1:
        hop = 1
    n = len(samples)
    if n < frame:
        return np.array([0.0]), np.array([0.0])
    n_frames = 1 + (n - frame) // hop
    times = np.arange(n_frames) * hop * 1000.0 / sr + frame_ms / 2.0
    energy = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = samples[i * hop:i * hop + frame]
        energy[i] = np.sqrt(np.mean(seg * seg)) if len(seg) > 0 else 0.0
    return times, energy


def _interp_energy(times, energy, t_ms):
    """在能量包络上线性插值取 t_ms 处的值。"""
    if len(times) == 0:
        return 0.0
    if t_ms <= times[0]:
        return energy[0]
    if t_ms >= times[-1]:
        return energy[-1]
    idx = np.searchsorted(times, t_ms)
    t0, t1 = times[idx - 1], times[idx]
    e0, e1 = energy[idx - 1], energy[idx]
    if t1 == t0:
        return e0
    return e0 + (e1 - e0) * (t_ms - t0) / (t1 - t0)


# =========================================================================
# 4. 辅音起始检测
# =========================================================================

def detect_consonant_start(times, energy, vowel_start_ms, prev_boundary_ms,
                           consonant, audio_start_ms=0.0):
    """检测辅音起始 (绝对 ms)。
    vowel_start_ms: 当前元音起始 (preutterance)。
    prev_boundary_ms: 前一个边界 (前一元音起始, 或 0)。
    返回辅音起始 ms (若无辅音返回 vowel_start_ms)。
    """
    if consonant is None:
        return vowel_start_ms

    cat = consonant_category(consonant)
    max_dur = CONSONANT_MAX_MS[cat]
    default_dur = CONSONANT_DEFAULT_MS[cat]

    # 参考元音能量: 取 vowel_start 后 40ms 中位数
    vs_idx = np.searchsorted(times, vowel_start_ms)
    end_idx = np.searchsorted(times, vowel_start_ms + 40.0)
    end_idx = max(end_idx, vs_idx + 1)
    vowel_ref = np.median(energy[vs_idx:end_idx]) if end_idx > vs_idx else 0.0
    if vowel_ref <= 1e-6:
        vowel_ref = np.median(energy[energy > 0]) if np.any(energy > 0) else 1e-6

    # 搜索窗口
    search_lo = max(prev_boundary_ms + 30.0, vowel_start_ms - max_dur, audio_start_ms)
    search_hi = vowel_start_ms
    lo_idx = np.searchsorted(times, search_lo, side="left")
    hi_idx = np.searchsorted(times, search_hi, side="right")
    if hi_idx <= lo_idx:
        return max(search_lo, vowel_start_ms - default_dur)

    win_t = times[lo_idx:hi_idx]
    win_e = energy[lo_idx:hi_idx]

    if cat == "stop":
        # 塞音/塞擦音: 先跳过爆破段 (vowel_start 前 ~12ms), 再向前找静音(闭塞)起点
        burst_lo = vowel_start_ms - 12.0
        b_lo_idx = np.searchsorted(times, max(search_lo, burst_lo), side="left")
        seg = energy[b_lo_idx:hi_idx]
        if len(seg) > 0:
            thr = max(0.10 * vowel_ref, 1e-5)
            # 从 burst_lo 向前找连续低能量段的起点
            sil_mask = seg < thr
            if np.any(sil_mask):
                # 找最长的连续 True 段的起始
                best_start = None
                cur_start = None
                best_len = 0
                for i, m in enumerate(sil_mask):
                    if m:
                        if cur_start is None:
                            cur_start = i
                    else:
                        if cur_start is not None:
                            length = i - cur_start
                            if length > best_len:
                                best_len = length
                                best_start = cur_start
                            cur_start = None
                if cur_start is not None:
                    length = len(sil_mask) - cur_start
                    if length > best_len:
                        best_len = length
                        best_start = cur_start
                if best_start is not None and best_len >= 2:
                    return float(times[b_lo_idx + best_start])
        # 回退: 取窗口内能量最小值
        min_idx = lo_idx + int(np.argmin(win_e))
        return float(times[min_idx])

    elif cat == "fricative":
        # 擦音: 找能量从低升高的位置 (向前扫描, 首次超过阈值)
        thr = 0.30 * vowel_ref
        above = win_e >= thr
        if np.any(above):
            first = np.argmax(above)
            # 略向前回退到能量明显低于阈值的点, 作为擦音起点
            j = first
            while j > 0 and win_e[j] > 0.5 * thr:
                j -= 1
            return float(win_t[j])
        min_idx = lo_idx + int(np.argmin(win_e))
        return float(times[min_idx])

    else:  # sonorant / other: 取能量谷底
        min_idx = lo_idx + int(np.argmin(win_e))
        valley = float(times[min_idx])
        # 若谷底过于接近元音起点 (无明显下降), 用默认时长
        if vowel_start_ms - valley < 15.0:
            return max(search_lo, vowel_start_ms - default_dur)
        return valley


def detect_sound_end(times, energy, vowel_start_ms, duration_ms,
                     tail_margin_ms=30.0, min_silence_ms=60.0):
    """检测语音实际结束位置 (绝对 ms)。

    用于尾部: oto.ini 的 region_end 通常落在元音持续段内, 之后还有元音尾音,
    真正静音在更靠后。这里以最后元音的参考能量为基准, 从音频末尾向前扫描,
    找到首个超过阈值的帧, 即为发音结束点。

    vowel_start_ms: 最后一个元音的起始, 用于估计参考能量。
    tail_margin_ms: 在最后高能量帧之后保留的尾音余量。
    min_silence_ms: 末尾至少需要这么长的低能量段才认为语音已结束。
    """
    if len(times) == 0:
        return duration_ms
    # 参考能量: 取 vowel_start 后 60ms 中位数
    s_idx = np.searchsorted(times, vowel_start_ms)
    e_idx = np.searchsorted(times, vowel_start_ms + 60.0)
    e_idx = max(e_idx, s_idx + 1)
    vowel_ref = np.median(energy[s_idx:e_idx]) if e_idx > s_idx else 0.0
    if vowel_ref <= 1e-6:
        vowel_ref = np.median(energy[energy > 0]) if np.any(energy > 0) else 1e-6
    thr = max(0.12 * vowel_ref, 1e-5)

    end_idx = np.searchsorted(times, duration_ms, side="right")
    end_idx = min(end_idx, len(energy))
    # 从末尾向前找首个超过阈值的帧
    for i in range(end_idx - 1, -1, -1):
        if energy[i] >= thr:
            sound_end = float(times[i]) + tail_margin_ms
            # 必须给末尾留出 min_silence_ms 的静音, 否则尾音延伸到音频末尾
            if sound_end > duration_ms - min_silence_ms:
                sound_end = max(duration_ms - min_silence_ms,
                                float(times[i]) + tail_margin_ms * 0.5)
            return max(vowel_start_ms + 30.0, min(sound_end, duration_ms))
    # 整段都低于阈值: 直接返回中点之前
    return max(vowel_start_ms + 30.0, duration_ms - min_silence_ms)


def extend_first_consonant_start(times, energy, old_cs, vowel_start):
    """对第一个辅音: 往前找静音结束位置, 让辅音起始更靠前。

    当前 old_cs 可能太靠后, 把辅音的起始部分(尤其擦音/响音的低能量起始)
    划进了 sil。这里以 [0, old_cs] 区间内能量最低的 20% 分位数作为静音参考,
    从 old_cs 往前扫, 只要能量还明显高于静音, 就继续往前, 直到能量落入
    静音水平 — 那个位置就是真正的辅音起始。
    """
    if old_cs <= 50.0:
        return old_cs
    cs_idx = np.searchsorted(times, old_cs)
    if cs_idx <= 2:
        return old_cs
    # 静音参考: [0, old_cs] 内能量最低的 20% 分位数
    sil_ref = float(np.percentile(energy[:cs_idx], 20))
    if sil_ref < 1e-7:
        sil_ref = 1e-7
    thr = sil_ref * 2.0 + 1e-5  # 略高于静音

    # 从 old_cs 往前找能量降到 thr 以下的位置
    j = cs_idx - 1
    while j > 0 and energy[j] >= thr:
        j -= 1
    new_cs = float(times[j]) if j > 0 else 0.0
    # 约束: 至少留 30ms sil, 不超过 old_cs, 不超过 vowel_start - 10
    new_cs = max(30.0, min(new_cs, old_cs))
    new_cs = min(new_cs, vowel_start - 10.0)
    return new_cs


# 仅对这几个擦音/擦音类辅音做精修: 旧辅音起始位置可能过早切到上个元音尾部
REFINE_CONSONANTS = {"s", "sh", "f", "h", "fy", "hy"}


def refine_consonant_start(times, energy, old_start, vowel_start, consonant,
                           min_segment_ms=15.0):
    """在 [old_start, 四分之三位置] 内找能量最低点, 作为新辅音起始。

    仅用于 s, sh, f, h 等擦音: 旧辅音起始可能过早, 切到了上个元音尾部;
    这类辅音前的元音尾→辅音过渡通常伴随一个能量低谷。

    四分之三位置 = old_start + 0.75 * (vowel_start - old_start)
    在该范围内取能量最低的帧, 设为新辅音起始。
    """
    if consonant not in REFINE_CONSONANTS:
        return old_start
    quarter_pos = old_start + 0.6 * (vowel_start - old_start)
    lo_idx = np.searchsorted(times, old_start, side="left")
    hi_idx = np.searchsorted(times, quarter_pos, side="right")
    min_seg = max(2, int(min_segment_ms / 5.0))  # hop=5ms
    if hi_idx - lo_idx < min_seg + 1:
        return old_start

    seg_e = energy[lo_idx:hi_idx]
    seg_t = times[lo_idx:hi_idx]
    # 取能量最低点
    min_i = int(np.argmin(seg_e))
    new_start = float(seg_t[min_i])
    # 约束: 新起始必须在 [old_start, vowel_start - 10] 之间
    new_start = max(old_start, min(new_start, vowel_start - 10.0))
    return new_start


# =========================================================================
# 5. TextGrid 写入
# =========================================================================

def write_textgrid(path, intervals, xmax):
    """用官方 textgrid 库写入 Praat TextGrid (长格式)。
    intervals: [(t_start, t_end, text), ...] 须有序。
    输入时间单位为 ms, Praat TextGrid 使用秒, 写入时转换。

    使用 strict=True (默认), 库会强制区间连续 (无间隙、无重叠),
    若存在间隙会抛错。这里在写入前做一道保险: 强制每个区间 end = 下一个区间 start,
    最后一个区间 end = xmax。
    """
    # 转为秒 + 强制连续 (消除可能的微小间隙/重叠)
    sec_intervals = []
    for i, (t0, t1, text) in enumerate(intervals):
        s0 = t0 / 1000.0
        if i + 1 < len(intervals):
            s1 = intervals[i + 1][0] / 1000.0  # 下一个的 start
        else:
            s1 = xmax / 1000.0
        sec_intervals.append((s0, s1, text))

    tg = tglib.TextGrid(name="phones", minTime=0.0, maxTime=xmax / 1000.0)
    tier = tglib.IntervalTier(name="phones", minTime=0.0, maxTime=xmax / 1000.0)
    for s0, s1, text in sec_intervals:
        # 跳过零长度区间 (textgrid 库不允许 minTime == maxTime)
        if s1 - s0 < 1e-6:
            continue
        tier.addInterval(tglib.Interval(s0, s1, text))
    tg.append(tier)
    tg.write(str(path))


def _fmt(x):
    if x == int(x):
        return str(int(x))
    return f"{x:.6f}".rstrip("0").rstrip(".")


# =========================================================================
# 6. 主转换流程
# =========================================================================

def build_intervals_for_sample(entries, samples, sr, times, energy, duration_ms):
    """为一个录音样本构建音素区间列表。"""
    # 去重: 同一音节可能被多个 alias 引用 (如 ず/づ 共享同一位置),
    # 按 vowel_start 排序后, 相邻 vowel_start 差 < 100ms 视为重复, 只保留第一个
    sorted_entries = sorted(entries, key=lambda e: e.vowel_start)
    uniq = []
    for e in sorted_entries:
        if uniq and e.vowel_start - uniq[-1].vowel_start < 100.0:
            continue
        uniq.append(e)
    if not uniq:
        return [(0.0, duration_ms, "sil")]

    intervals = []
    # 前导静音: 从 0 到第一个音节边界
    first = uniq[0]
    first_kana = alias_parts(first.alias)[1]
    first_phones = kana_to_phones(first_kana, KANA_ROMAJI, ROMAJI_PHONES)

    if first_phones is None:
        # 第一个音节无法解析: 整段设为 sil (未闭合, 等待后续关闭)
        intervals.append((0.0, None, "sil"))
    else:
        first_consonant = first_phones[0] if first_phones else None
        first_vowel = first_phones[1] if len(first_phones) > 1 else None
        if first_consonant is not None:
            cs = detect_consonant_start(times, energy, first.vowel_start, 0.0,
                                        first_consonant, 0.0)
            cs = max(0.0, min(cs, first.vowel_start - 5.0))
            # 第一个辅音: 往前找静音结束点, 让辅音起始更靠前 (避免 sil 尾部含辅音)
            cs = extend_first_consonant_start(times, energy, cs, first.vowel_start)
            intervals.append((0.0, cs, "sil"))
            intervals.append((cs, first.vowel_start, first_consonant))
        else:
            intervals.append((0.0, first.vowel_start, "sil"))
        if first_vowel is not None:
            intervals.append((first.vowel_start, None, first_vowel))
    prev_vowel_start = first.vowel_start

    last_entry = first
    for idx, e in enumerate(uniq):
        e_kana = alias_parts(e.alias)[1]
        phones = kana_to_phones(e_kana, KANA_ROMAJI, ROMAJI_PHONES)
        if phones is None:
            # 无法解析的假名 (如 R 拖音标记): 关闭上个区间, 追加 sil
            vs = e.vowel_start
            _close_last(intervals, vs)
            intervals.append((vs, None, "sil"))
            prev_vowel_start = vs
            last_entry = e
            continue
        consonant, vowel = phones
        if idx == 0:
            prev_vowel_start = e.vowel_start
            last_entry = e
            continue
        vs = e.vowel_start
        if consonant is not None:
            cs = detect_consonant_start(times, energy, vs, prev_vowel_start,
                                        consonant, 0.0)
            cs = max(prev_vowel_start + 5.0, min(cs, vs - 5.0))
            # 中间辅音 (仅 s/sh/f/h): 在 [cs, vs] 内找能量极低点,
            # 取离 cs 最近的极小点作为新辅音起始, 修正辅音起始过早(切到上个元音尾部)的问题
            cs = refine_consonant_start(times, energy, cs, vs, consonant)
            _close_last(intervals, cs)
            intervals.append((cs, vs, consonant))
        else:
            # V-V: 前元音直接接到当前元音起点 (相邻同元音不合并, 保留音节边界)
            _close_last(intervals, vs)
        intervals.append((vs, None, vowel if vowel else "pau"))
        prev_vowel_start = vs
        last_entry = e

    # 尾部处理: 通过能量检测找到语音实际结束位置, 把最后元音延伸到该位置
    # (oto 的 region_end 通常落在元音持续段内, 之后还有元音尾音);
    # 之后真正的静音段标记为 sil
    sound_end = detect_sound_end(times, energy, last_entry.vowel_start, duration_ms)
    # 不能早于 region_end 太多 (避免误判把元音截短)
    sound_end = max(sound_end, min(last_entry.region_end, duration_ms))
    _close_last(intervals, sound_end)
    if sound_end < duration_ms - 1e-3:
        intervals.append((sound_end, duration_ms, "sil"))

    # 仅过滤零长度区间 (不合并相邻相同音素 — 它们代表不同音节)
    cleaned = []
    for t0, t1, text in intervals:
        if t1 is None or t1 - t0 < 1e-4:
            continue
        cleaned.append((t0, t1, text))
    if not cleaned:
        cleaned.append((0.0, duration_ms, "sil"))
    # 保证覆盖到 duration_ms (末尾补静音)
    if cleaned[-1][1] < duration_ms - 1e-3:
        cleaned.append((cleaned[-1][1], duration_ms, "sil"))
    return cleaned


def _close_last(intervals, end):
    """将最后一个未闭合区间 (end=None) 闭合到 end。"""
    if intervals and intervals[-1][1] is None:
        t0, _, text = intervals[-1]
        intervals[-1] = (t0, end, text)


def find_singer_dirs(input_path):
    """在输入路径下查找含 oto.ini 的音高子目录。"""
    input_path = Path(input_path)
    oto_dirs = []
    for root, _dirs, files in os.walk(input_path):
        if "oto.ini" in files:
            oto_dirs.append(Path(root))
    oto_dirs.sort()
    return oto_dirs


def collect_samples(oto_dir, entries):
    """收集该目录下所有被 oto 引用的 wav 样本 (去重保序)。"""
    seen = set()
    order = []
    for e in entries:
        if e.sample not in seen:
            seen.add(e.sample)
            order.append(e.sample)
    return order


def main():
    ap = argparse.ArgumentParser(description="UTAU VCV (Japanese) -> TextGrid converter")
    ap.add_argument("-i", "--input", default=str(HERE / "ARO-utau-vcv-jpn"),
                    help="UTAU 歌声音源根目录 (含音高子目录与 oto.ini)")
    ap.add_argument("-o", "--output", default=str(HERE / "output"),
                    help="输出根目录 (将创建 wav/ 与 TextGrid/ 子目录)")
    ap.add_argument("--copy", action="store_true",
                    help="复制 wav (默认为硬链接, 失败则复制)")
    args = ap.parse_args()

    global KANA_ROMAJI, ROMAJI_PHONES
    hira = _load_simple_dict(PHONESET_DIR / "japanese-hira2romaji-dict.txt")
    kata = _load_simple_dict(PHONESET_DIR / "japanese-kata2romaji-dict.txt")
    KANA_ROMAJI = build_kana_romaji_map(hira, kata)
    # 注意: japanese-romaji-dict.txt 才是 罗马音->音素 映射 (如 'ka k a');
    #       japanese-romaji-phones.txt 是 音素->类别 描述, 不要用错。
    ROMAJI_PHONES = _load_romaji_phones(PHONESET_DIR / "japanese-romaji-dict.txt")

    out_root = Path(args.output)
    wav_dir = out_root / "wav"
    tg_dir = out_root / "TextGrid"
    wav_dir.mkdir(parents=True, exist_ok=True)
    tg_dir.mkdir(parents=True, exist_ok=True)

    oto_dirs = find_singer_dirs(args.input)
    if not oto_dirs:
        print(f"[错误] 未在 {args.input} 下找到任何 oto.ini", file=sys.stderr)
        sys.exit(1)

    # 解析 prefix.map (若存在), 获取子目录代表音高
    input_root = Path(args.input)
    prefix_map = parse_prefix_map(input_root / "prefix.map")

    # 收集所有 (oto_dir, sample, midi) 对
    tasks = []
    for d in oto_dirs:
        entries = parse_oto(d / "oto.ini")
        # 音高确定优先级:
        # 1. 从 alias 后缀提取 (如 'a か_A3' -> 57)
        # 2. 从目录名提取 (如 'A3' -> 57)
        # 3. 从 prefix.map 提取 (如 '_A3' 子目录 -> 中位数音高)
        # 4. 留空, 后续用 F0 检测
        midi_dir = parse_pitch_to_midi(d.name)
        # 如果目录名无法解析, 尝试 prefix.map
        if midi_dir is None and d.name in prefix_map:
            midi_dir = prefix_map[d.name]
        # 如果目录是根目录 (与输入根相同), 也查 prefix.map
        if midi_dir is None and d == input_root:
            pass  # 根目录样本, 稍后用 F0 检测
        for sample in collect_samples(d, entries):
            wav_path = d / sample
            if wav_path.exists():
                sub_entries = [e for e in entries if e.sample == sample]
                # 尝试从 alias 后缀提取音高
                midi = midi_dir
                if midi is None:
                    for e in sub_entries:
                        m = detect_pitch_from_alias(e.alias)
                        if m is not None:
                            midi = m
                            break
                tasks.append((d, sample, wav_path, sub_entries, midi))

    total = len(tasks)
    width = max(4, len(str(total)))
    print(f"共发现 {total} 个样本, 输出到 {out_root}")

    unresolved = set()
    ok = 0
    csv_rows = []  # [(idx_str, sample_name, romaji_seq)]
    f0_detected = 0
    for i, (d, sample, wav_path, sub_entries, midi) in enumerate(tasks, start=1):
        base = str(i).zfill(width)
        # 文件名: NNNN_MM (MM = MIDI 编号); 若无音高则先用临时名
        idx_str = f"{base}_{midi}" if midi is not None else base
        try:
            samples, sr = read_wav(wav_path)
        except Exception as ex:
            print(f"[{idx_str}] 跳过 {sample}: 读取失败 {ex}", file=sys.stderr)
            continue
        # 若仍无音高, 用 F0 检测回退
        if midi is None:
            midi = detect_f0_midi(samples, sr)
            if midi is not None:
                f0_detected += 1
                idx_str = f"{base}_{midi}"
        duration_ms = len(samples) / sr * 1000.0
        times, energy = energy_envelope(samples, sr)

        intervals = build_intervals_for_sample(sub_entries, samples, sr, times, energy, duration_ms)

        # 检查未解析假名 + 收集罗马音序列 (与 build_intervals_for_sample 相同的去重逻辑)
        romaji_list = []
        sorted_sub = sorted(sub_entries, key=lambda x: x.vowel_start)
        prev_vs = None
        for e in sorted_sub:
            if prev_vs is not None and e.vowel_start - prev_vs < 100.0:
                continue
            prev_vs = e.vowel_start
            _, kana, _ = alias_parts(e.alias)
            if kana_to_phones(kana, KANA_ROMAJI, ROMAJI_PHONES) is None:
                unresolved.add(kana)
            rom = KANA_ROMAJI.get(kana, "?")
            romaji_list.append(rom)
        sample_name = Path(sample).stem
        if sample_name.startswith("_"):
            sample_name = sample_name[1:]
        csv_rows.append((idx_str, sample_name, " ".join(romaji_list)))

        # 写 wav
        out_wav = wav_dir / f"{idx_str}.wav"
        _write_wav(out_wav, wav_path)
        # 写 TextGrid
        out_tg = tg_dir / f"{idx_str}.TextGrid"
        write_textgrid(out_tg, intervals, duration_ms)
        ok += 1
        if i % 20 == 0 or i == total:
            print(f"  [{i}/{total}] {d.name}/{sample} -> {idx_str}")

    print(f"\n完成: {ok}/{total} 个样本已转换。")
    if f0_detected > 0:
        print(f"[音高检测] {f0_detected} 个样本通过 F0 检测确定音高。")
    if unresolved:
        print("[警告] 以下假名未能解析为音素, 请补充字典:", ", ".join(sorted(unresolved)))

    # 生成序号 -> 罗马音标注 CSV
    csv_path = out_root / "label.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("index,wav,romaji\n")
        for idx_str, sample_name, romaji in csv_rows:
            f.write(f"{idx_str},{sample_name},{romaji}\n")
    print(f"已生成标注 CSV: {csv_path}")


def _write_wav(dst, src):
    """复制或硬链接 wav 文件。若目标已存在且为同一文件则跳过, 否则覆盖。"""
    import shutil
    if dst.exists():
        try:
            if os.path.samefile(str(src), str(dst)):
                return  # 已链接到同一文件
        except OSError:
            pass
        dst.unlink()
    try:
        os.link(str(src), str(dst))
    except OSError:
        shutil.copyfile(str(src), str(dst))


if __name__ == "__main__":
    main()
