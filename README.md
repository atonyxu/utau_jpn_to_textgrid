# UTAU VCV Japanese to Praat TextGrid Converter

A tool that converts UTAU VCV (Japanese) voice bank samples into Praat TextGrid phonetic annotations with automatic consonant onset detection and pitch labeling.

## Overview

[UTAU](http://utau2008.web.fc2.com/) is a Japanese singing voice synthesis software. Voice banks are organized by pitch directories, each containing `oto.ini` (phonetic boundary metadata) and `.wav` audio samples in VCV (Vowel-Consonant-Vowel) format.

This tool reads UTAU voice bank directories, parses the `oto.ini` files, analyzes the audio to detect precise consonant onset positions, and generates [Praat](https://www.fon.hum.uva.nl/praat/) TextGrid files with phone-level annotations — useful for phonetic research, forced alignment, and singing voice analysis.

## Features

- **Automatic `oto.ini` parsing** with encoding detection (Shift-JIS, UTF-8, EUC-JP)
- **Kana → Romaji → Phoneme conversion** supporting full Japanese kana set including:
  - Hiragana and Katakana
  - Compound kana (e.g., きゃ → kya, しゃ → sha)
  - Small kana combinations (ゃ, ゅ, ょ, ぁ–ぉ)
  - Geminate consonants (っ/ッ → cl)
- **Consonant onset detection** via audio energy analysis, with category-specific strategies:
  - **Stops & affricates** (k, t, p, ch, ts, etc.): detects closure (silence) onset before burst
  - **Fricatives** (s, sh, h, f, etc.): detects noise energy rise point
  - **Sonorants** (n, m, r, w, y, etc.): finds energy valley between adjacent vowels
- **F0 (pitch) detection** via autocorrelation as a fallback when metadata is unavailable
- **Pitch resolution** from multiple sources with priority:
  1. Alias suffix (e.g., `a か_A3`)
  2. Directory name (e.g., `A3`, `D4-N`, `F#4`)
  3. `prefix.map` mapping
  4. Audio-based F0 detection
- **CSV label export** (`label.csv`) mapping sample indices to romaji sequences

## Requirements

- Python 3.7+
- [NumPy](https://numpy.org/)
- [textgrid](https://pypi.org/project/textgrid/) (Praat TextGrid read/write library)

Install dependencies:

```bash
pip install numpy textgrid
```

## Usage

### Basic Usage

```bash
python utau_to_textgrid.py -i /path/to/utau_voicebank -o /path/to/output
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `-i`, `--input` | `ARO-utau-vcv-jpn` | UTAU voice bank root directory (contains pitch subdirectories with `oto.ini` + `.wav`) |
| `-o`, `--output` | `output` | Output root directory (creates `wav/` and `TextGrid/` subdirectories) |
| `--copy` | off | Copy `.wav` files instead of hard-linking (hard-link is default, falls back to copy on failure) |

### Example

```bash
python utau_to_textgrid.py -i ./my_voicebank -o ./aligned_data
```

## Input Structure

The tool expects a standard UTAU voice bank directory layout:

```
voicebank_root/
├── prefix.map          (optional — pitch-to-directory mapping)
├── A3/                 (pitch subdirectory, name encodes MIDI note)
│   ├── oto.ini         (phonetic boundary definitions)
│   ├── あ.wav          (audio samples)
│   ├── か.wav
│   └── ...
├── C4/
│   ├── oto.ini
│   └── ...
└── ...
```

### `oto.ini` Format

Each line in `oto.ini` follows the pattern:

```
filename.wav=alias,left,fixed,right,preutterance,overlap
```

- **alias**: VCV alias string, e.g., `a か_A3` (previous vowel `当前假名_pitch`)
- **left**: Left boundary offset (ms)
- **fixed**: Fixed consonant region length (ms)
- **right**: Right boundary offset (ms, often negative)
- **preutterance**: Consonant-to-vowel transition point relative to `left` (ms)
- **overlap**: Overlap region length (ms)

The vowel onset position is computed as `left + preutterance`.

### `prefix.map` Format

Optional file mapping musical pitches to directory names:

```
B3\t\t_A3
C4\t\t_A3
```

Each tab-separated line maps a note name to a subdirectory (prefixed with `_`).

## Output Structure

```
output/
├── wav/
│   ├── 0001_57.wav       (renamed: sequential index_MIDI note)
│   ├── 0002_60.wav
│   └── ...
├── TextGrid/
│   ├── 0001_57.TextGrid  (Praat TextGrid, phone-level intervals)
│   ├── 0002_60.TextGrid
│   └── ...
└── label.csv             (index → romaji mapping)
```

### `label.csv` Format

```csv
index,wav,romaji
0001_57,あ,a
0002_60,か,k a
```

### TextGrid Format

Each `.TextGrid` file contains a single interval tier named `phones` with non-overlapping, contiguous intervals covering the full audio duration. Interval labels use the following conventions:

| Label | Meaning |
|-------|---------|
| `sil` | Silence (leading/trailing) |
| `pau` | Pause (between vowels with no consonant) |
| `N` | Moraic nasal (ん/ン) |
| `cl` | Consonant closure (っ/ッ) |
| `k`, `s`, `t`, ... | Individual consonant phonemes |
| `a`, `i`, `u`, `e`, `o` | Vowel phonemes |

## Phone Set

The `jpn-phoneset/` directory contains the phonetic dictionaries:

| File | Description |
|------|-------------|
| `japanese-hira2romaji-dict.txt` | Hiragana → Romaji mapping (e.g., `か → ka`) |
| `japanese-kata2romaji-dict.txt` | Katakana → Romaji mapping (e.g., `カ → ka`) |
| `japanese-romaji-dict.txt` | Romaji → Phoneme list mapping (e.g., `ka → k a`, `kya → ky a`) |
| `japanese-romaji-phones.txt` | Phoneme category definitions (vowel, stop, fricative, nasal, etc.) |

### Supported Phonemes

**Vowels:** a, i, u, e, o, N

**Consonants (by category):**

| Category | Phonemes |
|----------|----------|
| Stops | k, kw, ky, g, gw, gy, t, ty, d, dy, p, py, b, by, ch, ts, j, z, cl |
| Fricatives | s, sh, h, hy, f, fy |
| Nasals | n, ny, m, my |
| Liquids | r, ry |
| Semivowels | w, y, v |

Compound kana (拗音) are automatically computed from base kana + small kana combinations (e.g., き + ゃ → kya → ky a).

## How It Works

### 1. Kana to Phoneme Conversion

1. Look up kana in hiragana/katakana → romaji dictionaries
2. For compound kana (base + small kana), compute romaji via rules (e.g., きゃ = ki + ゃ → kya)
3. Map romaji to phoneme sequences using `japanese-romaji-dict.txt`
4. Extract (consonant, vowel) pairs for each syllable

### 2. Consonant Onset Detection

Since `oto.ini` only provides the vowel onset (preutterance) position, consonant start times are detected from audio energy analysis:

- **Stops/Affricates** (k, t, p, ch, ts, j, z, etc.): Skip the burst region (~12ms before vowel onset), then scan backward to find the onset of the closure (sustained silence).
- **Fricatives** (s, sh, h, f, etc.): Scan backward from vowel onset to find where noise energy first rises above a threshold.
- **Sonorants** (n, m, r, w, y, etc.): Find the energy valley (minimum) between the previous vowel and current vowel.

A refinement step adjusts overly early consonant onsets for fricatives (s, sh, f, h) that may have been placed in the tail of the preceding vowel.

### 3. F0 (Pitch) Detection

When pitch cannot be determined from metadata (directory name, alias suffix, or `prefix.map`), the tool falls back to autocorrelation-based F0 detection:

1. Compute short-time RMS energy envelope
2. Select voiced frames (energy > 20% of maximum)
3. Run normalized autocorrelation on each voiced frame
4. Find the dominant pitch peak in the 70–500 Hz range
5. Take the median F0 across all voiced frames
6. Convert to MIDI note number: `midi = 69 + 12 * log2(f0 / 440)`

### 4. TextGrid Generation

Phoneme intervals are constructed by:

1. Sorting all `oto.ini` entries by vowel onset position
2. Deduplicating entries with nearly identical onset times (< 100ms apart)
3. For each syllable: placing a consonant interval before the vowel onset and a vowel interval from onset to the next boundary
4. Detecting the actual sound end via energy analysis for the trailing silence
5. Writing contiguous, non-overlapping intervals to a Praat TextGrid file

## Notes

- The tool requires `.wav` files to be in a format Python's `wave` module can read (PCM, 8/16/32-bit, mono or stereo)
- Multi-channel audio is automatically downmixed to mono
- Hard-linking is preferred for `.wav` output (fast, no disk overhead); falls back to copying if hard links fail (e.g., cross-device)
- Unresolved kana (not found in dictionaries) are reported as warnings with suggestions to supplement the dictionary
- The tool processes all pitch directories found under the input root, including nested subdirectories

## License

See repository for license information.
