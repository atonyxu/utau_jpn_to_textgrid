#!/usr/bin/env python3
"""
hifigan.py - 一键音频变调工具

使用 RMVPE 提取音高 (F0)，通过 PC-NSF-HiFiGAN 声码器实现保持共振峰的变调。
模型文件 (首次运行自动下载到脚本目录下的 models/):
  models/rmvpe.pt            - RMVPE 音高提取模型
  models/pc-nsf-hifigan.ckpt - PC-NSF-HiFiGAN 声码器模型

用法:
  # 单文件, 单调值
  python hifigan.py -i test.wav -o output -s 4

  # 单文件, 同时输出 +4 和 -3 两个调值
  python hifigan.py -i test.wav -o output -s 4 -3

  # 文件夹输入, 递归扫描所有 wav, 批量处理
  python hifigan.py -i input_wavs -o output -s 4 -3 --device cuda

  # 增大预读数, 进一步重叠 I/O 与 GPU (显存充裕时)
  python hifigan.py -i input_wavs -o output -s 4 --prefetch 2

输出文件命名: <input_basename>_<tag>.wav, tag 例如 +4 / -3 / +4.5
输出保持与输入相同的采样率、位深和声道数。
"""

import os

os.environ["LRU_CACHE_CAPACITY"] = "3"

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from librosa.filters import mel as librosa_mel_fn
from torchaudio.transforms import Resample
import librosa
import soundfile as sf

# 脚本所在目录 (用于定位 models/ 子目录, 与启动时的 cwd 无关)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

# 模型下载地址 (modelscope)
MODEL_URLS = {
    'rmvpe.pt': 'https://modelscope.cn/datasets/virtual-singer/daisy-base/resolve/master/model.pt',
    'pc-nsf-hifigan.ckpt': 'https://modelscope.cn/datasets/virtual-singer/daisy-base/resolve/master/pc-nsf-hifigan.ckpt',
}


def ensure_models(max_retries=3):
    """检查 models 目录下的模型文件, 缺失则用 pySmartDL 下载, 失败重试最多 3 次"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    missing = [name for name in MODEL_URLS
               if not os.path.exists(os.path.join(MODELS_DIR, name))]
    if not missing:
        return

    try:
        from pySmartDL import SmartDL
    except ImportError:
        print('pySmartDL 未安装, 正在尝试自动安装...')
        import subprocess
        subprocess.check_call(['pip', 'install', 'pySmartDL'])
        from pySmartDL import SmartDL

    import time
    skipped = []
    for name in missing:
        url = MODEL_URLS[name]
        dest = os.path.join(MODELS_DIR, name)
        success = False
        for attempt in range(1, max_retries + 1):
            print(f'下载模型: {name}  (第 {attempt}/{max_retries} 次)')
            print(f'  URL: {url}')
            print(f'  目标: {dest}')
            # 清理上次失败留下的残文件
            if os.path.exists(dest):
                os.remove(dest)
            try:
                dl = SmartDL(url, dest, progress_bar=False)
                dl.start(blocking=True)
                if dl.isSuccessful():
                    size_mb = dl.get_dl_size() / 1024 / 1024
                    print(f'  完成 ({size_mb:.1f} MB)')
                    success = True
                    break
                else:
                    errors = dl.get_errors()
                    print(f'  失败: {errors}')
            except Exception as e:
                print(f'  异常: {e}')
            if attempt < max_retries:
                wait = 2 * attempt
                print(f'  {wait}s 后重试...')
                time.sleep(wait)

        if not success:
            print(f'  [跳过] {name} 已达最大重试次数 ({max_retries}), 跳过')
            skipped.append(name)

    if skipped:
        print(f'\n警告: 以下模型下载失败已跳过: {skipped}')
        print('请检查网络后重新运行, 或手动下载到 models/ 目录')
        # 验证最终缺失情况
        still_missing = [n for n in MODEL_URLS
                         if not os.path.exists(os.path.join(MODELS_DIR, n))]
        if still_missing:
            print(f'仍缺失: {still_missing}')


# ==========================================================================
#  PC-NSF-HiFiGAN 声码器
# ==========================================================================

LRELU_SLOPE = 0.1

# PC-NSF-HiFiGAN 声码器配置 (硬编码, 对应 pc-nsf-hifigan-44.1k-hop512-128bin-2025.02)
VOCODER_CONFIG = {
    "discriminator_periods": [3, 5, 7, 11, 17, 23, 37],
    "mini_nsf": True,
    "resblock": "1",
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "resblock_kernel_sizes": [3, 7, 11],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 4, 4, 4],
    "upsample_rates": [8, 8, 2, 2, 2],
    "sampling_rate": 44100,
    "num_mels": 128,
    "hop_size": 512,
    "n_fft": 2048,
    "win_size": 2048,
    "fmin": 40,
    "fmax": 16000,
    "pc_aug": True,
}


class AttrDict(dict):
    """支持属性访问的字典"""

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getattr__(self, name):
        if name not in super(AttrDict, self).keys():
            return None
        return super(AttrDict, self).__getitem__(name)

    def __setattr__(self, key, value):
        super(AttrDict, self).__setitem__(key, value)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                               dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                               dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                               dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                               dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                               dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class SineGen(nn.Module):
    """正弦波生成器 (用于非 mini_nsf 模式)"""

    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1,
                 noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02sine(self, f0, upp):
        rad = f0 / self.sampling_rate * torch.arange(1, upp + 1, device=f0.device)
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad = rad.reshape(f0.shape[0], -1, 1)
        rad = torch.multiply(
            rad,
            torch.arange(1, self.dim + 1, device=f0.device).reshape(1, 1, -1)
        )
        rand_ini = torch.rand(1, 1, self.dim, device=f0.device)
        rand_ini[..., 0] = 0
        rad += rand_ini
        sines = torch.sin(2 * np.pi * rad)
        return sines

    @torch.no_grad()
    def forward(self, f0, upp):
        f0 = f0.unsqueeze(-1)
        sine_waves = self._f02sine(f0, upp) * self.sine_amp
        uv = (f0 > self.voiced_threshold).float()
        uv = F.interpolate(uv.transpose(2, 1), scale_factor=upp,
                           mode='nearest').transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves


class SourceModuleHnNSF(nn.Module):
    """谐波源模块 (用于非 mini_nsf 模式)"""

    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp,
                                 add_noise_std, voiced_threshold)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp):
        sine_wavs = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge


class Generator(nn.Module):
    """PC-NSF-HiFiGAN 生成器"""

    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.mini_nsf = h.mini_nsf
        self.noise_sigma = h.noise_sigma

        if h.mini_nsf:
            self.source_sr = h.sampling_rate / int(np.prod(h.upsample_rates[2:]))
            self.upp = int(np.prod(h.upsample_rates[:2]))
        else:
            self.source_sr = h.sampling_rate
            self.upp = int(np.prod(h.upsample_rates))
            self.m_source = SourceModuleHnNSF(
                sampling_rate=h.sampling_rate, harmonic_num=8
            )
            self.noise_convs = nn.ModuleList()

        self.conv_pre = weight_norm(
            Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2
        ch = h.upsample_initial_channel
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            ch //= 2
            self.ups.append(
                weight_norm(ConvTranspose1d(2 * ch, ch, k, u, padding=(k - u) // 2))
            )
            for j, (k2, d) in enumerate(zip(h.resblock_kernel_sizes,
                                            h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k2, d))
            if not h.mini_nsf:
                if i + 1 < len(h.upsample_rates):
                    stride_f0 = int(np.prod(h.upsample_rates[i + 1:]))
                    self.noise_convs.append(Conv1d(
                        1, ch, kernel_size=stride_f0 * 2,
                        stride=stride_f0, padding=stride_f0 // 2
                    ))
                else:
                    self.noise_convs.append(Conv1d(1, ch, kernel_size=1))
            elif i == 1:
                self.source_conv = Conv1d(1, ch, 1)
                self.source_conv.apply(init_weights)

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def fastsinegen(self, f0):
        """MiniNSF 的高速正弦波生成"""
        n = torch.arange(1, self.upp + 1, device=f0.device)
        s0 = f0.unsqueeze(-1) / self.source_sr
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * n + 0.5 * ds0 * n * (n - 1) / self.upp
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad = rad.reshape(f0.shape[0], 1, -1)
        sines = torch.sin(2 * np.pi * rad)
        return sines

    def forward(self, x, f0):
        if self.mini_nsf:
            har_source = self.fastsinegen(f0)
        else:
            har_source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        if self.noise_sigma is not None and self.noise_sigma > 0:
            x += self.noise_sigma * torch.randn_like(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            if not self.mini_nsf:
                x_source = self.noise_convs[i](har_source)
                x = x + x_source
            elif i == 1:
                x_source = self.source_conv(har_source)
                x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


class STFT:
    """Mel 频谱图提取 (与声码器训练一致的配置)"""

    def __init__(self, sr=44100, n_mels=128, n_fft=2048, win_size=2048,
                 hop_length=512, fmin=40, fmax=16000, clip_val=1e-5,
                 device=None):
        self.target_sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        mel_basis = librosa_mel_fn(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        self.mel_basis = torch.from_numpy(mel_basis).float().to(self.device)

    def get_mel(self, y, center=False):
        if torch.min(y) < -1.0:
            print('min value is ', torch.min(y))
        if torch.max(y) > 1.0:
            print('max value is ', torch.max(y))

        window = torch.hann_window(self.win_size, device=self.device)

        y = F.pad(
            y.unsqueeze(1),
            ((self.win_size - self.hop_length) // 2,
             (self.win_size - self.hop_length + 1) // 2),
            mode='reflect'
        )
        y = y.squeeze(1)

        spec = torch.stft(
            y, self.n_fft, hop_length=self.hop_length,
            win_length=self.win_size, window=window,
            center=center, pad_mode='reflect', normalized=False,
            onesided=True, return_complex=True
        ).abs()

        spec = torch.matmul(self.mel_basis, spec)
        spec = dynamic_range_compression_torch(spec, clip_val=self.clip_val)
        return spec


def load_vocoder(ckpt_path, device):
    """加载 PC-NSF-HiFiGAN 声码器 (配置已硬编码在 VOCODER_CONFIG)"""
    h = AttrDict(VOCODER_CONFIG)

    generator = Generator(h)
    try:
        cp_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        cp_dict = torch.load(ckpt_path, map_location='cpu')

    if 'generator' in cp_dict:
        state_dict = cp_dict['generator']
    elif 'state_dict' in cp_dict:
        state_dict = {k.replace('generator.', ''): v
                      for k, v in cp_dict['state_dict'].items()
                      if k.startswith('generator.')}
    else:
        state_dict = cp_dict

    generator.load_state_dict(state_dict)
    generator.eval()
    generator.remove_weight_norm()
    generator.to(device)
    return generator, h


# ==========================================================================
#  RMVPE 音高提取器
# ==========================================================================

RMVPE_SAMPLE_RATE = 16000
RMVPE_N_CLASS = 360
RMVPE_N_MELS = 128
RMVPE_MEL_FMIN = 30
RMVPE_MEL_FMAX = RMVPE_SAMPLE_RATE // 2
RMVPE_WINDOW_LENGTH = 1024
RMVPE_CONST = 1997.3794084376191


class RMVPEMelSpectrogram(nn.Module):
    """RMVPE 使用的 Mel 频谱提取 (16kHz)"""

    def __init__(self, n_mel_channels, sampling_rate, win_length, hop_length,
                 n_fft=None, mel_fmin=0, mel_fmax=None, clamp=1e-5):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels,
            fmin=mel_fmin, fmax=mel_fmax, htk=True
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, center=True):
        keyshift_key = '0_' + str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(
                self.win_length
            ).to(audio.device)

        fft = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.hann_window[keyshift_key],
            center=center, return_complex=True
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_features,
                          num_layers=num_layers, batch_first=True,
                          bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        if self.is_shortcut:
            return self.conv(x) + self.shortcut(x)
        return self.conv(x) + x


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1,
                 momentum=0.01):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.conv[i](x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1,
                 momentum=0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, (3, 3), stride,
                               (1, 1), out_padding, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i in range(self.n_blocks):
            x = self.conv2[i](x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks,
                 out_channels=16, momentum=0.01):
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(in_channels, out_channels, kernel_size,
                                n_blocks, momentum)
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for i in range(self.n_encoders):
            _, x = self.layers[i](x)
            concat_tensors.append(_)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks,
                 momentum=0.01):
        super().__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(
            ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)
        )
        for i in range(self.n_inters - 1):
            self.layers.append(
                ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
            )

    def forward(self, x):
        for i in range(self.n_inters):
            x = self.layers[i](x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks,
                 momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i in range(self.n_decoders):
            x = self.layers[i](x, concat_tensors[-1 - i])
        return x


class TimbreFilter(nn.Module):
    def __init__(self, latent_rep_channels):
        super().__init__()
        self.layers = nn.ModuleList()
        for latent_rep in latent_rep_channels:
            self.layers.append(ConvBlockRes(latent_rep[0], latent_rep[0]))

    def forward(self, x_tensors):
        out_tensors = []
        for i, layer in enumerate(self.layers):
            out_tensors.append(layer(x_tensors[i]))
        return out_tensors


class DeepUnet0(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4,
                 in_channels=1, en_out_channels=16):
        super().__init__()
        self.encoder = Encoder(in_channels, RMVPE_N_MELS, en_de_layers,
                               kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2, self.encoder.out_channel,
            inter_layers, n_blocks
        )
        self.tf = TimbreFilter(self.encoder.latent_channels)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers,
                               kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class E2E0(nn.Module):
    """RMVPE 端到端模型 (不带 TimbreFilter 的前向路径)"""

    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5,
                 inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.unet = DeepUnet0(kernel_size, n_blocks, en_de_layers,
                              inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * RMVPE_N_MELS, 256, n_gru),
                nn.Linear(512, RMVPE_N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * RMVPE_N_MELS, RMVPE_N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid()
            )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


def to_local_average_f0(hidden, thred=0.03):
    """从 hidden 状态解码 F0 (局部加权平均)"""
    idx = torch.arange(RMVPE_N_CLASS, device=hidden.device)[None, None, :]
    idx_cents = idx * 20 + RMVPE_CONST
    center = torch.argmax(hidden, dim=2, keepdim=True)
    start = torch.clip(center - 4, min=0)
    end = torch.clip(center + 5, max=RMVPE_N_CLASS)
    idx_mask = (idx >= start) & (idx < end)
    weights = hidden * idx_mask
    product_sum = torch.sum(weights * idx_cents, dim=2)
    weight_sum = torch.sum(weights, dim=2)
    cents = product_sum / (weight_sum + (weight_sum == 0))
    f0 = 10 * 2 ** (cents / 1200)
    uv = hidden.max(dim=2)[0] < thred
    f0 = f0 * ~uv
    return f0.squeeze(0).cpu().numpy()


class RMVPE:
    """RMVPE 音高提取器"""

    def __init__(self, model_path, hop_length=160):
        self.resample_kernel = {}
        model = E2E0(4, 1, (2, 2))
        try:
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(model_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        model.eval()
        self.hop_length = hop_length
        self.seg_length = 32 * hop_length
        self.model = model
        self.mel_extractor = RMVPEMelSpectrogram(
            RMVPE_N_MELS, RMVPE_SAMPLE_RATE, RMVPE_WINDOW_LENGTH,
            hop_length, None, RMVPE_MEL_FMIN, RMVPE_MEL_FMAX
        )
        self.resample_kernel = {}

    def infer_from_audio(self, audio, sample_rate=16000, device=None,
                         thred=0.03, use_viterbi=False):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        audio = torch.from_numpy(audio).float().unsqueeze(0).to(device)
        if sample_rate == 16000:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(
                    sample_rate, 16000, lowpass_filter_width=128
                )
            self.resample_kernel[key_str] = self.resample_kernel[key_str].to(device)
            audio_res = self.resample_kernel[key_str](audio)

        B, T = audio_res.shape
        n_frames = T // self.hop_length + 1
        T1 = T + self.hop_length
        T_pad = self.seg_length * ((T1 - 1) // self.seg_length + 1) - T1
        audio_res = F.pad(audio_res, (0, T_pad))
        mel_extractor = self.mel_extractor.to(device)
        self.model = self.model.to(device)
        mel = mel_extractor(audio_res, center=True)
        with torch.no_grad():
            hidden = self.model(mel)
        f0 = to_local_average_f0(hidden[:, :n_frames], thred=thred)
        return f0


# ==========================================================================
#  变调主流程
# ==========================================================================

def _format_shift_tag(shift_semitones):
    """把半音数格式化为文件名后缀, 如 +4 / -3 / +4.5"""
    return f'{shift_semitones:+g}'


def scan_wav_files(input_path):
    """输入为文件则返回单元素列表; 为文件夹则递归扫描所有 .wav/.WAV 文件"""
    if os.path.isfile(input_path):
        return [input_path]
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f'输入路径不存在: {input_path}')
    wav_files = []
    for root, _, files in os.walk(input_path):
        for f in files:
            if f.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, f))
    wav_files.sort()
    return wav_files


class PitchShifter:
    """变调处理器: 一次性加载 RMVPE + 声码器, 复用于所有文件和调值"""

    def __init__(self, rmvpe_path, vocoder_ckpt, device='auto',
                 thred=0.03, use_viterbi=False):
        if device == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device(device)
        self.thred = thred
        self.use_viterbi = use_viterbi
        self.vocoder_sr = 44100

        # GPU 自动优化参数
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            # 依据显存自动选择是否启用半精度加速
            # (本模型对 fp16 不稳健, 仍用 fp32, 但开启 cudnn benchmark)
            gpu_name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            print(f'[GPU] {gpu_name}, 显存 {vram_mb:.0f} MB, '
                  f'cudnn.benchmark=True')
        print(f'[1/4] 使用设备: {self.device}')

        # 加载模型 (只加载一次)
        print('[2/4] 加载 RMVPE...')
        self.rmvpe = RMVPE(rmvpe_path, hop_length=160)
        print('[3/4] 加载声码器...')
        self.generator, self.h = load_vocoder(vocoder_ckpt, self.device)
        h_config = AttrDict(VOCODER_CONFIG)
        self.h_config = h_config
        self.stft = STFT(
            sr=h_config.sampling_rate,
            n_mels=h_config.num_mels,
            n_fft=h_config.n_fft,
            win_size=h_config.win_size,
            hop_length=h_config.hop_size,
            fmin=h_config.fmin,
            fmax=h_config.fmax,
            device=self.device,
        )
        print('[4/4] 模型就绪')

    def _extract_features(self, audio_44100):
        """提取 Mel 和对齐的 F0 (GPU)"""
        audio_t = torch.from_numpy(audio_44100).float().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            mel = self.stft.get_mel(audio_t, center=False).squeeze(0)
        n_mel_frames = mel.shape[-1]

        f0_rmvpe = self.rmvpe.infer_from_audio(
            audio_44100, sample_rate=self.vocoder_sr, device=self.device,
            thred=self.thred, use_viterbi=self.use_viterbi
        )
        n_f0_frames = len(f0_rmvpe)
        # 对齐 F0 到 Mel 帧数
        f0_times = np.arange(n_f0_frames) * (160.0 / RMVPE_SAMPLE_RATE)
        mel_times = np.arange(n_mel_frames) * (
            self.h_config.hop_size / self.h_config.sampling_rate
        )
        f0_aligned = np.interp(mel_times, f0_times, f0_rmvpe)
        return mel, f0_aligned

    def process_audio(self, audio_mono, orig_sr, orig_subtype, orig_channels,
                      base_name, output_dir, shift_list):
        """对一个音频执行所有调值的变调并保存"""
        target_len = len(audio_mono)
        # 重采样到声码器采样率
        if orig_sr != self.vocoder_sr:
            audio_44100 = librosa.resample(
                audio_mono, orig_sr=orig_sr, target_sr=self.vocoder_sr
            )
        else:
            audio_44100 = audio_mono

        mel, f0_aligned = self._extract_features(audio_44100)
        mel_in = mel.unsqueeze(0).to(self.device)

        results = []
        for shift_semitones in shift_list:
            if shift_semitones != 0:
                f0_shifted = f0_aligned * (2.0 ** (shift_semitones / 12.0))
                f0_shifted[f0_aligned == 0] = 0
            else:
                f0_shifted = f0_aligned

            f0_in = torch.from_numpy(f0_shifted).float().unsqueeze(0).to(self.device)
            with torch.inference_mode():
                wav_out = self.generator(mel_in, f0_in).view(-1)
            wav_out = wav_out.cpu().numpy()

            if len(wav_out) > target_len:
                wav_out = wav_out[:target_len]
            # 重采样回原始采样率
            if orig_sr != self.vocoder_sr:
                wav_out = librosa.resample(
                    wav_out, orig_sr=self.vocoder_sr, target_sr=orig_sr
                )
                if len(wav_out) > target_len:
                    wav_out = wav_out[:target_len]
            # 多声道复制
            if orig_channels > 1:
                wav_out = np.stack([wav_out] * orig_channels, axis=1)

            tag = _format_shift_tag(shift_semitones)
            out_path = os.path.join(output_dir, f'{base_name}_{tag}.wav')
            sf.write(out_path, wav_out, orig_sr, subtype=orig_subtype)
            results.append(out_path)
        return results

    @staticmethod
    def _load_audio(input_path):
        """读取音频文件 (CPU/磁盘 I/O), 返回单声道 + 原始格式信息"""
        info = sf.info(input_path)
        audio_multi, _ = sf.read(input_path, dtype='float32', always_2d=True)
        if audio_multi.shape[1] > 1:
            audio_mono = audio_multi.mean(axis=1)
        else:
            audio_mono = audio_multi[:, 0]
        return (audio_mono, info.samplerate, info.subtype, info.channels,
                info.frames)


def run_pipeline(input_path, output_dir, shift_semitones_list,
                 rmvpe_path=None, vocoder_ckpt=None,
                 device='auto', thred=0.03, use_viterbi=False,
                 prefetch=1, quiet=False):
    """
    批量变调主入口:
      - input_path 可为文件或文件夹 (文件夹会递归扫描 .wav)
      - 模型只加载一次, 所有文件复用
      - 生产者线程预读下一个音频 (磁盘 I/O + 重采样), 与 GPU 计算重叠

    参数:
        prefetch: 预读的文件数 (重叠 I/O 与 GPU), 默认 1
    """
    if rmvpe_path is None:
        rmvpe_path = os.path.join(MODELS_DIR, 'rmvpe.pt')
    if vocoder_ckpt is None:
        vocoder_ckpt = os.path.join(MODELS_DIR, 'pc-nsf-hifigan.ckpt')

    os.makedirs(output_dir, exist_ok=True)
    wav_files = scan_wav_files(input_path)
    if not wav_files:
        raise FileNotFoundError(f'未找到 wav 文件: {input_path}')
    if not quiet:
        print(f'发现 {len(wav_files)} 个 wav 文件, 调值 {shift_semitones_list}, '
              f'预计输出 {len(wav_files) * len(shift_semitones_list)} 个文件\n')

    # 加载处理器 (模型只加载一次)
    shifter = PitchShifter(rmvpe_path, vocoder_ckpt, device, thred, use_viterbi)

    import queue
    import threading

    # 生产者: 预读音频 (CPU/磁盘), 与 GPU 计算重叠
    q = queue.Queue(maxsize=prefetch)
    SENTINEL = None

    def producer(files):
        for fp in files:
            try:
                audio_mono, orig_sr, orig_subtype, orig_channels, frames = \
                    PitchShifter._load_audio(fp)
                q.put((fp, audio_mono, orig_sr, orig_subtype,
                       orig_channels, frames))
            except Exception as e:
                print(f'  [跳过] 读取失败 {fp}: {e}')
                q.put((fp, None, None, None, None, 0))
        q.put(SENTINEL)

    t = threading.Thread(target=producer, args=(wav_files,), daemon=True)
    t.start()

    total = len(wav_files)
    done = 0
    all_results = []
    import time
    t0 = time.time()
    while True:
        item = q.get()
        if item is SENTINEL:
            break
        fp, audio_mono, orig_sr, orig_subtype, orig_channels, frames = item
        done += 1
        if audio_mono is None:
            continue
        base_name = os.path.splitext(os.path.basename(fp))[0]
        if quiet:
            print(f'[{done}/{total}] {os.path.basename(fp)}')
        else:
            dur = frames / orig_sr if orig_sr else 0
            print(f'[{done}/{total}] {os.path.basename(fp)}  '
                  f'({orig_sr}Hz, {orig_subtype}, {orig_channels}ch, {dur:.1f}s)')
        try:
            results = shifter.process_audio(
                audio_mono, orig_sr, orig_subtype, orig_channels,
                base_name, output_dir, shift_semitones_list
            )
            if quiet:
                count = len(results)
                tags = [_format_shift_tag(s) for s in shift_semitones_list]
                print(f'       -> {count} outputs: {", ".join(tags)}')
            else:
                for r in results:
                    print(f'     -> {os.path.basename(r)}')
            all_results.extend(results)
        except Exception as e:
            print(f'  [失败] {fp}: {e}')
        # 周期性清理显存碎片
        if shifter.device.type == 'cuda' and done % 5 == 0:
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    if not quiet:
        print(f'\n完成! 共生成 {len(all_results)} 个文件于: {output_dir}')
        print(f'总耗时 {elapsed:.1f}s, 平均 {elapsed / max(total, 1):.1f}s/文件')
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='PC-NSF-HiFiGAN + RMVPE 批量音频变调 (支持文件夹输入)'
    )
    parser.add_argument(
        '-i', '--input', type=str, default='test.wav',
        help='输入音频文件或文件夹 (文件夹将递归扫描 .wav)'
    )
    parser.add_argument(
        '-o', '--output-dir', type=str, default='output',
        help='输出文件夹 (默认: output); 文件名自动为 <input>_<tag>.wav'
    )
    parser.add_argument(
        '-s', '--shift', type=float, nargs='+', default=[0],
        help='变调半音数 (可指定多个, 范围 -12 ~ +12), 例如 -s 4 -3'
    )
    parser.add_argument(
        '-d', '--device', type=str, default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='推理设备 (默认: auto)'
    )
    parser.add_argument(
        '--rmvpe', type=str,
        default=os.path.join(MODELS_DIR, 'rmvpe.pt'),
        help='RMVPE 模型路径'
    )
    parser.add_argument(
        '--vocoder', type=str,
        default=os.path.join(MODELS_DIR, 'pc-nsf-hifigan.ckpt'),
        help='声码器模型路径'
    )
    parser.add_argument(
        '--thred', type=float, default=0.03,
        help='RMVPE 清浊音阈值 (默认: 0.03)'
    )
    parser.add_argument(
        '--viterbi', action='store_true',
        help='使用 Viterbi 解码 F0 (默认: 局部加权平均)'
    )
    parser.add_argument(
        '--prefetch', type=int, default=1,
        help='预读文件数, 重叠 I/O 与 GPU (默认: 1)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='简化输出, 每完成一个文件只打印一行'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 启动时检查模型文件, 缺失则自动下载到 <脚本目录>/models/
    ensure_models()

    run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        shift_semitones_list=args.shift,
        rmvpe_path=args.rmvpe,
        vocoder_ckpt=args.vocoder,
        device=args.device,
        thred=args.thred,
        use_viterbi=args.viterbi,
        prefetch=args.prefetch,
        quiet=args.quiet,
    )
