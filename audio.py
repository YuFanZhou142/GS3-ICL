from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


def _to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _pcm_bytes_to_waveform(raw: bytes, sample_width: int, num_channels: int) -> torch.Tensor:
    buffer = memoryview(bytearray(raw))
    if sample_width == 1:
        data = torch.frombuffer(buffer, dtype=torch.uint8).clone()
        waveform = (data.to(torch.float32) - 128.0) / 128.0
    elif sample_width == 2:
        waveform = torch.frombuffer(buffer, dtype=torch.int16).clone().to(torch.float32)
        waveform = waveform / 32768.0
    elif sample_width == 3:
        data = torch.frombuffer(buffer, dtype=torch.uint8).clone().to(torch.int32)
        data = data.view(-1, 3)
        waveform = data[:, 0] | (data[:, 1] << 8) | (data[:, 2] << 16)
        sign_bit = 1 << 23
        waveform = torch.where((waveform & sign_bit) != 0, waveform - (1 << 24), waveform)
        waveform = waveform.to(torch.float32) / float(1 << 23)
    elif sample_width == 4:
        waveform = torch.frombuffer(buffer, dtype=torch.int32).clone().to(torch.float32)
        waveform = waveform / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if num_channels > 1:
        if waveform.numel() % num_channels != 0:
            raise ValueError("WAV payload length is not divisible by channel count")
        waveform = waveform.view(-1, num_channels).transpose(0, 1).contiguous()
        waveform = waveform.mean(dim=0)

    return waveform.contiguous()


def load_wav(path: str | Path) -> tuple[torch.Tensor, int]:
    """Load a PCM WAV file into a mono float32 waveform tensor."""

    wav_path = _to_path(path)
    with wave.open(str(wav_path), "rb") as handle:
        num_channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        num_frames = handle.getnframes()
        raw = handle.readframes(num_frames)

    waveform = _pcm_bytes_to_waveform(raw, sample_width, num_channels)
    return waveform, sample_rate


def resample_waveform(waveform: torch.Tensor, orig_sample_rate: int, target_sample_rate: int) -> torch.Tensor:
    """Resample a 1D waveform using linear interpolation."""

    if orig_sample_rate == target_sample_rate:
        return waveform.clone()
    if waveform.numel() == 0:
        return waveform.clone()

    waveform_ = waveform.to(torch.float32).view(1, 1, -1)
    target_length = max(1, int(round(waveform_.shape[-1] * float(target_sample_rate) / float(orig_sample_rate))))
    resampled = F.interpolate(waveform_, size=target_length, mode="linear", align_corners=False)
    return resampled.view(-1)


def _hz_to_mel(hz: torch.Tensor | float) -> torch.Tensor:
    hz_tensor = torch.as_tensor(hz, dtype=torch.float32)
    return 2595.0 * torch.log10(1.0 + hz_tensor / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def _mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> torch.Tensor:
    num_freq_bins = n_fft // 2 + 1
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, dtype=torch.float32)
    hz_points = _mel_to_hz(mel_points)
    fft_bins = torch.floor((n_fft + 1) * hz_points / float(sample_rate)).to(torch.long)

    fb = torch.zeros(n_mels, num_freq_bins, dtype=torch.float32)
    for m in range(1, n_mels + 1):
        left = int(fft_bins[m - 1].item())
        center = int(fft_bins[m].item())
        right = int(fft_bins[m + 1].item())

        left = max(left, 0)
        center = min(max(center, left + 1), num_freq_bins - 1)
        right = min(max(right, center + 1), num_freq_bins)

        if center > left:
            fb[m - 1, left:center] = torch.linspace(0.0, 1.0, center - left, dtype=torch.float32)
        if right > center:
            fb[m - 1, center:right] = torch.linspace(1.0, 0.0, right - center, dtype=torch.float32)

    return fb


def log_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    target_sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    win_length: int = 400,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: float | None = None,
    center: bool = False,
    power: float = 2.0,
    log_offset: float = 1e-6,
) -> torch.Tensor:
    """Convert a mono waveform into a log-mel feature sequence."""

    if waveform.numel() == 0:
        return waveform.new_empty((0, n_mels), dtype=torch.float32)

    if sample_rate != target_sample_rate:
        waveform = resample_waveform(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    if f_max is None:
        f_max = float(sample_rate) / 2.0

    window = torch.hann_window(win_length, device=waveform.device, dtype=torch.float32)
    spectrum = torch.stft(
        waveform.to(torch.float32),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        return_complex=True,
    )
    magnitude = spectrum.abs().pow(power)
    mel_fb = _mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max).to(magnitude.device, magnitude.dtype)
    mel = mel_fb @ magnitude
    return torch.log(mel + log_offset).transpose(0, 1).contiguous()


def slice_segments(
    sequence: torch.Tensor,
    segment_length: int,
    *,
    hop_length: int | None = None,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Split a sequence into fixed-length segments along the first dimension."""

    if hop_length is None:
        hop_length = segment_length
    if sequence.shape[0] == 0:
        shape = (0, segment_length, *sequence.shape[1:])
        return sequence.new_empty(shape)

    length = int(sequence.shape[0])
    starts = list(range(0, max(length, 1), hop_length))
    last_start = max(length - segment_length, 0)
    if starts[-1] != last_start and last_start >= 0:
        starts.append(last_start)
    starts = sorted(set(starts))

    segments = []
    for start in starts:
        stop = start + segment_length
        chunk = sequence[start:stop]
        if chunk.shape[0] < segment_length:
            pad_shape = (segment_length - chunk.shape[0], *sequence.shape[1:])
            pad = sequence.new_full(pad_shape, pad_value)
            chunk = torch.cat([chunk, pad], dim=0)
        segments.append(chunk)

    return torch.stack(segments, dim=0)


def chunk_waveform(
    waveform: torch.Tensor,
    segment_samples: int,
    *,
    hop_samples: int | None = None,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Slice a waveform into a batch of sample windows."""

    return slice_segments(waveform.view(-1, 1), segment_samples, hop_length=hop_samples, pad_value=pad_value).squeeze(-1)
