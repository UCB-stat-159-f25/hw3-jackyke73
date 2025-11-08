from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.fft import rfft, irfft, rfftfreq
from scipy.io.wavfile import write as wavwrite
def whiten(strain: np.ndarray, fs: float, seglen: int | float = 4, overlap: int | float = 2, eps: float = 1e-12) -> np.ndarray:
    strain = np.asarray(strain, dtype=float)
    nperseg = int(seglen * fs)
    noverlap = int(overlap * fs)
    freqs, psd = welch(strain, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Nt = strain.size
    freqs_r = rfftfreq(Nt, 1.0 / fs)
    hf = rfft(strain)
    psd_i = np.interp(freqs_r, freqs, psd)
    white_hf = hf / (np.sqrt(psd_i / 2.0) + eps)
    white = irfft(white_hf, n=Nt)
    return white.astype(np.float64)


def write_wavfile(path: str | Path, fs: int | float, data: np.ndarray, normalize: bool = True) -> None:
    x = np.asarray(data, dtype=float)
    if normalize:
        peak = float(np.max(np.abs(x))) or 1.0
        x = (x / peak * 32767.0).astype(np.int16)
    else:
        x = x.astype(np.int16)
    wavwrite(str(path), int(fs), x)


def reqshift(data: np.ndarray, fshift: float, fs: float) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    t = np.arange(len(data)) / float(fs)
    return np.real(data * np.exp(2j * np.pi * fshift * t))


def plot_psd_axes(strain: np.ndarray, fs: float, seglen: int | float = 4, overlap: int | float = 2,
                  ax=None, title: str | None = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    nperseg = int(seglen * fs)
    noverlap = int(overlap * fs)
    f, Pxx = welch(np.asarray(strain, dtype=float), fs=fs, nperseg=nperseg, noverlap=noverlap)
    ax.loglog(f, np.sqrt(Pxx))
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("ASD [strain/√Hz]")
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    return ax

__all__ = ["whiten", "write_wavfile", "reqshift", "plot_psd_axes"]
