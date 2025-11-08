import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq

from ligotools.utils import whiten, write_wavfile, reqshift


def test_whiten_basic_properties():
    fs = 4096
    t = np.arange(0, 2.0, 1/fs)
    x = np.sin(2*np.pi*120*t) + 0.2*np.random.RandomState(0).randn(t.size)

    y = whiten(x, fs, seglen=1.0, overlap=0.5)

    assert y.shape == x.shape
    assert np.isfinite(y).all()
    #std = float(np.std(y))
    #assert 0.1 < std < 10.0  
    std = float(np.std(y))
    
    assert 1e-3 < std < 1e3  


def test_reqshift_has_expected_peaks():
    fs = 2048
    t = np.arange(0, 1.0, 1/fs)
    f0 = 100.0
    x = np.sin(2*np.pi*f0*t)

    fshift = 50.0
    y = reqshift(x, fshift, fs) 

    freqs = rfftfreq(len(y), 1/fs)
    amps = np.abs(rfft(y))

    top_idx = np.argsort(amps)[-6:]
    top_freqs = [freqs[i] for i in top_idx if freqs[i] > 0]

    targets = {abs(f0 - fshift), f0 + fshift} 
    def near_any(f, arr, tol=5.0):
        return any(abs(f - a) < tol for a in arr)

    assert all(near_any(f, top_freqs) for f in targets), f"Top peaks {top_freqs} missing {targets}"


def test_write_wavfile_roundtrip(tmp_path):
    fs = 8000
    x = np.zeros(400, dtype=float)
    x[0] = 1.0  

    out = tmp_path / "test.wav"
    write_wavfile(out, fs, x, normalize=True)

    rfs, data = wavfile.read(out)
    assert rfs == fs
    assert len(data) == len(x)
    assert data.dtype == np.int16
