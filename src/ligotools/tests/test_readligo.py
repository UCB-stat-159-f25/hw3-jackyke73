from pathlib import Path
import numbers
import numpy as np
import ligotools.readligo as rl

def _pick_one():
    data = Path("data")
    matches = sorted(data.glob("H-*.hdf5")) or sorted(data.glob("L-*.hdf5"))
    assert matches, f"No .hdf5 files found in {data.resolve()}"
    return matches[0]


def test_read_hdf5_basic_structure():
    f = _pick_one()
    out = rl.read_hdf5(str(f))
    assert isinstance(out, tuple)
    assert len(out) >= 2
    strain = out[0]
    assert isinstance(strain, np.ndarray)
    assert strain.ndim == 1
    assert strain.size > 0
    gps_or_time = out[1]
    assert isinstance(gps_or_time, numbers.Real)
    
def test_strain_is_finite_and_not_constant():
    f = _pick_one()
    out = rl.read_hdf5(str(f))
    strain = out[0]
    assert np.isfinite(strain).all()
    assert np.nanstd(strain) > 0
