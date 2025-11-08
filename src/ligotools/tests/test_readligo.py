# src/ligotools/tests/test_readligo.py
from pathlib import Path
import numbers
import numpy as np
import ligotools.readligo as rl


def _pick_one():
    data = Path("data")
    # use whichever detector file is there
    matches = sorted(data.glob("H-*.hdf5")) or sorted(data.glob("L-*.hdf5"))
    assert matches, f"No .hdf5 files found in {data.resolve()}"
    return matches[0]


def test_read_hdf5_basic_structure():
    f = _pick_one()
    out = rl.read_hdf5(str(f))

    # 1. it should be a tuple with at least strain + something else
    assert isinstance(out, tuple)
    assert len(out) >= 2

    # 2. first item = strain array
    strain = out[0]
    assert isinstance(strain, np.ndarray)
    assert strain.ndim == 1
    assert strain.size > 0

    # 3. second item in *your* version is a GPS start time (a number)
    gps_or_time = out[1]
    assert isinstance(gps_or_time, numbers.Real)


def test_read_hdf5_has_metadata_dict_somewhere():
    f = _pick_one()
    out = rl.read_hdf5(str(f))

    # look through the rest of the returned values for a dict
    meta_dicts = [x for x in out if isinstance(x, dict)]
    assert meta_dicts, "expected at least one metadata dict in read_hdf5 return"
    md = meta_dicts[0]

    # very light checks: it should have at least one key
    assert len(md) >= 1
