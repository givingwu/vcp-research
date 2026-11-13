import json
import pytest
import pandas as pd
from pathlib import Path
from fetch_real import SYMBOLS
from run_real import benchmark_metrics


def test_universe_exact_20_plus_spy():
    assert len(SYMBOLS)==21
    assert len(set(SYMBOLS))==21
    assert 'SPY' in SYMBOLS


def test_benchmark_total_return_simple():
    dates=pd.bdate_range('2024-01-02',periods=4)
    bars=pd.DataFrame({'open':[100,101,102,103], 'close':[100,101,102,110]},index=dates)
    b=benchmark_metrics(bars,'2024-01-01','2024-12-31')
    assert b['total_return']==0.1
    assert b['bars']==4


def test_missing_manifest_refuses_real_result(tmp_path,monkeypatch):
    import run_real
    monkeypatch.setattr(run_real,'DATA',tmp_path)
    with pytest.raises(RuntimeError,match='manifest missing'):
        run_real.validate_universe()