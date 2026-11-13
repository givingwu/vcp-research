"""Deterministic simulated prices for checking that the pipeline runs.
NOT real quotes and NOT evidence of profitability.
"""
import numpy as np
import pandas as pd
from pathlib import Path
rng=np.random.default_rng(20261113)
out=Path('data_synthetic');out.mkdir(exist_ok=True)
dates=pd.bdate_range('2017-01-02','2026-10-30')
for k,sym in enumerate(['FAKE_A','FAKE_B','FAKE_C','FAKE_D','FAKE_E','FAKE_F','FAKE_G','FAKE_H','FAKE_I','FAKE_J']):
    n=len(dates)
    logret=0.0005+(k-4.5)*0.00008+rng.normal(0,0.017,n)
    close=50*np.exp(np.cumsum(logret))
    opening=np.roll(close,1)*np.exp(rng.normal(0,0.005,n));opening[0]=50
    high=np.maximum(close,opening)*(1+rng.uniform(0.002,0.014,n))
    low=np.minimum(close,opening)*(1-rng.uniform(0.002,0.014,n))
    vol=rng.integers(350000,1400000,n)
    pd.DataFrame({'Date':dates,'Open':opening,'High':high,'Low':low,'Close':close,'Volume':vol}).to_csv(out/f'{sym}.csv',index=False)
print('Generated deterministic SYNTHETIC bars in data_synthetic/')