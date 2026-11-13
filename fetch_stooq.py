"""Fetch sample US OHLCV data; requires network and is not PIT complete universe.
Usage: python fetch_stooq.py AAPL MSFT NVDA ...
"""
import argparse
import io
from pathlib import Path
import requests
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('symbols',nargs='+')
p.add_argument('--start',default='20150101')
p.add_argument('--end',default='20261101')
p.add_argument('--out',type=Path,default=Path('data'))
a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
for sym in a.symbols:
    url='https://stooq.com/q/d/l/'
    response=requests.get(url,params={'s':sym.lower()+'.us','i':'d','d1':a.start,'d2':a.end},timeout=20,headers={'User-Agent':'Mozilla/5.0'})
    response.raise_for_status()
    df=pd.read_csv(io.StringIO(response.text))
    if not {'Date','Open','High','Low','Close','Volume'}<=set(df.columns):
        raise RuntimeError(f'{sym} had no valid OHLCV data; check ticker/vendor limits')
    df.to_csv(a.out/f'{sym.upper()}.csv',index=False)
    print(sym,len(df))