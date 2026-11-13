"""Fetch explicitly selected US stocks and SPY with yfinance; no synthetic fallback.
Run: python fetch_real.py --end 2026-01-01
Requires: pip install yfinance pandas
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

GROWTH = ['NVDA','TSLA','AMD','META','AVGO','PLTR','SMCI','NFLX','AMZN','CRWD']
CONTROL = ['AAPL','MSFT','GOOGL','JPM','JNJ','KO','PG','WMT','XOM','COST']
SYMBOLS = GROWTH + CONTROL + ['SPY']

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--start',default='2010-01-01')
    parser.add_argument('--end',default='2026-01-01',help='Exclusive end. Defaults to full 2010-2025 history.')
    parser.add_argument('--out',type=Path,default=Path('data_real'))
    args=parser.parse_args()
    try:
        import yfinance as yf
    except ImportError:
        sys.exit('Install yfinance first: pip install yfinance')
    args.out.mkdir(parents=True,exist_ok=True)
    manifest={'provider':'Yahoo Finance via yfinance','downloaded_at_utc':datetime.now(timezone.utc).isoformat(),
              'start':args.start,'end_exclusive':args.end,'symbols_requested':SYMBOLS,
              'price_adjustment':'auto_adjust=True (splits and dividend-adjusted OHLC)',
              'caveat':'Survivorship-biased fixed contemporary basket; adjustment/volume conventions should be audited',
              'results':{}}
    failed=[]
    for sym in SYMBOLS:
        try:
            d=yf.download(sym,start=args.start,end=args.end,auto_adjust=True,actions=False,progress=False,
                          threads=False,timeout=25)
            if isinstance(d.columns,pd.MultiIndex):
                d.columns=d.columns.get_level_values(0)
            d=d.reset_index()
            if 'Date' not in d.columns or not {'Open','High','Low','Close','Volume'} <= set(d.columns):
                raise ValueError('Missing required daily columns')
            d=d[['Date','Open','High','Low','Close','Volume']].dropna()
            d['Date']=pd.to_datetime(d['Date']).dt.tz_localize(None).dt.strftime('%Y-%m-%d')
            if len(d)<250:raise ValueError(f'Only {len(d)} daily bars')
            d.to_csv(args.out / f'{sym}.csv',index=False)
            manifest['results'][sym]={'status':'ok','rows':len(d),'first':str(d.Date.iloc[0]),
                                      'last':str(d.Date.iloc[-1])}
            print(f'{sym:6s} {len(d):5d} {d.Date.iloc[0]} -> {d.Date.iloc[-1]}',flush=True)
        except Exception as exc:
            failed.append(sym)
            manifest['results'][sym]={'status':'failed','error':str(exc)}
            print(f'FAILED {sym}: {exc}',file=sys.stderr,flush=True)
    (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if failed:
        sys.exit(f'Download incomplete, missing {failed}. Do not treat it as a 20-stock experiment.')
    print('All 20 stocks plus SPY downloaded successfully.')

if __name__=='__main__':main()