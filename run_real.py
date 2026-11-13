"""Guarded full protocol: 20 equities vs SPY, 2010-2025. Never falls back to synthetic."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from vcp_research.core import Config, load_prices, run, STRATEGIES
from fetch_real import GROWTH, CONTROL

DATA=Path('data_real'); OUT=Path('results_real')
EXPECTED=set(GROWTH+CONTROL+['SPY'])

def validate_universe():
    if not (DATA/'manifest.json').exists():
        raise RuntimeError('Real-market manifest missing. Run python fetch_real.py first. No synthetic fallback.')
    m=json.loads((DATA/'manifest.json').read_text())
    if m.get('provider')!='Yahoo Finance via yfinance':
        raise RuntimeError('Unexpected data provider')
    if set(m.get('results',{}))!=EXPECTED or any(x['status']!='ok' for x in m['results'].values()):
        raise RuntimeError('Incomplete requested universe. Review manifest.json.')
    raw=load_prices(DATA)
    if set(raw)!=EXPECTED:
        raise RuntimeError(f'CSV universe mismatch: missing={EXPECTED-set(raw)}, extra={set(raw)-EXPECTED}')
    for ticker,d in raw.items():
        if d.index.min()>pd.Timestamp('2024-01-01') or d.index.max()<pd.Timestamp('2025-12-15'):
            raise RuntimeError(f'{ticker}: insufficient test-period coverage')
    return raw

def benchmark_metrics(spy, start, end):
    d=spy.loc[start:end]
    if len(d)<2: return {}
    # Full-price adjusted, gross buy/hold from first bar open to final bar close.
    initial=float(d.open.iloc[0]); final=float(d.close.iloc[-1])
    days=(d.index[-1]-d.index[0]).days
    returns=d.close.pct_change().dropna()
    dd=(d.close/d.close.cummax()-1).min()
    vol=returns.std(ddof=0)
    return {'bars':len(d),'start':str(d.index[0].date()),'end':str(d.index[-1].date()),
            'total_return':round(final/initial-1,4),'cagr':round((final/initial)**(365.25/days)-1,4),
            'max_drawdown':round(float(dd),4),'sharpe_rf0':round(float(returns.mean()/vol*252**.5),3) if vol>0 else None,
            'note':'SPY total-return-adjusted buy-and-hold; no commission/slippage; unlike active strategies.'}

def main():
    raw=validate_universe()
    spy=raw.pop('SPY')
    cfg=Config(train_end='2019-12-31',validation_end='2022-12-31',
               test_start='2023-01-01',test_end='2025-12-31')
    equity_dir=OUT/'equities'; equity_dir.mkdir(parents=True,exist_ok=True)
    # Keep SPY out of selection universe. Its CSV is only benchmark.
    for sym,d in raw.items():
        x=d.reset_index();x.to_csv(equity_dir/f'{sym}.csv',index=False)
    results=run(equity_dir,OUT,cfg)
    results['benchmark_spy']={
        'train':benchmark_metrics(spy,'2010-01-01','2019-12-31'),
        'validation':benchmark_metrics(spy,'2020-01-01','2022-12-31'),
        'test':benchmark_metrics(spy,'2023-01-01','2025-12-31')}
    results['metadata']['source_manifest']='../data_real/manifest.json'
    results['metadata']['status']='EXPLORATORY_REAL_DATA_SURVIVORSHIP_BIASED'
    results['metadata']['limitations']=['Current selected stocks only; survivorship and selection bias',
       'VCP is heuristic, not J Law exact algorithm','Dividend-adjusted Yahoo bars; verify adjustment and volumes',
       'Independent folds reset accounts; not a continuous compounded performance history',
       'SPY benchmark has no trading costs, differing from active strategies']
    (OUT/'summary.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
    lines=['# VCP vs Momentum — exploratory real stock backtest','',
           'Source: Yahoo Finance (yfinance, adjusted OHLCV). 20 selected contemporary stocks + SPY.',
           '**Important:** fixed contemporary stock basket; survivorship bias. Not J Law exact strategy.','',
           '| Test (2023–2025) | Return | Max drawdown | Sharpe (rf=0) | Closed trades |',
           '|---|---:|---:|---:|---:|']
    for s in STRATEGIES:
        q=results['test'][s]
        lines.append(f"| {s} | {q['total_return']:.2%} | {q['max_drawdown']:.2%} | {q['sharpe_rf0']} | {q['trades_closed']} |")
    s=results['benchmark_spy']['test']
    lines.append(f"| SPY buy-and-hold (gross) | {s['total_return']:.2%} | {s['max_drawdown']:.2%} | {s['sharpe_rf0']} | — |")
    lines += ['', 'Caution: less than 100 closed trades per signal strategy is low statistical power.',
              'No claim of alpha without historical point-in-time universe and robustness analysis.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print((OUT/'REPORT.md').read_text())

if __name__=='__main__':main()