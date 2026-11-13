import numpy as np
import pandas as pd
from vcp_research.core import add_features,vcp_at,backtest,Config,metrics


def make_data():
    idx=pd.bdate_range('2020-01-01',periods=260)
    price=100+np.arange(260)*0.3
    return add_features(pd.DataFrame({'open':price,'high':price*1.01,'low':price*.99,
        'close':price,'volume':1_000_000},index=idx))


def test_features_no_forward_dependency():
    d=make_data()
    before=d.iloc[240][['sma50','sma200','mom126']].to_numpy(dtype=float)
    modified=d[['open','high','low','close','volume']].copy()
    modified.loc[modified.index[-1],['open','high','low','close']]=10_000
    after=add_features(modified).iloc[240][['sma50','sma200','mom126']].to_numpy(dtype=float)
    assert np.allclose(before,after)
    a=vcp_at(d,240)
    b=vcp_at(add_features(modified),240)
    assert a==b


def test_next_open_execution_gap_stop_and_cost():
    d=make_data().iloc[-6:].copy()
    d.loc[d.index[1],['open','high','low','close']]=[100,103,97,102]
    d.loc[d.index[2],['open','high','low','close']]=[98,100,90,94]
    sig=pd.DataFrame({'entry':[True,False,False,False,False,False],
        'stop':[98.,np.nan,np.nan,np.nan,np.nan,np.nan],
        'exit':[False]*6},index=d.index)
    cfg=Config(commission_bps=10,slippage_bps=10,max_position_fraction=0.2)
    c,t=backtest({'TEST':d},{'TEST':sig},cfg,'unit',str(d.index[0].date()),str(d.index[-1].date()))
    assert len(t)==1
    assert t.iloc[0].entry_date==d.index[1]
    assert t.iloc[0].exit_date==d.index[1] # same-day low triggers stop conservatively
    assert t.iloc[0].reason=='entry_day_stop'
    assert t.iloc[0].pnl<0


def test_missing_symbols_no_signal_errors():
    d=make_data()
    assert not bool(d.trend.iloc[20])
    assert bool(d.trend.iloc[-1])
    empty=pd.DataFrame({'entry':False,'stop':np.nan,'exit':False},index=d.index)
    c,t=backtest({'X':d},{'X':empty},Config(),'A_trend','2020-01-01','2021-01-01')
    assert len(t)==0
    assert metrics(c,t,100_000)['total_return']==0