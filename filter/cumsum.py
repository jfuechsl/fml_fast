import pandas as pd

from . import _cumsum

def get_filter_events(y, h):
    if isinstance(h, pd.Series):
        df0 = pd.DataFrame({'y': y, 'h': h}).dropna()
    else:
        df0 = pd.DataFrame({'y': y, 'h': h})
        # h = pd.Series(h, index=y.index)
    evt = []
    Sp = 0
    Sm = 0
    dy = df0['y'].diff()
    for i in dy.index[1:]:
        Sp = max(0, Sp + dy.loc[i])
        Sm = min(0, Sm + dy.loc[i])
        if Sm < -df0.at[i, 'h']:
            Sm = 0
            evt.append(i) 
        elif Sp > df0.at[i, 'h']:
            Sp = 0
            evt.append(i)
    return pd.DatetimeIndex(evt)

def get_filter_events_fast(y, h):
    if isinstance(h, pd.Series):
        df0 = pd.DataFrame({'y': y, 'h': h}).dropna()
    else:
        df0 = pd.DataFrame({'y': y, 'h': h})
        # h = pd.Series(h, index=y.index)
    df0['dy'] = df0['y'].diff()
    eidx = _cumsum.filter_events_idx(df0['dy'].values, df0['h'].values, df0.shape[0])
    evt = df0.index[eidx]
    return pd.DatetimeIndex(evt)
