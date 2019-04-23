import pandas as pd
import numpy as np

from . import _fracdiff as cfd

def get_weights(d, thres, lim=None):
    w = [1.]
    k = 1
    while True:
        w_ = -w[-1]/k*(d-k+1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
        if lim and k >= lim:
            break
    return np.array(w[::-1]).reshape(-1,1)

def frac_diff(series, d, thres=1e-5):
    # constant width window (new solution)
    w = get_weights(d, thres, series.shape[0])
    width = len(w)-1
    df = {}
    for name in series.columns:
        series_f = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series()
        for iloc1 in range(width, series_f.shape[0]):
            loc0 = series_f.index[iloc1-width]
            loc1 = series_f.index[iloc1]
            # if not np.isfinite(series.loc[loc1, name]):
            #     continue # exclude Nas
            df_[loc1] = np.dot(w.T, series_f.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df

def frac_diff_fast(series, d, thres=1e-5):
    df = {}
    for name in series.columns:
        series_f = series[[name]].fillna(method='ffill').dropna()
        dx = cfd.frac_diff_ffd(series_f.values, d, thres)
        df[name] = pd.Series(dx, index=series_f.index).dropna()
    df = pd.concat(df, axis=1)
    return df
