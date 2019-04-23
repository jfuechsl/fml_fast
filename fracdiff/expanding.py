import pandas as pd
import numpy as np

def get_weights(d, size):
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1]/k*(d-k+1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

def frac_diff(series, d, thres=.01):
    '''
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    # 1) compute weights for the longest series
    w = get_weights(d, series.shape[0])
    # 2) determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_>thres].shape[0]
    # 3) apply weights to values
    df = {}
    for name in series.columns:
        series_f = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series()
        for iloc in range(skip, series_f.shape[0]):
            loc = series_f.index[iloc]
            if not np.isfinite(series.loc[loc, name]):
                continue # exclude NaNs
            df_[loc] = np.dot(w[-(iloc+1):, :].T, series_f.loc[:loc])[0, ]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df

def frac_diff_fast(series, d):
    # from: https://github.com/philipperemy/fractional-differentiation-time-series/blob/master/fracdiff/fracdiff.py
    df = {}
    for name in series.columns:
        series_f = series[[name]].fillna(method='ffill').dropna()
        x = series_f.values
        T = len(x)
        np2 = int(2 ** np.ceil(np.log2(2 * T - 1)))
        k = np.arange(1, T)
        b = np.append([1], np.cumprod((k - d - 1) / k))
        z = np.zeros(np2 - T)
        z1 = np.append(b, z)
        z2 = np.append(x, z)
        dx = np.fft.ifft(np.fft.fft(z1) * np.fft.fft(z2))
        dx = np.real(dx[0:T])
        df[name] = pd.Series(dx, index=series_f.index)
    df = pd.concat(df, axis=1)
    return df
