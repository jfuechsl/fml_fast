import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def plot_weights(get_weights_func, d_range, n_plots, size):
    w = pd.DataFrame()
    for d in np.linspace(d_range[0], d_range[1], n_plots):
        w_ = get_weights_func(d, size=size)
        w_ = pd.DataFrame(w_, index=list(range(w_.shape[0]))[::-1], columns=[d])
        w = w.join(w_, how='outer')
    ax = w.plot()
    ax.legend(loc='upper right')
    plt.show()
    return

def test_frac_diff_d(series, column, frac_diff, min_d, max_d, steps, thres=1e-5):
    out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    for d in np.linspace(min_d, max_d, steps):
        df = frac_diff(series[[column]], d, thres)
        corr = np.corrcoef(series.loc[df.index, column], df[column])[0, 1]
        df = adfuller(df[column], maxlag=1, regression='c', autolag=None)
        out.loc[d] = list(df[:4]) + [df[4]['5%']] + [corr] # with critical value
    return out

def plot_frac_diff_d(d_data):
    d_data[['adfStat', 'corr']].plot(secondary_y='adfStat')
    plt.axhline(d_data['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')
