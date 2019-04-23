import pandas as pd
import numpy as np

from . import _sequentialbootstrap

def comp_ind_matrix(bar_idx, t1):
    # get indicator matrix
    ind = np.zeros((bar_idx.shape[0], t1.shape[0]))
    for i, (t0_, t1_) in enumerate(t1.iteritems()):
        ti0 = bar_idx.searchsorted(t0_)
        ti1 = bar_idx.searchsorted(t1_)
        ind[ti0:ti1+1, i] = 1
    ind = pd.DataFrame(ind, index=bar_idx, columns=range(t1.shape[0]))
    return ind

def comp_ind_mask(bar_idx, t1):
    indm = np.zeros((2, t1.shape[0])).astype(int)
    for i, (t0_, t1_) in enumerate(t1.iteritems()):
        ti0_ = bar_idx.searchsorted(t0_)
        ti1_ = bar_idx.searchsorted(t1_)
        indm[0, i] = ti0_
        indm[1, i] = ti1_
    return indm

def comp_c(indm, M):
    c = np.zeros(M)
    for i in range(indm.shape[1]):
       ti0_ = indm[0, i]
       ti1_ = indm[1, i]
       c[ti0_:ti1_+1] += 1
    return c

def comp_avg_uniqueness(ind):
    # average uniqueness from indicator matrix
    c = ind.sum(axis=1) # concurrency
    u = ind.div(c, axis=0) # uniqueness
    avg_u = u[u>0].mean() # average uniqueness
    return avg_u

def comp_avg_uniqueness_m(indm, M):
    c = comp_c(indm, M)
    avgu = np.zeros(indm.shape[1])
    for i in range(indm.shape[1]):
       ti0_ = indm[0, i]
       ti1_ = indm[1, i]
       avgu[i] = (1. / c[ti0_:ti1_+1]).mean()
    return avgu

def sequential_bootstrap_slooow(ind, s_length=None):
    # generate a sample via sequential bootstrap
    if s_length is None:
        s_length = ind.shape[1]
    phi = []
    while len(phi) < s_length:
        avg_u = pd.Series()
        for i in ind:
            ind_ = ind[phi+[i]] # reduce ind matrix
            avg_u.loc[i] = comp_avg_uniqueness(ind_).iloc[-1]
        prob = avg_u/avg_u.sum() # draw prob
        phi += [np.random.choice(ind.columns, p=prob)]
    return phi

def sequential_bootstrap(indm, M, s_length=None):
    return _sequentialbootstrap.sequential_bootstrap(indm, M, s_length)
