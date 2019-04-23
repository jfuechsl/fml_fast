# cython: language_level=3, boundscheck=False

import numpy as np
f64 = np.dtype('f8')

# credit to: https://github.com/philipperemy/fractional-differentiation-time-series

def get_weight_ffd(float d, float thres, Py_ssize_t lim):
    w, k = [1.], 1
    ctr = 0
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
        ctr += 1
        if ctr == lim - 1:
            break
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

def frac_diff_ffd(x, d, thres=1e-5):
    w = get_weight_ffd(d, thres, len(x)).T
    cdef Py_ssize_t width = w.shape[1] - 1
    x = x.reshape(-1,1)
    cdef Py_ssize_t N = x.shape[0]
    output = np.full(N, np.nan, dtype=f64)
    cdef Py_ssize_t i
    for i in range(width, N):
        output[i] = np.dot(w, x[i - width:i + 1])[0]
    return output
