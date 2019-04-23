# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np

def filter_events_idx(double [:] dy, double [:]h, long M):
    cdef long i
    cdef long [:] evt
    cdef double Sp, Sm, Sp_, Sm_
    evt = np.zeros(0).astype(long)
    Sp = 0.
    Sm = 0.
    for i in range(1, M):
        Sp_ = Sp + dy[i]
        if Sp_ > 0.:
            Sp = Sp_
        else:
            Sp = 0.
        Sm_ = Sm + dy[i]
        if Sm_ < 0.:
            Sm = Sm_
        else:
            Sm = 0.
        if Sm < -h[i]:
            Sm = 0.
            evt = np.append(evt, i) 
        elif Sp > h[i]:
            Sp = 0.
            evt = np.append(evt, i) 
    return evt
