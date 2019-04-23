# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np

def sequential_bootstrap(Py_ssize_t [:,:] indm, Py_ssize_t M, s_length=None):
    cdef Py_ssize_t [:] indices
    cdef Py_ssize_t N, Nsamples, i, j, k, phi_
    cdef Py_ssize_t t0, t1
    cdef double [:] avg_u, c
    cdef double avg_u_, Savg_u_
    N = indm.shape[1]
    Nsamples = N
    if s_length is not None:
        Nsamples = <Py_ssize_t>s_length
    phi = []
    indices = np.linspace(0, N, num=N, endpoint=False).astype(int)
    avg_u = np.zeros(N)
    c = np.zeros(M) # concurrency
    for i in range(Nsamples):
        Savg_u_ = 0.
        for j in range(N):
            # BEGIN inline: compute average uniqueness of the sample candidate
            avg_u_ = 0.
            t0 = indm[0, j]
            t1 = indm[1, j]
            for k in range(t0, t1+1):
                # compute the average uniqueness at time k
                # but don't update the concurrency yet - it's a candidate, not yet sampled
                avg_u_ += 1. / (c[k] + 1.)
            avg_u_ /= <double>(t1-t0+1)
            # END inline
            avg_u[j] = avg_u_
            Savg_u_ += avg_u_
        # normalize the average uniqueness values, arriving at the sample probabilities
        for j in range(N):
            avg_u[j] /= Savg_u_
        # sample the new element
        phi_ = np.random.choice(indices, p=avg_u)
        # now we update the concurrency values with the new sample added
        t0 = indm[0, phi_]
        t1 = indm[1, phi_]
        for k in range(t0, t1+1):
            c[k] += 1.
        # and add the new sample to our bootstrap set
        phi.append(phi_)
    return phi
