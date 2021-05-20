cimport cython
import numpy as np
from libc.stdint cimport int8_t

ctypedef int8_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef unsigned short hamming(str seq_a, str seq_b, Py_ssize_t l, DTYPE_t[::1] ord_l, DTYPE_t[:, ::1] alphabet_mat):
    cdef unsigned short distance = 0
    cdef unsigned short a, b
    for i in range(l):
        a = ord_l[ord(seq_a[i])]
        b = ord_l[ord(seq_b[i])]
        distance += alphabet_mat[a][b]
    return distance

@cython.boundscheck(False)
@cython.wraparound(False)
def kernel(data_block, num_seqs, stride_len, result_arr, num_thread_pairs, num_pairs, idx_map, idy_map, ord_l, equivalence_d):
    for i in range(num_pairs):
        curr_idx = idx_map[i]
        curr_idy = idy_map[i]
        result_arr[i] = hamming(data_block[curr_idx], data_block[curr_idy], stride_len, ord_l, equivalence_d)
    return result_arr