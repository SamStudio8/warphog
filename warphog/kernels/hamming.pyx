cimport cython
import numpy as np
from libc.stdint cimport int8_t, uint16_t, uint32_t

ctypedef int8_t DTYPE_t
ctypedef uint16_t RTYPE_t
ctypedef uint32_t MTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef unsigned short hamming(char* seq_a, char* seq_b, unsigned int l, DTYPE_t[::1] ord_l, DTYPE_t[:, ::1] alphabet_mat):
    cdef unsigned short distance = 0
    cdef unsigned short a, b
    for i in range(l):
        a = ord_l[seq_a[i]]
        b = ord_l[seq_b[i]]
        distance += alphabet_mat[a][b]
    return distance

@cython.boundscheck(False)
@cython.wraparound(False)
def kernel(data_block, num_seqs, unsigned int stride_len, RTYPE_t[::1] result_arr, num_thread_pairs, unsigned long num_pairs, MTYPE_t[::1] idx_map, MTYPE_t[::1] idy_map, ord_l, alphabet_mat):
    cdef unsigned int curr_idx, curr_idy
    for i in range(num_pairs):
        curr_idx = idx_map[i]
        curr_idy = idy_map[i]
        result_arr[i] = hamming(data_block[curr_idx].encode(), data_block[curr_idy].encode(), stride_len, ord_l, alphabet_mat)
    return result_arr