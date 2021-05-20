cimport cython
import numpy as np # We can swap this to cimport later, but will need to fix the includes in setup.py

@cython.boundscheck(False)
cdef unsigned short hamming(seq_a, seq_b, equivalence_d):
    distance = 0
    for i in range(len(seq_a)):
        if seq_a[i] not in equivalence_d[ seq_b[i] ]:
            distance += 1
    return distance

@cython.boundscheck(False)
def kernel(data_block, num_seqs, stride_len, result_arr, num_thread_pairs, num_pairs, idx_map, idy_map, block=None, grid=None, equivalence_d={}):
    for i in range(num_pairs):
        curr_idx = idx_map[i]
        curr_idy = idy_map[i]
        result_arr[i] = hamming(data_block[curr_idx], data_block[curr_idy], equivalence_d)
    return result_arr