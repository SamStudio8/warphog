import random
import sys
from math import ceil, sqrt

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pytest
from warphog import cores
from warphog.util import DEFAULT_ALPHABET


def hamming(seq_a, seq_b, equivalence_d):
    assert len(seq_a) == len(seq_b)
    distance = 0
    for i in range(len(seq_a)):
        if seq_a[i] not in equivalence_d[ seq_b[i] ]:
            distance += 1
    return distance

def test_hamming_test():
    tests = [
        ("AAAAA", "CCCCC", 5),
        ("ACGTN", "ACGTN", 0),
        ("ACGTN", "NACGT", 3),
        ("ACGT", "ACGN", 0),
        ("ACGT", "ACNT", 0),
        ("ACGT", "ANGT", 0),
        ("ACGT", "NCGT", 0),
        ("AAAAA", "NNNNN", 0),
        ("NNNNN", "ACGTN", 0),
        ("TTTTT", "UUUUU", 0),
        ("RRRRRR", "ACGTNR", 3),
        ("YYYYYY", "ACGTNY", 3),
        ("SSSSSS", "ACGTNS", 3),
        ("WWWWWW", "ACGTNW", 3),
        ("KKKKKK", "ACGTNK", 3),
        ("MMMMMM", "ACGTNM", 3),
        ("BBBBBB", "ACGTNB", 2),
        ("DDDDDD", "ACGTND", 2),
        ("HHHHHH", "ACGTNH", 2),
        ("VVVVVV", "ACGTNV", 2),
        ("??????", "?ACGTN", 5),
        ("ACGTN-", "ACGTN-", 0),
    ]
    for test in tests:
        distance = hamming(test[0], test[1], DEFAULT_ALPHABET.equivalent_d)
        assert distance == test[2]
        distance = hamming(test[1], test[0], DEFAULT_ALPHABET.equivalent_d)
        assert distance == test[2]

def do_kernel(expected, seq_block, num_pairs, idx_map, idy_map, block_dim, grid_dim):
    tot_tests = 0

    core = cores.GPUWarpCore(seq_block, DEFAULT_ALPHABET)
    d = _do_kernel(core, seq_block, num_pairs, idx_map, idy_map, block_dim, grid_dim, pairs_per_thread=1)
    n_tests = do_kernel_test(expected, d)
    tot_tests += n_tests
    assert n_tests == num_pairs

    core = cores.CPUPreWarpCore(seq_block, DEFAULT_ALPHABET)
    d = _do_kernel(core, seq_block, num_pairs, idx_map, idy_map, block_dim, grid_dim, pairs_per_thread=1)
    n_tests = do_kernel_test(expected, d)
    tot_tests += n_tests
    assert n_tests == num_pairs

    sys.stderr.write("%d pairs checked" % tot_tests)
    return tot_tests

def _do_kernel(core, seq_block, num_pairs, idx_map, idy_map, block_dim, grid_dim, pairs_per_thread=1):
    d = np.zeros(num_pairs, dtype=np.uint16)
    core.put_d(d)

    core.put_maps(idx_map, idy_map)

    seq_len = len(seq_block[0])
    core.kernel.engage(
        core.data_block,
        np.uint(len(seq_block)),
        np.uint32(seq_len), # msa stride
        core.result_arr,
        np.int32(pairs_per_thread),
        np.uint(num_pairs),
        core.idx_map,
        core.idy_map,
        block=block_dim,
        grid=grid_dim,
    )
    core.get_d(d)
    return d


def setup_test_M1_100A():
    """Return the data required to test the 100A sequence with shuffling T"""
    seq_dim_x = 1
    seq_dim_y = 100
    pairs = seq_dim_x * seq_dim_y

    a = ["A"] * 100
    seq_block = []

    seq_block.append(''.join(a))
    expected = [0]

    for i in range(seq_dim_y-1):
        curr_seq = a.copy()
        curr_seq[i] = "T"
        seq_block.append(''.join(curr_seq))
        expected.append(1)

    # Map all sequences against first sequence
    idx_map = np.asarray([0] * pairs, dtype=np.uint32)
    idy_map = np.asarray(list(range(pairs)), dtype=np.uint32)
    return seq_block, idx_map, idy_map, expected

def setup_test_M2_triangle():
    seq_block = []

    for i in range(1000):
        seq_block.append( ''.join(random.choice(list(DEFAULT_ALPHABET.alphabet_set)) for i in range(10)) )
    assert len(seq_block) > 0

    # Map all sequences against each other
    idx_map, idy_map = np.asarray(np.triu_indices(len(seq_block)), dtype=np.uint32)

    expected = []
    for i in range(len(idx_map)):
        distance = hamming(
            seq_block[idx_map[i]],
            seq_block[idy_map[i]],
            DEFAULT_ALPHABET.equivalent_d,
        )
        expected.append(distance)
    return seq_block, idx_map, idy_map, expected

def do_kernel_test(expected, d):
    assert len(expected) == len(d)
    n_tests = 0
    for i, v in enumerate(d):
        assert v == expected[i]
        n_tests += 1
    return n_tests

def test_e2e_warphog_M2_block11():
    seq_block, idx_map, idy_map, expected = setup_test_M2_triangle()
    pairs = len(expected)

    # Triangular
    block_dim = (1,1,1)
    grid_width = sqrt(pairs)
    grid_dim = ( ceil(grid_width / (block_dim[0])), ceil(grid_width / (block_dim[1])) )

    num_tests = do_kernel(
        expected,
        seq_block,
        pairs,
        idx_map,
        idy_map,
        block_dim,
        grid_dim,
    )
    assert num_tests == (pairs * len(cores.CORES))


def test_e2e_warphog_M2_block44():
    seq_block, idx_map, idy_map, expected = setup_test_M2_triangle()
    pairs = len(expected)

    # Triangular
    block_dim = (4,4,1)
    grid_width = sqrt(pairs)
    grid_dim = ( ceil(grid_width / (block_dim[0])), ceil(grid_width / (block_dim[1])) )

    num_tests = do_kernel(
        expected,
        seq_block,
        pairs,
        idx_map,
        idy_map,
        block_dim,
        grid_dim,
    )
    assert num_tests == (pairs * len(cores.CORES))


def test_e2e_warphog_M1_block11():
    seq_block, idx_map, idy_map, expected = setup_test_M1_100A()
    pairs = len(expected)

    # Rectangular
    block_dim = (1,1,1)
    grid_dim = (ceil(len(idx_map) / block_dim[0]), ceil(len(idy_map) / (block_dim[1])) )

    num_tests = do_kernel(
        expected,
        seq_block,
        pairs,
        idx_map,
        idy_map,
        block_dim,
        grid_dim
    )
    assert num_tests == (pairs * len(cores.CORES))



def test_e2e_warphog_M1_block44():
    seq_block, idx_map, idy_map, expected = setup_test_M1_100A()
    pairs = len(expected)

    # Rectangular
    block_dim = (4,4,1)
    grid_dim = (ceil(len(idx_map) / block_dim[0]), ceil(len(idy_map) / (block_dim[1])) )

    num_tests = do_kernel(
        expected,
        seq_block,
        pairs,
        idx_map,
        idy_map,
        block_dim,
        grid_dim
    )
    assert num_tests == (pairs * len(cores.CORES))
