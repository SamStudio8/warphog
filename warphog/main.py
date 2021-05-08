import argparse
import datetime
import sys
from math import ceil, sqrt

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np

from warphog.loaders import LOADERS
from warphog.encoders import ENCODERS
from warphog.cuda.kernels import KERNELS
from warphog.util import alphabet, alphabet_lookup

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--loader", choices=LOADERS.keys(), required=True)
    parser.add_argument("--loader-limit", type=int)
    parser.add_argument("--encoder", choices=ENCODERS.keys(), required=True)
    parser.add_argument("--kernel", choices=KERNELS.keys(), required=True)
    args = parser.parse_args()
    warphog(args)


def warphog(args):
    # Read sequences into block
    base_converter = ENCODERS[args.encoder](alphabet=alphabet, lookup=alphabet_lookup)

    fa_loader = LOADERS[args.loader](n=args.loader_limit, fasta=args.fasta, bc=base_converter)
    seq_block = fa_loader.get_block()
    l = fa_loader.get_length()
    print("huge block loaded")

    msa_char_block = np.frombuffer("".join(seq_block).encode(), dtype=np.byte)
    msa_gpu = gpuarray.to_gpu(msa_char_block)

    num_seqs = fa_loader.get_count()
    print(num_seqs, l)

    d = np.zeros(ceil( ((num_seqs * (num_seqs+1)) / 2 )), dtype=np.uint16)
    #d_gpu = cuda.mem_alloc(d.nbytes)
    #cuda.memcpy_htod(d_gpu, d)
    d_gpu = gpuarray.to_gpu(d)
    print("huge boi on gpu")

    THREADS_PER_BLOCK_X = 4
    THREADS_PER_BLOCK_Y = 4
    PAIRS_PER_THREAD = 1 # TODO busted lel

    block=(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1)
    #grid=( ceil(num_seqs / (PAIRS_PER_THREAD * THREADS_PER_BLOCK)), ceil(num_seqs / (THREADS_PER_BLOCK)) )
    grid_width = (sqrt(( num_seqs * (num_seqs + 1) ) / 2))
    print( (grid_width / (PAIRS_PER_THREAD * THREADS_PER_BLOCK_X)) * (grid_width / (THREADS_PER_BLOCK_Y)) * THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y)
    grid=( ceil(grid_width / (PAIRS_PER_THREAD * THREADS_PER_BLOCK_X)), ceil(grid_width / (THREADS_PER_BLOCK_Y)) )
    print(block)
    print(grid)

    idx_map, idy_map = np.asarray(np.triu_indices(num_seqs), dtype=np.uint16)
    idx_map_gpu = gpuarray.to_gpu(idx_map)
    idy_map_gpu = gpuarray.to_gpu(idy_map)

    print("THREAD COUNT %d" % (block[0]*block[1]*grid[0]*grid[1]))
    print("MAPS LEN", len(idx_map), len(idy_map))


    start = datetime.datetime.now()

    kernel = KERNELS[args.kernel]()
    kernel.prepare_kernel(alphabet=alphabet, alphabet_lookup=alphabet_lookup)
    kernel = kernel.get_compiled_kernel()

    kernel(
        msa_gpu,
        np.int32(num_seqs),
        np.int32(l), # msa stride
        d_gpu.gpudata,
        np.int32(PAIRS_PER_THREAD),
        np.uint32(len(idx_map)),
        idx_map_gpu,
        idy_map_gpu,
        block=block,
        grid=grid,
    )
    print("fetching huge boi from gpu")
    #cuda.memcpy_dtoh(d, d_gpu)
    d_gpu.get(d)

    end = datetime.datetime.now()
    delta = end-start
    print(delta)

    print(d)
    dd = np.zeros((num_seqs, num_seqs), dtype=np.uint16)
    dd[idx_map, idy_map] = d
    print(dd)
    s = (d > 0).sum()

    print("%d non-zero edit distances found" % s)
    print("%.2fM sequence comparions / s" % ((s / delta.total_seconds())/1e6))
    print("%.2fB base comparions / s" % ( (fa_loader.get_length() * (s / delta.total_seconds()))/1e9))



