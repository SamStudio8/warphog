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
from warphog.util import DEFAULT_ALPHABET

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--loader", choices=LOADERS.keys(), required=True)
    parser.add_argument("--loader-limit", type=int)
    parser.add_argument("--encoder", choices=ENCODERS.keys(), required=True)
    parser.add_argument("--kernel", choices=KERNELS.keys(), required=True)
    args = parser.parse_args()
    warphog(args)


class WarpHog(object):
    def __init__(self, n):
        self.num_seqs = n
        self.triangle_size = ceil(( n * (n + 1) ) / 2)
        self.idx_map, self.idy_map = np.asarray(np.triu_indices(n), dtype=np.uint16)

    def output_tsv(self, d):
        for i in range(self.triangle_size):
            sys.stdout.write('\t'.join([str(x) for x in [
                self.idx_map[i],
                self.idy_map[i],
                d[i],
            ]]) + '\n')

def init_concrete(alphabet, encoder, loader, loader_limit, fasta):
    if alphabet:
        raise NotImplementedError()
    alphabet = DEFAULT_ALPHABET
    base_converter = ENCODERS[encoder](alphabet=alphabet)
    fa_loader = LOADERS[loader](n=loader_limit, fasta=fasta, bc=base_converter)
    return alphabet, fa_loader

def warphog(args):
    # Init concrete implementations of components
    alphabet, fa_loader = init_concrete(None, args.encoder, args.loader, args.loader_limit, args.fasta)

    # Load a block of sequences from FASTA
    seq_block = fa_loader.get_block()
    l = fa_loader.get_length()
    num_seqs = fa_loader.get_count()
    print("huge block loaded: (%d, %d)" % (num_seqs, l))

    hog = WarpHog(n=num_seqs)

    # Send data block to GPU
    msa_char_block = np.frombuffer("".join(seq_block).encode(), dtype=np.byte)
    msa_gpu = gpuarray.to_gpu(msa_char_block)

    # Init return array and send to GPU
    triangle_size = ceil(( num_seqs * (num_seqs + 1) ) / 2)
    d = np.zeros(triangle_size, dtype=np.uint16)
    #d_gpu = cuda.mem_alloc(d.nbytes)
    #cuda.memcpy_htod(d_gpu, d)
    d_gpu = gpuarray.to_gpu(d)
    print("huge boi on gpu")

    # Generate triangle map and send to GPU
    idx_map_gpu = gpuarray.to_gpu(hog.idx_map)
    idy_map_gpu = gpuarray.to_gpu(hog.idy_map)

    # Init GPU grid
    THREADS_PER_BLOCK_X = 4
    THREADS_PER_BLOCK_Y = 4
    PAIRS_PER_THREAD = 1 # TODO busted lel

    block=(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1)
    grid_width = sqrt(triangle_size)
    print( (grid_width / (PAIRS_PER_THREAD * THREADS_PER_BLOCK_X)) * (grid_width / (THREADS_PER_BLOCK_Y)) * THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y)
    grid=( ceil(grid_width / (PAIRS_PER_THREAD * THREADS_PER_BLOCK_X)), ceil(grid_width / (THREADS_PER_BLOCK_Y)) )
    print(block)
    print(grid)

    print("THREAD COUNT %d" % (block[0]*block[1]*grid[0]*grid[1]))
    print("MAPS LEN", len(hog.idx_map), len(hog.idy_map))

    # Hog the warps
    start = datetime.datetime.now()

    kernel = KERNELS[args.kernel]()
    kernel.prepare_kernel(alphabet=alphabet)
    kernel = kernel.get_compiled_kernel()

    kernel(
        msa_gpu,
        np.int32(num_seqs),
        np.int32(l), # msa stride
        d_gpu.gpudata,
        np.int32(PAIRS_PER_THREAD),
        np.uint32(hog.triangle_size),
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
    dd[hog.idx_map, hog.idy_map] = d
    print(dd)
    s = (d > 0).sum()

    print("%d non-zero edit distances found (%.2f%%)" % (s, s/triangle_size*100.0))
    print("%.2fM sequence comparions / s" % ( (triangle_size / delta.total_seconds())/ 1e6) )
    total_bases = fa_loader.get_length() * triangle_size
    print("%.2fB base comparions / s" % ( (total_bases / delta.total_seconds()) / 1e9 ))

    hog.output_tsv(d)
