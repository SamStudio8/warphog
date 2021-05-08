import argparse
import datetime
import sys
from abc import ABC, abstractmethod
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


class WarpHog(ABC):
    def __init__(self, n):
        self.num_seqs = n
        self.num_pairs = self._get_num_pairs()
        self.idx_map, self.idy_map = self._get_thread_map()

    @abstractmethod
    def _get_thread_map(self):
        raise NotImplementedError()

    def get_thread_map(self):
        if self.idx_map is None or self.idy_map is None:
            self.idx_map, self.idy_map = self._get_thread_map()
        else:
            return self.idx_map, self.idy_map

    @abstractmethod
    def _get_num_pairs(self):
        raise NotImplementedError()

    def get_num_pairs(self):
        if not self.num_pairs:
            self.num_pairs = self._get_num_pairs()
        else:
            return self.num_pairs

    def output_tsv(self, d):
        for i in range(self.num_pairs):
            sys.stdout.write('\t'.join([str(x) for x in [
                self.idx_map[i],
                self.idy_map[i],
                d[i],
            ]]) + '\n')

class TriangularWarpHog(WarpHog):

    def _get_thread_map(self):
        return np.asarray(np.triu_indices(self.num_seqs), dtype=np.uint16)

    def _get_num_pairs(self):
        return ceil(( self.num_seqs * (self.num_seqs + 1) ) / 2)

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
    num_seqs = fa_loader.get_count()
    print("huge block loaded: (%d, %d)" % (num_seqs, fa_loader.get_length()))

    hog = TriangularWarpHog(n=num_seqs)

    # Send data block to GPU
    msa_char_block = np.frombuffer("".join(seq_block).encode(), dtype=np.byte)
    msa_gpu = gpuarray.to_gpu(msa_char_block)

    # Init return array and send to GPU
    d = np.zeros(hog.get_num_pairs(), dtype=np.uint16)
    #d_gpu = cuda.mem_alloc(d.nbytes)
    #cuda.memcpy_htod(d_gpu, d)
    d_gpu = gpuarray.to_gpu(d)
    print("huge boi on gpu")

    # Generate triangle map and send to GPU
    idx_map, idy_map = hog.get_thread_map()
    idx_map_gpu = gpuarray.to_gpu(idx_map)
    idy_map_gpu = gpuarray.to_gpu(idy_map)

    # Init GPU grid
    THREADS_PER_BLOCK_X = 4
    THREADS_PER_BLOCK_Y = 4
    PAIRS_PER_THREAD = 1 # TODO busted lel

    block=(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1)
    grid_width = sqrt(hog.get_num_pairs())
    print( (grid_width / (PAIRS_PER_THREAD * THREADS_PER_BLOCK_X)) * (grid_width / (THREADS_PER_BLOCK_Y)) * THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y)
    grid=( ceil(grid_width / (PAIRS_PER_THREAD * THREADS_PER_BLOCK_X)), ceil(grid_width / (THREADS_PER_BLOCK_Y)) )
    print(block)
    print(grid)

    print("THREAD COUNT %d" % (block[0]*block[1]*grid[0]*grid[1]))
    print("MAPS LEN", len(idx_map), len(idy_map))

    # Hog the warps
    start = datetime.datetime.now()

    kernel = KERNELS[args.kernel]()
    kernel.prepare_kernel(alphabet=alphabet)
    kernel = kernel.get_compiled_kernel()

    kernel(
        msa_gpu,
        np.int32(num_seqs),
        np.int32(fa_loader.get_length()), # msa stride
        d_gpu,
        np.int32(PAIRS_PER_THREAD),
        np.uint32(hog.get_num_pairs()),
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

    num_pairs = hog.get_num_pairs()
    print("%d non-zero edit distances found (%.2f%%)" % (s, s/num_pairs*100.0))
    print("%.2fM sequence comparions / s" % ( (num_pairs / delta.total_seconds())/ 1e6) )
    total_bases = fa_loader.get_length() * num_pairs
    print("%.2fB base comparions / s" % ( (total_bases / delta.total_seconds()) / 1e9 ))

    hog.output_tsv(d)
