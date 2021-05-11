import argparse
import datetime
import sys

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np

from warphog.hogs import HOGS
from warphog.loaders import LOADERS
from warphog.encoders import ENCODERS
from warphog.cuda.kernels import KERNELS
from warphog.util import DEFAULT_ALPHABET

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--loader", choices=LOADERS.keys(), required=True)
    parser.add_argument("--loader-limit", type=int, default=-1)
    parser.add_argument("--encoder", choices=ENCODERS.keys(), required=True)
    parser.add_argument("--kernel", choices=KERNELS.keys(), required=True)
    parser.add_argument("--hog", choices=HOGS.keys(), required=True)
    parser.add_argument("-k", type=int, default=-1)
    parser.add_argument("-o")
    args = parser.parse_args()
    warphog(args)


def init_concrete(alphabet, encoder, loader, loader_limit, fasta):
    if alphabet:
        raise NotImplementedError()
    alphabet = DEFAULT_ALPHABET
    base_converter = ENCODERS[encoder](alphabet=alphabet)
    fa_loader = LOADERS[loader](limit=loader_limit, fasta=fasta, bc=base_converter)
    return alphabet, fa_loader

def warphog(args):
    # Init concrete implementations of components
    alphabet, fa_loader = init_concrete(None, args.encoder, args.loader, args.loader_limit, args.fasta)

    # Load a block of sequences from FASTA
    seq_block = fa_loader.get_block()
    num_seqs = fa_loader.get_count()
    print("huge block loaded: (%d, %d)" % (num_seqs, fa_loader.get_length()))

    # TODO How to set m here
    hog = HOGS[args.hog](n=num_seqs)

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

    print("THREAD COUNT %d" % hog.thread_count)
    print("MAPS LEN", len(idx_map), len(idy_map))

    # Hog the warps
    start = datetime.datetime.now()

    kernel = KERNELS[args.kernel]()
    kernel.prepare_kernel(alphabet=alphabet)
    kernel = kernel.get_compiled_kernel()

    kernel(
        msa_gpu,
        np.uint(num_seqs),
        np.uint32(fa_loader.get_length()), # msa stride
        d_gpu,
        np.int32(hog.pairs_per_thread),
        np.uint(hog.get_num_pairs()),
        idx_map_gpu,
        idy_map_gpu,
        block=hog.block_dim,
        grid=hog.grid_dim,
    )
    print("fetching huge boi from gpu")
    #cuda.memcpy_dtoh(d, d_gpu)
    d_gpu.get(d)

    end = datetime.datetime.now()
    delta = end-start
    print(delta)

    dd = np.zeros((hog.seq_dim_x, hog.seq_dim_y), dtype=np.uint16)
    hog.broadcast_result(d, dd)
    print(dd)
    s = (d > 0).sum()


    num_pairs = hog.get_num_pairs()
    print("%d non-zero edit distances found (%.2f%%)" % (s, s/num_pairs*100.0))
    print("%.2fM sequence comparions / s" % ( (num_pairs / delta.total_seconds())/ 1e6) )
    total_bases = fa_loader.get_length() * num_pairs
    print("%.2fB base comparions / s" % ( (total_bases / delta.total_seconds()) / 1e9 ))

    if args.o:
        start = datetime.datetime.now()
        b_written = hog.output_tsv(d, args.o, k=args.k, names=fa_loader.names_to_idx)
        mb_written = b_written / 1e6
        end = datetime.datetime.now()
        delta = end-start
        print("%.2f GB written in %s (%.2f MB/s)" % (mb_written / 1000, str(delta), mb_written / delta.total_seconds()))
