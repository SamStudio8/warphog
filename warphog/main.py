import argparse
import datetime
import os
import sys

import numpy as np

from warphog import hogs
from warphog.cores import CORES
from warphog.encoders import ENCODERS
from warphog.loaders import LOADERS
from warphog.util import DEFAULT_ALPHABET

CORES["prewarp-beta"] = None

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--query")
    parser.add_argument("--loader", choices=LOADERS.keys(), required=True)
    parser.add_argument("--loader-limit", type=int, default=-1)
    parser.add_argument("--encoder", choices=ENCODERS.keys(), required=True)
    parser.add_argument("--core", choices=CORES.keys(), required=True, default="warp")
    parser.add_argument("-k", type=int, default=-1)
    parser.add_argument("-t", type=int, default=-1)
    parser.add_argument("-o")
    args = parser.parse_args()

    warphog(args)

def warphog(args):
    # Init concrete implementations of components
    alphabet = DEFAULT_ALPHABET
    base_converter = ENCODERS[args.encoder](alphabet=alphabet)

    query_block = []
    query_count = 0
    if args.query:
        query_fa_loader = LOADERS[args.loader](fasta=args.query, bc=base_converter)
        query_block = query_fa_loader.get_block()
        query_count = query_fa_loader.get_count()

    if args.core == "warp":
        import pycuda.autoinit
        import pycuda.driver as cuda
    elif args.core == "prewarp-beta":
        if not args.query:
            raise Exception("prewarp-beta only supports modes with --query")
        if args.encoder != "bytes":
            raise Exception("must use --encoder bytes with prewarp-beta")
        queries = {}
        for i, name in enumerate(query_fa_loader.names):
            queries = {name: query_block[i]}
        warphog_cpu(args, alphabet, queries)
        return

    # Load a block of sequences from FASTA
    fa_loader = LOADERS[args.loader](fasta=args.target, bc=base_converter, offset=query_count)
    seq_block = fa_loader.get_block(target_n=args.loader_limit)
    num_seqs = fa_loader.get_count()
    print("huge block loaded: (%d, %d)" % (num_seqs, fa_loader.get_length()))

    if args.query:
        hog = hogs.RectangularWarpHog(n=num_seqs + query_count, m=query_count)
    else:
        hog = hogs.TriangularWarpHog(n=num_seqs)

    core = CORES[args.core](query_block + seq_block, alphabet)

    # Init return array and send to GPU
    d = np.zeros(hog.get_num_pairs(), dtype=np.uint16)
    core.put_d(d)

    print("huge boi")

    # Generate triangle map and send to GPU
    idx_map, idy_map = hog.get_thread_map()
    core.put_maps(idx_map, idy_map)

    print("PAIR COUNT %d" % hog.get_num_pairs())
    print("THREAD COUNT %d" % hog.thread_count)
    print("MAPS LEN", len(idx_map), len(idy_map))

    # Hog the warps
    start = datetime.datetime.now()

    #core.engage(hog.get_num_pairs, idx_map_gpu, idy_map_gpu, block_dim=hog.block_dim, grid_dim=hog.grid_dim)
    core.kernel.engage(
        core.data_block,
        np.uint(hog.num_seqs),
        np.uint32(fa_loader.get_length()), # msa stride
        core.result_arr,
        np.int32(hog.pairs_per_thread),
        np.uint(hog.get_num_pairs()),
        core.idx_map,
        core.idy_map,
        block=hog.block_dim,
        grid=hog.grid_dim,
    )
    print("fetching huge boi")
    core.get_d(d)

    end = datetime.datetime.now()
    delta = end-start
    print(delta)

    dd = np.zeros((hog.seq_dim_x, hog.seq_dim_y), dtype=np.uint16)
    hog.broadcast_result(d, dd)

    print(dd)
    if dd.shape[0] == dd.shape[1]:
        dd = dd + dd.T - np.diag(np.diag(dd))
        print(dd)
    if args.o:
        np.save(args.o, dd, allow_pickle=False)


    s = (d > 0).sum()
    num_pairs = hog.get_num_pairs()
    print("%d non-zero edit distances found (%.2f%%)" % (s, s/num_pairs*100.0))
    print("%.2fM sequence comparions / s" % ( (num_pairs / delta.total_seconds())/ 1e6) )
    total_bases = fa_loader.get_length() * num_pairs
    print("%.2fB base comparions / s" % ( (total_bases / delta.total_seconds()) / 1e9 ))

    if args.o:
        start = datetime.datetime.now()

        names = {}
        if args.query:
            names.update( query_fa_loader.names_to_idx )
        names.update( fa_loader.names_to_idx )

        b_written = hog.output_tsv(d, args.o, k=args.k, names=names)
        mb_written = b_written / 1e6
        end = datetime.datetime.now()
        delta = end-start
        print("%.2f GB written in %s (%.2f MB/s)" % (mb_written / 1000, str(delta), mb_written / delta.total_seconds()))

def warphog_cpu(args, alphabet, queries):
    from multiprocessing import Process, Queue, Array, cpu_count
    from warphog.kernels.hamming import kernel_wrapb  # pylint: disable=no-name-in-module,import-error
    from math import ceil
    sys.stderr.write(f"[NOTE] {len(queries)} queries loaded\n")

    n_procs = cpu_count()
    if args.t > 0:
        n_procs = args.t
    sys.stderr.write(f"[NOTE] {n_procs} processes\n")

    processes = []

    def kernel_fp_hamming(loader, queries, ord_l, alphabet_matrix, block_start, block_end, block):
        # Passing large sequences through the multiprocessing queue seemed slow.
        # Additionally, we are limited by the speed of ceph, but multiple file
        # handlers can communuicate with different mds independently, circumventing
        # read speed limitations -- so each process gets their own file handle to --target.
        tells, names, seqs = loader.get_block()
        while names:
            for i, name in enumerate(names):
                if tells[i] > block_end:
                    # Line starts after end of block, should be picked up by
                    # next block
                    print("[NOTE] Cowardly leaving block")
                    return

                for q_name, q_seq in queries.items():
                    #distance = 0
                    #seq_a_idx = np.take(ord_l, np.array([q_seq]).view(np.uint8))
                    #seq_b_idx = np.take(ord_l, np.array([seq.encode()]).view(np.uint8))
                    #distance = np.sum(alphabet_matrix[ seq_a_idx, seq_b_idx ])
                    print(block, q_name, name, kernel_wrapb(q_seq, seqs[i], len(q_seq), ord_l, alphabet_matrix))
            tells, names, seqs = loader.get_block()

    def kernel_output():
        pass

    ord_l = np.asarray(alphabet.alphabet_ord_list, dtype=np.int8)
    alphabet_matrix = alphabet.alphabet_matrix

    # Determine --target chunks
    target_size = os.path.getsize(args.target)
    block_size = ceil( target_size / n_procs )
    block_start = 0

    for proc_no in range(n_procs):
        p = Process(target=kernel_fp_hamming, args=(
            LOADERS["trivial"](fasta=args.target, bc=ENCODERS["bytes"](alphabet=alphabet), seek_offset=block_start), #TODO Init inside kernel?
            queries,
            ord_l,
            alphabet_matrix,
            block_start,
            block_start+block_size,
            proc_no,
        ))
        processes.append(p)

        block_start += block_size

    # Engage
    for p in processes:
        p.start()

    # Block
    for p in processes:
        p.join()

