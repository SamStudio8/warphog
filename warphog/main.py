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

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--query")
    parser.add_argument("--loader-limit", type=int, default=-1)
    parser.add_argument("--core", choices=CORES.keys(), required=True, default="warp")
    parser.add_argument("-k", type=int, default=-1)
    parser.add_argument("-t", type=int, default=-1)
    parser.add_argument("-o", default='-')
    args = parser.parse_args()

    warphog(args)

def warphog(args):
    # Init concrete implementations of components
    alphabet = DEFAULT_ALPHABET
    base_converter = ENCODERS["bytes"](alphabet=alphabet) # bytes is the only encoder that works atm anyway

    query_block = []
    query_count = 0
    if args.query:
        query_fa_loader = LOADERS["heng"](fasta=args.query, bc=base_converter)
        query_block = query_fa_loader.get_block()
        query_count = query_fa_loader.get_count()

    if args.core == "warp":
        import pycuda.autoinit
        import pycuda.driver as cuda

    if args.t > 0 and args.core == "warp":
        sys.stderr.write("[NOTE] Threads option -t ignored when using GPU\n")
    if args.t > 0 and args.core == "prewarp" and not args.query:
        sys.stderr.write("[NOTE] Threads option -t ignored when using CPU without --query\n")

    if args.core == "prewarp":
        if args.query:
            sys.stderr.write("[NOTE] Using prewarp-beta mode to speed up --query on CPU\n")

            queries = {}
            for i, name in enumerate(query_fa_loader.names):
                queries[name] = query_block[i]
            warphog_cpu(args, alphabet, queries)
            return

    # Load a block of sequences from FASTA
    fa_loader = LOADERS["heng"](fasta=args.target, bc=base_converter, offset=query_count)
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

    # Commented out partly because I can't be bothered making this work with the
    # prewarp-beta, but also because it was useless anyway
    #
    #dd = np.zeros((hog.seq_dim_x, hog.seq_dim_y), dtype=np.uint16)
    #hog.broadcast_result(d, dd)
    #print(dd)
    #if dd.shape[0] == dd.shape[1]:
    #    dd = dd + dd.T - np.diag(np.diag(dd))
    #    print(dd)
    #if args.o:
    #    np.save(args.o, dd, allow_pickle=False)

    s = (d > 0).sum()
    num_pairs = hog.get_num_pairs()
    print("%d non-zero edit distances found (%.2f%%) in %s" % (s, s/num_pairs*100.0, str(delta)))
    print("%.2fM sequence comparions / s" % ( (num_pairs / delta.total_seconds())/ 1e6) )
    total_bases = fa_loader.get_length() * num_pairs
    print("%.2fB base comparions / s" % ( (total_bases / delta.total_seconds()) / 1e9 ))

    if args.o != '-':
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


def kernel_fp_hamming(out_q, loader, queries, ord_l, alphabet_matrix, block_start, block_end, block):
    from warphog.kernels.hamming import kernel_wrapb  # pylint: disable=no-name-in-module,import-error

    # Passing large sequences through the multiprocessing queue seemed slow.
    # Additionally, we are limited by the speed of ceph, but multiple file
    # handlers can communuicate with different mds independently, circumventing
    # read speed limitations -- so each process gets their own file handle to --target.
    tells, names, seqs = loader.get_block(target_n=1)
    result_block = {
        "working_time": 0,
        "result": [],
    }
    done = False
    start = datetime.datetime.now()
    count = 0
    while names and not done:
        for i, name in enumerate(names):
            count += 1
            for q_name, q_seq in queries.items():
                #distance = 0
                #seq_a_idx = np.take(ord_l, np.array([q_seq]).view(np.uint8))
                #seq_b_idx = np.take(ord_l, np.array([seq.encode()]).view(np.uint8))
                #distance = np.sum(alphabet_matrix[ seq_a_idx, seq_b_idx ])
                d = kernel_wrapb(q_seq, seqs[i], len(q_seq), ord_l, alphabet_matrix)
                result_block["result"].append({
                    "block": block,
                    "qname": q_name,
                    "tname": name,
                    "distance": d,
                })
        tells, names, seqs = loader.get_block(target_n=1)
    end = datetime.datetime.now()

    # Adding to Multiprocessing.Queue is awfully slow for many small objects,
    # so cache up and blow the entire result_block at the thing instead
    #TODO Probably a compromise to be made here between pushing as much as
    # possible but not maintaining huge lists to append on
    result_block["working_time"] = end-start
    result_block["loader_len"] = loader.get_length()
    out_q.put(result_block)

def kernel_fp_output(out_fp, out_q, n_workers, k):
    s = 0
    num_pairs = 0
    b_written = 0
    writing_time = datetime.timedelta(0)
    working_time = datetime.timedelta(0)
    l = None

    wall_start = datetime.datetime.now()

    working_workers = n_workers
    while working_workers > 0:
        work = out_q.get()
        working_workers -= 1

        if not l:
            l = work["loader_len"]
        start = datetime.datetime.now()
        for res in work["result"]:
            num_pairs += 1
            if res["distance"] > 0:
                s += 1

            if k > -1:
                if res["distance"] > k:
                    continue
                
            b_written += out_fp.write('\t'.join([
                res["qname"],
                res["tname"],
                str(res["distance"]),
            ]) + '\n')
        end = datetime.datetime.now()
        sys.stderr.write("[NOTE] %d workers remaining\n" % working_workers)
        writing_time += (end-start)

    wall_end = datetime.datetime.now()
    wall_delta = wall_end - wall_start

    print("%d non-zero edit distances found (%.2f%%) in %s" % (s, s/num_pairs*100.0, str(wall_delta)))
    print("%.2fM sequence comparions / s" % ( (num_pairs / wall_delta.total_seconds())/ 1e6) )
    total_bases = l * num_pairs
    print("%.2fB base comparions / s" % ( (total_bases / wall_delta.total_seconds()) / 1e9 ))

    mb_written = b_written / 1e6
    print("%.2f GB written in %s (%.2f MB/s)" % (mb_written / 1000, str(writing_time), mb_written / writing_time.total_seconds()))

def warphog_cpu(args, alphabet, queries):
    from multiprocessing import Process, Queue, Array, cpu_count
    
    from math import ceil
    sys.stderr.write(f"[NOTE] {len(queries)} queries loaded\n")

    n_procs = cpu_count()
    if args.t > 0:
        n_procs = args.t
    sys.stderr.write(f"[NOTE] {n_procs} processes\n")

    processes = []
    
    out_q = Queue()

    ord_l = np.asarray(alphabet.alphabet_ord_list, dtype=np.int8)
    alphabet_matrix = alphabet.alphabet_matrix

    # Determine --target chunks
    target_size = os.path.getsize(args.target)
    block_size = ceil( target_size / n_procs )
    block_start = 0

    # Add an extra process for writing out results
    if args.o == '-':
        out_fp = sys.stdout
    else:
        out_fp = open(args.o, 'w')

    p = Process(target=kernel_fp_output, args=(
        out_fp,
        out_q,
        n_procs,
        args.k,
    ))
    processes.append(p)

    for proc_no in range(n_procs):
        p = Process(target=kernel_fp_hamming, args=(
            out_q,
            LOADERS["trivial"](fasta=args.target, bc=ENCODERS["bytes"](alphabet=alphabet), seek_offset=block_start, seek_end=block_start+block_size), #TODO Init inside kernel?
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

    if args.o != '-':
        out_fp.close()

