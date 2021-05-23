from abc import ABC, abstractmethod
from multiprocessing import Process, Queue, Array
from math import ceil
import sys
import os
import ctypes
import numpy as np

class FastaLoader(ABC):

    def __init__(self, fasta=None, bc=None, offset=0, seek_offset=0, **kwargs):
        self.count = 0
        self.fasta = fasta
        self.base_converter = bc

        self.idx_offset = offset
        self.names_to_idx = {}

        self.names = []

        self.seek_offset = seek_offset #TODO Move to readers that can use it...

        self.seq_len = None

        if not bc:
            raise Exception("You must specify a base converter to the FastaLoader")


    def add_seq(self, name, seq):
        self.names.append(name)
        self.count += 1
        if self.seq_len is None:
            self.seq_len = len(seq)
        else:
            if len(seq) != self.seq_len:
                raise Exception("sequences must be same length")
        return self.base_converter.convert_seq(seq)

    @abstractmethod
    def get_block(self, target_n=-1):
        raise NotImplementedError()
    def get_length(self):
        return self.seq_len
    def get_count(self):
        return self.count


class TrivialFastaLoader(FastaLoader):
    # Passing large sequences through the multiprocessing queue seemed slow.
    # Additionally, we are limited by the speed of ceph, but multiple file
    # handlers can communuicate with different mds independently, circumventing
    # read speed limitations -- so each process gets their own file handle to --target.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handle = open(self.fasta)
        self.handle.seek(self.seek_offset)
        self.first = True
        self.tell = self.handle.tell()

    #TODO Does not match the interface for HengFastaLoader
    def get_block(self, target_n=-1):
        line = self.handle.readline()
        curr_block_size = 0
        names = []
        seqs = []
        tells = []
        while line:
            if line[0] == '>':
                tells.append(self.handle.tell() - len(line))
                name = line[1:-1]

                if self.first:
                    self.first = False
            else:
                if self.first:
                    # If the first line in the block does not start >
                    # ignore it, the previous block will pick it up
                    pass
                else:
                    # Must be the sequence instead
                    seq = line.strip()
                    names.append(name)
                    seq = self.add_seq(name, seq)
                    seqs.append(seq)
                    curr_block_size += 1

            if curr_block_size >= target_n and target_n > 0:
                return tells, names, seqs

            line = self.handle.readline()

        return tells, names, seqs

class StripedFastaLoaderGroup(FastaLoader):

    def get_block(self):
        raise NotImplementedError()

    def get_read(self):
        """
        def read(out_q, loader, target_n, block_end):
            block_names = []
            block_block = None
            tells, names, seqs = loader.get_block(target_n=1)
            done = False
            while names and not done:
                for i, name in enumerate(names):
                    if tells[i] > block_end:
                        # Line starts after end of block, should be picked up by
                        # next block
                        sys.stderr.write("[NOTE] Cowardly leaving block\n")
                        done = True
                        break

                    #if not block_block:
                    #    block_block = seqs[i]
                    #else:
                    #    block_block += seqs[i]
                    #block_names.append(name)

                    #out_q.put({
                    #    "names": name,
                    #    "block": seqs[i],
                    #    "l": loader.get_length(),
                    #})
                tells, names, seqs = loader.get_block(target_n=1)
            out_q.put(None)
        """

        def read(out_q, loader, target_n, block_end, block_i, block_len_q, data_matrix):
            # Phase 1 - count tells
            block_tells = 0
            tells, names, seqs = loader.get_block(target_n=1)
            done = False

            while names and not done:
                for i, name in enumerate(names):
                    if tells[i] > block_end:
                        # Line starts after end of block, should be picked up by
                        # next block
                        sys.stderr.write("[NOTE] Cowardly leaving block\n")
                        done = True
                        break
                    block_tells += 1
                tells, names, seqs = loader.get_block(target_n=1)
            out_q.put({
                "block_i": block_i,
                "block_len": block_tells,
            })
            loader.handle.seek(loader.seek_offset) # move pointer back and wait for instruction
            loader.first = True


            # Wait for instruction
            block_len_dat = block_len_q.get()
            block_offset = block_len_dat[block_i]
            sys.stderr.write("[NOTE] block %d starting with offset %d\n" % (block_i, block_offset))

            tells, names, seqs = loader.get_block(target_n=1)
            done = False
            count = 0
            while names and not done:
                for i, name in enumerate(names):
                    if tells[i] > block_end:
                        # Line starts after end of block, should be picked up by
                        # next block
                        sys.stderr.write("[NOTE] Cowardly leaving block\n")
                        done = True
                        break
                    data_matrix[count + block_offset] = np.array([seqs[i].encode()]).view(np.uint8)
                    count += 1
                tells, names, seqs = loader.get_block(target_n=1)

        return read

    def get_gather(self):
        """
        def gather(block_q, out_q, n_workers):
            ret = {
                "names": [],
                "block": None,
                "l": 0,
            }

            working_workers = n_workers
            while working_workers > 0:
                work = out_q.get()
                if not work:
                    working_workers -= 1
                    sys.stderr.write("[NOTE] %d workers remaining\n" % working_workers)
                else:
                    ret["names"].extend(work["names"])

                    if not ret["block"]:
                        ret["block"] = work["block"]
                    else:
                        ret["block"] += work["block"]

                    if not ret["l"]:
                        ret["l"] = work["l"]
                    else:
                        if ret["l"] != work["l"]:
                            raise Exception("mismatched l")
            block_q.put(ret)
        """

        def gather(block_q, out_q, n_workers, block_len_q):
            block_lens = [None] * n_workers

            # Phase 1
            working_workers = n_workers
            while working_workers > 0:
                work = out_q.get()
                working_workers -= 1

                block_lens[work["block_i"]] = work["block_len"]

                sys.stderr.write("[NOTE] %d workers remaining\n" % working_workers)

            # Phase 2
            block_offsets = [None] * n_workers
            block_offset_sum = 0
            for i in range(n_workers):
                block_offsets[i] = block_offset_sum
                block_offset_sum += block_lens[i]

            for i in range(n_workers):
                block_len_q.put(block_offsets)

        return gather

    def read_block(self, n_procs, loader, **kwargs):
        sys.stderr.write(f"[NOTE] Striped read FASTA with {n_procs} processes\n")

        dim_x = 557927
        dim_y = 29903

        # Init data block
        data_matrix = np.frombuffer(
                Array(ctypes.c_int8, dim_x * dim_y, lock = False),
                dtype=ctypes.c_int8,
        )
        data_matrix = data_matrix.reshape(dim_x, dim_y)

        processes = []
        out_q = Queue()
        block_len_q = Queue()
        block_q = Queue()

        # Determine --target chunks
        target_size = os.path.getsize(self.fasta)
        block_size = ceil( target_size / n_procs )
        block_start = 0

        # Add an extra process for gathering block
        p = Process(target=self.get_gather(), args=(
            block_q,
            out_q,
            n_procs,
            block_len_q,
        ))
        processes.append(p)

        for proc_no in range(n_procs):
            p = Process(target=self.get_read(), args=(
                out_q,
                TrivialFastaLoader(fasta=self.fasta, bc=self.base_converter, seek_offset=block_start, offset=kwargs.get("offset")), #todo init inside kernel?
                -1,
                block_start+block_size,
                proc_no,
                block_len_q,
                data_matrix,
            ))
            processes.append(p)

            block_start += block_size

        # Engage
        for p in processes:
            p.start()

        # Block
        for p in processes:
            p.join()

        return seq_block["names"], seq_block["block"]


class HengFastaLoader(FastaLoader):
    # thanks heng
    def readfq(self, fp): # this is a generator function
        last = None # this is a buffer keeping the last unprocessed line
        while True: # mimic closure; is it a bad idea?
            if not last: # the first record or a record following a fastq
                for l in fp: # search for the start of the next record
                    if l[0] in '>@': # fasta/q header line
                        last = l[:-1] # save this line
                        break
            if not last: break
            name, seqs, last = last[1:].partition(" ")[0], [], None
            for l in fp: # read the sequence
                if l[0] in '@+>':
                    last = l[:-1]
                    break
                seqs.append(l[:-1])
            if not last or last[0] != '+': # this is a fasta record
                yield name, ''.join(seqs), None # yield a fasta record
                if not last: break
            else: # this is a fastq record
                seq, leng, seqs = ''.join(seqs), 0, []
                for l in fp: # read the quality
                    seqs.append(l[:-1])
                    leng += len(l) - 1
                    if leng >= len(seq): # have read enough quality
                        last = None
                        yield name, seq, ''.join(seqs); # yield a fastq record
                        break
                if last: # reach EOF before reading enough quality
                    yield name, seq, None # yield a fasta record instead
                    break

    def get_block(self, target_n=-1):
        #msa_char_block = np.zeros( (1, self.get_length()) , dtype=np.int32)
        seq_block = []
        curr_block_size = 0
        for name_i, seq_i, qual_i in self.readfq(open(self.fasta)):
            if self.count % 10000 == 0:
                print(self.count)
            #np.append(msa_char_block, (base_converter.convert_base(base) for base in seq_i))
            #seq_block.append( (base_converter.convert_base(base) for base in seq_i) )
            self.names_to_idx[self.idx_offset + self.count] = name_i
            curr_block_size += 1
            seq = self.add_seq(name_i, seq_i)
            seq_block.append(seq)

            #seq_block.append(seq_i)
            if curr_block_size >= target_n and target_n > 0:
                break
        #return msa_char_block
        return seq_block


LOADERS = {
    "heng": HengFastaLoader,
    "trivial": TrivialFastaLoader,
}
