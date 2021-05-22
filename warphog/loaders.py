from abc import ABC, abstractmethod

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
    def get_block(self, target_n=1):
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

    def get_block(self, target_n=1):
        line = self.handle.readline()
        curr_block_size = 0
        names = []
        seqs = []
        tells = []
        while line:
            if line[0] == '>':
                tells.append(self.handle.tell() - len(line))
                name = line.strip()

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

    def get_block(self, target_n=1):
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
