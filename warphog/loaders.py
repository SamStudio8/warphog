from abc import ABC, abstractmethod

class FastaLoader(ABC):

    def __init__(self, fasta=None, limit=-1, bc=None):
        self.count = 0
        self.fasta = fasta
        self.limit = limit
        self.base_converter = bc
        if not bc:
            raise Exception("You must specify a base converter to the FastaLoader")

    @abstractmethod
    def get_block(self):
        raise NotImplementedError()
    @abstractmethod
    def get_length(self):
        raise NotImplementedError()
    def get_count(self):
        return self.count

class TestFastaLoader(FastaLoader):
    def get_block(self):
        seq_block = [
            "AAAAAAAAAA",
            "AAAAAAAAAC",
            "AAAAAAAACC",
            "AAAAAAACCC",
            "AAAAAACCCC",
            "AAAAACCCCC",
            "AAAACCCCCC",
            "AAACCCCCCC",
            "AACCCCCCCC",
            "ACCCCCCCCC",
            "CCCCCCCCCC",
            "NGGGGGGGGA",
            "NNGGGGGGAA",
            "NNNGGGGAAA",
            "NNNNGGAAAA",
            "NNNNNAAAAA",
            "NNNNNNGGGC",
            "NNNNNNNCCC",
            "NNNNNNNNAA",
            "NNNNNNNNNT",
        ]
        self.count = len(seq_block)
        return seq_block

    def get_length(self):
        return 10

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

    def get_block(self):
        #msa_char_block = np.zeros( (1, self.get_length()) , dtype=np.int32)
        seq_block = []
        curr_seq_num_i = 0
        for name_i, seq_i, qual_i in self.readfq(open(self.fasta)):
            if curr_seq_num_i % 10000 == 0:
                print(curr_seq_num_i)
            curr_seq_num_i += 1
            #np.append(msa_char_block, (base_converter.convert_base(base) for base in seq_i))
            #seq_block.append( (base_converter.convert_base(base) for base in seq_i) )
            seq_block.append( self.base_converter.convert_seq(seq_i) )
            self.count += 1

            #seq_block.append(seq_i)
            if curr_seq_num_i >= self.limit and self.limit > 0:
                break
        #return msa_char_block
        return seq_block

    def get_length(self):
        return 29903

LOADERS = {
    "heng": HengFastaLoader,
    "test": TestFastaLoader,
}
