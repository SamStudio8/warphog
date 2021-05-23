from abc import ABC, abstractmethod

class BaseConverter(ABC):

    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.join_ch = ''

    @abstractmethod
    def convert_base(self, b):
        raise NotImplementedError()
    def convert_seq(self, seq):
        a = []
        for base in seq:
            a.append(self.convert_base(base))
        return a

class NoBaseConverter(BaseConverter):
    def convert_base(self, b):
        return b
    def convert_seq(self, seq):
        return seq


class ListBaseConverter(BaseConverter):
    def convert_base(self, b):
        return self.alphabet.lookup(b)
    def convert_seq(self, seq):
        a = []
        for base in seq:
            a.append( self.alphabet.lookup(base) )
        return a

class OrdBaseConverter(BaseConverter):
    def convert_base(self, b):
        return self.alphabet.lookup_ord(b)

    def convert_seq(self, seq):
        a = []
        for base in seq:
            a.append(self.alphabet.lookup_ord(base))
        return a

class BytesConverter(BaseConverter):
    def __init__(self, alphabet):
        super().__init__(alphabet)
        self.join_ch = b''

    def convert_base(self, b):
        raise NotImplementedError()
    def convert_seq(self, seq):
        return seq.encode()


ENCODERS = {
    "none": NoBaseConverter,
    "bytes": BytesConverter,
}
