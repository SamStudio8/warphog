from abc import ABC, abstractmethod

class BaseConverter(ABC):

    def __init__(self, alphabet, lookup):
        self.alphabet = alphabet
        self.alphabet_lookup = lookup

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
        return self.alphabet_lookup[b]
    def convert_seq(self, seq):
        a = []
        for base in seq:
            a.append( self.alphabet_lookup[base] )
        return a

class OrdBaseConverter(BaseConverter):
    def convert_base(self, b):
        return alphabet_ord_lookup_l[ord(b)]

    def convert_seq(self, seq):
        a = []
        for base in seq:
            a.append(alphabet_ord_lookup_l[ord(base)])
        return a

class TomsLuckyBaseConverter(BaseConverter):
    def convert_base(self, b):
        raise NotImplementedError()

ENCODERS = {
    "none": NoBaseConverter,

}
