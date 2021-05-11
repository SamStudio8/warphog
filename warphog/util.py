import numpy as np

default_pairs = [
    ('N', 'A'),
    ('N', 'C'),
    ('N', 'T'),
    ('N', 'G'),

    ('T', 'U'),

    ('R', 'A'),
    ('R', 'G'),

    ('Y', 'C'),
    ('Y', 'T'),

    ('S', 'G'),
    ('S', 'C'),

    ('W', 'A'),
    ('W', 'T'),

    ('K', 'G'),
    ('K', 'T'),

    ('M', 'A'),
    ('M', 'C'),

    ('B', 'C'),
    ('B', 'G'),
    ('B', 'T'),

    ('D', 'A'),
    ('D', 'G'),
    ('D', 'T'),

    ('H', 'A'),
    ('H', 'C'),
    ('H', 'T'),

    ('V', 'A'),
    ('V', 'C'),
    ('V', 'G'),

    ('-', '-'),
    ('?', '?'),
]

class Alphabet(object):
    def __init__(self, equivalent_tuples):
        self.equivalent_tuples = equivalent_tuples
        self.alphabet_set = self._alphabet_set_from_pairs(equivalent_tuples)
        self.alphabet = sorted(list(self.alphabet_set))
        self.alphabet_lookup = { base:i for i, base in enumerate(self.alphabet) }
        self.alphabet_matrix = self._alphabet_matrix_from_pairs(equivalent_tuples, self.alphabet_lookup)
        self.alphabet_len = len(self.alphabet)
        self.equivalent_d = self._alphabet_d_from_pairs(equivalent_tuples)

        self.alphabet_ord_list = self._alphabet_ord_list_from_alphabet(self.alphabet)

    def lookup(self, b):
        return self.alphabet_lookup[b]

    def lookup_ord(self, b):
        return self.alphabet_ord_list[ord(b)]

    def _alphabet_set_from_pairs(self, pairs):
        alphabet = set([])
        for pair in pairs:
            alphabet.add(pair[0])
            alphabet.add(pair[1])
        return alphabet

    def _alphabet_d_from_pairs(self, pairs):
        d = {}
        for pair in pairs:
            a = pair[0]
            b = pair[1]

            if a not in d:
                d[a] = set([])
            if b not in d:
                d[b] = set([])

            d[a].add(b)
            d[b].add(a)
            d[a].add(a)
            d[b].add(b)
        return d

    def _alphabet_matrix_from_pairs(self, pairs, lookup):
        alphabet_len = len(lookup)
        alphabet_matrix = np.ones( (alphabet_len, alphabet_len) )

        for pair in pairs:
            a = lookup[pair[0]]
            b = lookup[pair[1]]

            alphabet_matrix[a][a] = 0
            alphabet_matrix[a][b] = 0
            alphabet_matrix[b][a] = 0
            alphabet_matrix[b][b] = 0

        return alphabet_matrix


    def _alphabet_ord_list_from_alphabet(self, alphabet):
        ord_lookup = { ord(base):i for i, base in enumerate(alphabet) }

        ord_list = np.zeros(max( [ord(base)+1 for base in alphabet] ), dtype=np.int8)
        ord_list.fill(-1)
        for base in alphabet:
            ord_list[ord(base)] = ord_lookup[ord(base)]
        return ord_list


DEFAULT_ALPHABET = Alphabet(default_pairs)

