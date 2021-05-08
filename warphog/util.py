import numpy as np

pairs = [
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
alphabet = set([])
for pair in pairs:
    alphabet.add(pair[0])
    alphabet.add(pair[1])
alphabet.add('-')

alphabet = sorted(list(alphabet))
alphabet_lookup = { base:i for i, base in enumerate(alphabet) }
