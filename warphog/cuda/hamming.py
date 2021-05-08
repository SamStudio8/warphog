import importlib.resources as pkg_resources
import numpy as np
import sys

from pycuda.compiler import SourceModule as cpp

# TODO Remove assumption on NxN and use NxM
# TODO fucking tests ffs
# TODO Guard python from sending arrays with elements larger than unsigned short

from warphog.util import pairs

def prep_cu(alphabet, alphabet_lookup):

    alphabet_len = len(alphabet)
    alphabet_ord_lookup = { ord(base):i for i, base in enumerate(alphabet) }
    alphabet_ord_lookup_l = np.zeros(max( [ord(base)+1 for base in alphabet] ), dtype=np.int8)
    alphabet_ord_lookup_l.fill(-1)
    print(len(alphabet_ord_lookup_l))
    for base in alphabet:
        alphabet_ord_lookup_l[ord(base)] = alphabet_ord_lookup[ord(base)]

    alphabet_matrix = np.ones((alphabet_len, alphabet_len))

    for pair in pairs:
        a = alphabet_lookup[pair[0]]
        b = alphabet_lookup[pair[1]]

        alphabet_matrix[a][a] = 0
        alphabet_matrix[a][b] = 0
        alphabet_matrix[b][a] = 0
        alphabet_matrix[b][b] = 0


    alphabet_matrix_cpp = []

    alphabet_matrix_cpp.append("__device__ int ord_lookup[%d] = {%s};" % (len(alphabet_ord_lookup_l), ','.join([str(x) for x in alphabet_ord_lookup_l])))

    alphabet_matrix_cpp.append("__device__ int equivalent_lookup[%d][%d] = {" % (alphabet_len, alphabet_len))
    print(alphabet_matrix)
    for row in alphabet_matrix:
        str_row = ','.join([str(int(x)) for x in row])
        alphabet_matrix_cpp.append('{ %s },' % str_row)
    alphabet_matrix_cpp.append("};")

    alphabet_matrix_cpp.append("__device__ const char* alphabet = \"%s\";" % ''.join(alphabet))

    print('\n'.join(alphabet_matrix_cpp))

    # Luckily for me, we can use Hamming distance over Levenshtein distance as
    # we have an MSA already. All strings are the same size. Our MSA drops insertions
    # and deletions are marked as a symbol.
    # Presumably we could do something clever and turn the string into integers to do
    # some magic bit shifting. Regardless, CUDA will be faster than dumping it on CPU.
    #
    hamming_cu = pkg_resources.read_text(sys.modules['.'.join(__name__.split('.')[:-1])], 'hamming.cu')
    return cpp('\n'.join(alphabet_matrix_cpp) + hamming_cu).get_function("hamming_distance")


