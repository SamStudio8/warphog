import sys
import numpy as np
from abc import ABC, abstractmethod

import importlib.resources as pkg_resources

from pycuda.compiler import SourceModule as cpp

class KernelPrepper(ABC):

    def __init__(self):
        self.f = None
        self.pre_kernel = []
        self.kernel = []
        self.post_kernel = []

    @abstractmethod
    def prepare_kernel(self, **kwargs):
        raise NotImplementedError()

    def set_kernel(self, module, cu):
        self.kernel = [pkg_resources.read_text(sys.modules[module], cu)]

    def get_kernel_lines(self):
        return self.pre_kernel + self.kernel + self.post_kernel

    def get_compiled_kernel(self, f=None):
        if not f:
            if not self.f:
                raise Exception("Kernel must define default function name to execute")
            f = self.f
        return cpp('\n'.join(self.get_kernel_lines())).get_function(self.f)

class SamHammingKernelPrepper(KernelPrepper):
    # Luckily for me, we can use Hamming distance over Levenshtein distance as
    # we have an MSA already. All strings are the same size. Our MSA drops insertions
    # and deletions are marked as a symbol.
    # Presumably we could do something clever and turn the string into integers to do
    # some magic bit shifting. Regardless, CUDA will be faster than dumping it on CPU.
    # TODO Remove assumption on NxN and use NxM
    # TODO fucking tests ffs
    # TODO Guard python from sending arrays with elements larger than unsigned short

    def __init__(self):
        super().__init__()
        self.f = "hamming_distance"
        self.set_kernel("warphog.cuda", "hamming.cu")

    def prepare_kernel(self, **kwargs):
        alphabet = kwargs.get("alphabet")

        if not alphabet:
            raise Exception("Kernel missing required kwargs...")

        alphabet_matrix_cpp = []

        alphabet_matrix_cpp.append("__device__ int ord_lookup[%d] = {%s};" % (len(alphabet.alphabet_ord_list), ','.join([str(x) for x in alphabet.alphabet_ord_list])))

        alphabet_matrix_cpp.append("__device__ int equivalent_lookup[%d][%d] = {" % (alphabet.alphabet_len, alphabet.alphabet_len))
        print(alphabet.alphabet_matrix)
        for row in alphabet.alphabet_matrix:
            str_row = ','.join([str(int(x)) for x in row])
            alphabet_matrix_cpp.append('{ %s },' % str_row)
        alphabet_matrix_cpp.append("};")

        alphabet_matrix_cpp.append("__device__ const char* alphabet = \"%s\";" % ''.join(alphabet.alphabet))

        print('\n'.join(alphabet_matrix_cpp))

        self.pre_kernel = alphabet_matrix_cpp

KERNELS = {
    "sam": SamHammingKernelPrepper,
}
