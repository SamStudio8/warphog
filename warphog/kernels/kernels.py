import importlib.resources as pkg_resources
import sys
from abc import ABC, abstractmethod

import numpy as np
from pycuda.compiler import SourceModule as cpp


class KernelPrepper(ABC):

    def __init__(self):
        self.f = None
        self.pre_kernel = []
        self.kernel = None
        self.kernel_lines = []
        self.post_kernel = []

    @abstractmethod
    def prepare_kernel(self, **kwargs):
        raise NotImplementedError()

    def engage(self, data_block, num_seqs, stride_len, result_arr, num_thread_pairs, num_pairs, idx_map, idy_map, block=None, grid=None):
        return self.kernel(data_block, num_seqs, stride_len, result_arr, num_thread_pairs, num_pairs, idx_map, idy_map, block=block, grid=grid) # pylint: disable=not-callable


class LessNaivePreWarpPythonHammingKernel(KernelPrepper):
    def __init__(self):
        def hamming(seq_a, seq_b, equivalence_d):
            distance = 0
            for i in range(len(seq_a)):
                if seq_a[i] not in equivalence_d[ seq_b[i] ]:
                    distance += 1
            return distance
        self.f = hamming

    #def prepare_kernel(self, **kwargs):
    #    alphabet = kwargs.get("alphabet")
    #    n_procs = kwargs.get("n_procs")

    #    if not alphabet or n_procs:
    #        raise Exception("Kernel missing required kwargs...")
    #    self.alphabet = alphabet
    #    self.n_procs = n_procs

    def prepare_kernel(self, **kwargs):
        alphabet = kwargs.get("alphabet")

        if not alphabet:
            raise Exception("Kernel missing required kwargs...")
        self.alphabet = alphabet

        from warphog.kernels.hamming import kernel  # pylint: disable=no-name-in-module,import-error
        self.kernel = kernel

    def engage(self, data_block, num_seqs, stride_len, result_arr, num_thread_pairs, num_pairs, idx_map, idy_map, block=None, grid=None):
        ord_l = np.asarray(self.alphabet.alphabet_ord_list, dtype=np.int8)
        return self.kernel(data_block, num_seqs, stride_len, result_arr, num_thread_pairs, num_pairs, idx_map, idy_map, ord_l, self.alphabet.alphabet_matrix)


class SamHammingKernelPrepper(KernelPrepper):
    # Luckily for me, we can use Hamming distance over Levenshtein distance as
    # we have an MSA already. All strings are the same size. Our MSA drops insertions
    # and deletions are marked as a symbol.
    # Presumably we could do something clever and turn the string into integers to do
    # some magic bit shifting. Regardless, CUDA will be faster than dumping it on CPU.
    # TODO Guard python from sending arrays with elements larger than unsigned short

    def __init__(self):
        super().__init__()
        self.f = "hamming_distance"
        self.set_kernel("warphog.kernels", "hamming.cu")

    def set_kernel(self, module, cu):
        self.kernel_lines = [pkg_resources.read_text(sys.modules[module], cu)]

    def get_kernel_lines(self):
        return self.pre_kernel + self.kernel_lines + self.post_kernel

    def prepare_kernel(self, **kwargs):
        alphabet = kwargs.get("alphabet")

        if not alphabet:
            raise Exception("Kernel missing required kwargs...")
        self.alphabet = alphabet

        alphabet_matrix_cpp = []

        alphabet_matrix_cpp.append("__device__ int ord_lookup[%d] = {%s};" % (len(alphabet.alphabet_ord_list), ','.join([str(x) for x in alphabet.alphabet_ord_list])))

        #TODO This could be a bitmap
        alphabet_matrix_cpp.append("__device__ int equivalent_lookup[%d][%d] = {" % (alphabet.alphabet_len, alphabet.alphabet_len))
        print(alphabet.alphabet_matrix)
        for row in alphabet.alphabet_matrix:
            str_row = ','.join([str(int(x)) for x in row])
            alphabet_matrix_cpp.append('{ %s },' % str_row)
        alphabet_matrix_cpp.append("};")

        alphabet_matrix_cpp.append("__device__ const char* alphabet = \"%s\";" % ''.join(alphabet.alphabet))

        print('\n'.join(alphabet_matrix_cpp))

        self.pre_kernel = alphabet_matrix_cpp

        self.kernel = cpp('\n'.join(self.get_kernel_lines())).get_function(self.f)
        return self.kernel

KERNELS = {
    "sam": SamHammingKernelPrepper,
    "python": LessNaivePreWarpPythonHammingKernel,
}
