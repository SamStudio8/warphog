from abc import ABC, abstractmethod

import numpy as np
from pycuda import gpuarray

from warphog.kernels.kernels import KERNELS


class WarpCore(ABC):

    def __init__(self, seq_block, alphabet):
        self.seq_block = seq_block
        self.seq_block_l = len(seq_block)
        self.seq_l = len(seq_block[0]) #TODO clean
        self.data_block = seq_block
        self.alphabet = alphabet

        self.d = None
        self.kernel = None
        self.idx_map = None
        self.idy_map = None

    @property
    def result_arr(self):
        return self.d

    def engage(self):
        raise NotImplementedError()

    @abstractmethod
    def put_d(self):
        raise NotImplementedError()

    @abstractmethod
    def get_d(self):
        raise NotImplementedError()

    @abstractmethod
    def put_maps(self, idx_map, idy_map):
        raise NotImplementedError()


class CPUPreWarpCore(WarpCore):

    def __init__(self, seq_block, alphabet):
        super().__init__(seq_block, alphabet)
        #self.data_block = self._make_data_block(seq_block)

        self.kernel = KERNELS["python"]()
        self.kernel.prepare_kernel(alphabet=alphabet)

    def _make_data_block(self, seq_block):
        msa_char_block = np.fromstring("".join(seq_block).encode(), dtype=np.byte)
        return msa_char_block

    def put_d(self, d):
        self.d = d

    def get_d(self, d):
        pass

    def put_maps(self, idx_map, idy_map):
        self.idx_map = idx_map
        self.idy_map = idy_map


class GPUWarpCore(WarpCore):

    def __init__(self, seq_block, alphabet):
        super().__init__(seq_block, alphabet)
        self.d = None
        self.data_block = self._make_data_block(seq_block)

        self.kernel = KERNELS["sam"]()
        self.kernel.prepare_kernel(alphabet=alphabet)

    def put_d(self, d):
        self.d = gpuarray.to_gpu(d)

    def get_d(self, d):
        self.d.get(d)

    def _make_data_block(self, seq_block):
        msa_char_block = np.frombuffer("".join(seq_block).encode(), dtype=np.byte)
        msa_gpu = gpuarray.to_gpu(msa_char_block)
        return msa_gpu

    def put_maps(self, idx_map, idy_map):
        idx_map_gpu = gpuarray.to_gpu(idx_map)
        idy_map_gpu = gpuarray.to_gpu(idy_map)
        self.idx_map = idx_map_gpu
        self.idy_map = idy_map_gpu

    #def engage(self, n_pairs, idx_map, idy_map, **kwargs):
    #    block_dim = kwargs.get("block_dim")
    #    grid_dim = kwargs.get("grid_dim")
    
    #    if not block_dim or grid_dim:
    #        raise Exception("GPUWarpCore.engage requires block_dim and grid_dim")


CORES = {
    "warp": GPUWarpCore,
    "prewarp": CPUPreWarpCore,
}
