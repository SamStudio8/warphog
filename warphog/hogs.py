from abc import ABC, abstractmethod
from math import ceil, sqrt

import numpy as np
import sys

class WarpHog(ABC):
    def __init__(self, n, m=None):
        self.num_seqs = n

        self.n = n
        self.m = m if m is not None else n
        self.seq_dim_x, self.seq_dim_y = self.n, self.m

        self.idx_map, self.idy_map = self._get_thread_map()

        self.block_dim_x = 4
        self.block_dim_y = 4
        self.block_dim_z = 1

        self.pairs_per_thread = 1 #TODO busted

    @property
    def seq_dim(self):
        return self.get_seq_dim()

    @property
    def block_dim(self):
        return (self.block_dim_x, self.block_dim_y, self.block_dim_z)

    @property
    def grid_dim(self):
        return self._get_grid_dim()

    @property
    def thread_count(self):
        block_dim = self.block_dim
        grid_dim = self.grid_dim
        return block_dim[0] * block_dim[1] * block_dim[2] * grid_dim[0] * grid_dim[1]

    @abstractmethod
    def _get_grid_dim(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_thread_map(self):
        raise NotImplementedError()

    def get_thread_map(self):
        if self.idx_map is None or self.idy_map is None:
            self.idx_map, self.idy_map = self._get_thread_map()
        return self.idx_map, self.idy_map

    @property
    def num_pairs(self):
        return self._get_num_pairs()

    @abstractmethod
    def _get_num_pairs(self):
        raise NotImplementedError()

    def get_num_pairs(self):
        if not self.num_pairs:
            self.num_pairs = self._get_num_pairs()
        return self.num_pairs

    def broadcast_result(self, src, dest):
        idx_map, idy_map = self.get_thread_map()
        dest[idx_map, idy_map] = src

    def output_tsv(self, d):
        for i in range(self.num_pairs):
            sys.stdout.write('\t'.join([str(x) for x in [
                self.idx_map[i],
                self.idy_map[i],
                d[i],
            ]]) + '\n')


class AsymmetricalRectangleWarpHog(WarpHog):

    def __init__(self, n, m=1):
        super().__init__(n=n, m=m)

    def _get_num_pairs(self):
        return self.seq_dim_x * self.seq_dim_y

    def _get_thread_map(self):
        # Return a rectangle, regardless of whether some of the queries will cause repeats
        x, y = np.asarray(np.indices((self.seq_dim_x, self.seq_dim_y)), dtype=np.uint32)
        return x.flatten(), y.flatten()

    def _get_grid_dim(self):
        return ( ceil(self.seq_dim_x / (self.pairs_per_thread * self.block_dim_x)), ceil(self.seq_dim_y / (self.block_dim_y)) )


class TriangularWarpHog(WarpHog):

    def _get_thread_map(self):
        return np.asarray(np.triu_indices(self.num_seqs), dtype=np.uint32)

    def _get_num_pairs(self):
        return ceil(( self.num_seqs * (self.num_seqs + 1) ) / 2)

    def _get_grid_dim(self):
        grid_width = sqrt(self.get_num_pairs())
        return ( ceil(grid_width / (self.pairs_per_thread * self.block_dim_x)), ceil(grid_width / (self.block_dim_y)) )


HOGS = {
    "all": TriangularWarpHog,
    "some": AsymmetricalRectangleWarpHog,
}
