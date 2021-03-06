from abc import ABC, abstractmethod
from math import ceil, sqrt

import numpy as np


class WarpHog(ABC):
    def __init__(self, n, m=None):
        self.n = n
        self.m = m if m is not None else n
        self.seq_dim_x, self.seq_dim_y = self.n, self.m

        self.idx_map, self.idy_map = self._get_thread_map()

        self.block_dim_x = 4
        self.block_dim_y = 4
        self.block_dim_z = 1

        self.pairs_per_thread = 1 #TODO busted

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

    def output_tsv(self, d, o, k=-1, names=None):
        l = 0
        idx_map, idy_map = self.get_thread_map()

        if k > -1:
            idx = np.where(d <= k)[0]
            val = d[idx]
        else:
            idx = np.where(d >= 0)[0]
            val = d[idx]
        idx_mapped = idx_map[idx]
        idy_mapped = idy_map[idx]


        with open(o, 'w', buffering=1024000) as fh:
            for i in range(len(idx)):
                if idx_mapped[i] == idy_mapped[i]:
                    continue

                if names:
                    s = "%s\t%s\t%d\n" % (names[idx_mapped[i]], names[idy_mapped[i]], val[i])
                else:
                    s = "%d\t%d\t%d\n" % (idx_mapped[i], idy_mapped[i], val[i])
                l += fh.write(s)
        return l



class RectangularWarpHog(WarpHog):

    def __init__(self, n, m):
        if m is None:
            raise Exception("m must be specified for RectangularWarpHog")
        super().__init__(n=n, m=m)

    def _get_num_pairs(self):
        return self.seq_dim_x * self.seq_dim_y

    def _get_thread_map(self):
        # Return a rectangle, regardless of whether some of the queries will cause repeats
        x, y = np.asarray(np.indices((self.seq_dim_x, self.seq_dim_y)), dtype=np.uint32)
        return x.flatten(), y.flatten()

    def _get_grid_dim(self):
        return ( ceil(self.seq_dim_x / (self.pairs_per_thread * self.block_dim_x)), ceil(self.seq_dim_y / (self.block_dim_y)) )

    @property
    def num_seqs(self):
        return self.seq_dim_x + self.seq_dim_y

class TriangularWarpHog(WarpHog):

    def __init__(self, n, m=None):
        if m is not None and m != n:
            raise Exception("n and m must have same rank for TriangularWarpHog")
        super().__init__(n=n, m=m)

    def _get_thread_map(self):
        return np.asarray(np.triu_indices(self.num_seqs), dtype=np.uint32)

    def _get_num_pairs(self):
        return ceil(( self.num_seqs * (self.num_seqs + 1) ) / 2)

    def _get_grid_dim(self):
        grid_width = sqrt(self.get_num_pairs())
        return ( ceil(grid_width / (self.pairs_per_thread * self.block_dim_x)), ceil(grid_width / (self.block_dim_y)) )

    @property
    def num_seqs(self):
        return self.seq_dim_x

