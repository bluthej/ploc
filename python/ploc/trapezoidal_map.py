from ploc.ploc import PyTrapMap
import numpy as np


class TrapMap:
    def __init__(self, x, y, cells):
        points = np.column_stack([x, y]).astype(np.float64)
        self._tm = PyTrapMap(points, cells.astype(np.int64))

    def __call__(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError("x and y must be array-like with the same shape")

        # Rust does the heavy lifting, and expects 2D arrays.
        query = np.column_stack([x.ravel(), y.ravel()])
        indices = self._tm.locate_many(query).reshape(x.shape)

        return indices
