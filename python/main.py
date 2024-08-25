from ploc import TrapMap
import numpy as np
from matplotlib.tri import Triangulation

n = 200
x, y = np.meshgrid(np.linspace(0.0, 10.0, n), np.linspace(0.0, 10.0, n))
x = x.ravel()
y = y.ravel()
tri = Triangulation(x, y)
triangles = tri.triangles.astype(int)

mpl_tm = tri.get_trifinder()
tm = TrapMap(x, y, triangles)

query = np.array([[0.1, 0.1], [0.2, 0.3]])
x_query, y_query = query.T
print(tm(x_query, y_query))
print(mpl_tm(x_query, y_query))
