from py_ploc import TrapMap
import numpy as np
from matplotlib.tri import Triangulation, TrapezoidMapTriFinder

x, y = np.meshgrid(np.linspace(0.0, 10.0), np.linspace(0.0, 10.0))
x = x.ravel()
y = y.ravel()
tri = Triangulation(x, y)
triangles = tri.triangles.astype(int)
points = np.stack([x, y]).T

mpl_tm = tri.get_trifinder()
tm = TrapMap(points, triangles)

query = np.array([[0.1, 0.1], [0.2, 0.3]])
x_query, y_query = query.T
print(tm.locate_many(query))
print(mpl_tm(x_query, y_query))
