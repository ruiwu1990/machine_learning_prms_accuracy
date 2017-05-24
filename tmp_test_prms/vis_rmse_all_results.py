from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd
import numpy as np

df = pd.read_csv('rmse_all_results.csv')

a = df['alpha'].tolist()
b = df['window_size'].tolist()
c = df['rmse'].tolist()

a = np.asarray(a)
b = np.asarray(b)
c = np.asarray(c)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(a, b, c)

ax.set_xlabel('Alpha')
ax.set_ylabel('Window Per')
ax.set_zlabel('RMSE')

plt.show()

# fig = plt.figure()
# ax = fig.gca(projection='3d')

# # Plot the surface.
# surf = ax.plot_surface(np.asarray(a), np.asarray(b), np.asarray(c), cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)


# ax.set_zlim(3.0, 7.0)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()


