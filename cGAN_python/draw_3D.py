import numpy as np
from matplotlib import pyplot as plt

nmse_3D = np.load("NMSE_3D.npy")

x = np.array(range(-10, 41, 5))
y = np.array((range(3, 26)))

X, Y = np.meshgrid(x, y)

# nmse = nmse_3D.mean(axis=0)
nmse = nmse_3D[1, :, :]

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf1 = ax.plot_surface(X, Y, nmse_3D[0, :, :])
surf2 = ax.plot_surface(X, Y, nmse_3D[1, :, :])
surf3 = ax.plot_surface(X, Y, nmse_3D[2, :, :])

# fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel("SNR (dB)")
plt.ylabel("The number of MPCs")
ax.set_zlabel("NMSE (dB)")
plt.show()

fig2 = plt.figure(figsize=(8, 6))

for (i, path) in enumerate(y):
    if path in [3, 13, 25]:
        plt.plot(x, nmse[i, :], "x-", label="nmse with path %02d" % path)
    else:
        plt.plot(x, nmse[i, :], "--", label="nmse with path %02d" % path)

plt.legend(loc='best', ncol=3)
plt.grid(True)

plt.show()