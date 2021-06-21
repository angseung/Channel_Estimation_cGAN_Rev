import numpy as np
import matplotlib.pyplot as plt

noext_4 = np.load("NMSE_4_NOEXT.npy").mean(axis=0)
noext_10 = np.load("NMSE_10_NOEXT.npy").mean(axis=0)

ext_4 = np.load("NMSE_4_EXT.npy").mean(axis=0)
ext_10 = np.load("NMSE_10_EXT.npy").mean(axis=0)

ext_2 = np.load("NMSE_2_EXT.npy").mean(axis=0)

ori_nmse = np.load("NMSE_refers.npy").mean(axis=0)

fig = plt.figure()
# plt.plot(range(-10, 41, 5), noext_4, "o--", label="no ext, BS=4")
# plt.plot(range(-10, 41, 5), noext_10, "x--", label="no ext, BS=10")
plt.plot(range(-10, 41, 5), ext_4, "o--", label="ext, BS=4")
plt.plot(range(-10, 41, 5), ext_10, "x--", label="ext, BS=10")
plt.plot(range(-10, 41, 5), ext_2, "ro--", label="ext, BS=2", markersize=8)
plt.plot(range(-10, 41, 5), ori_nmse, "kx--", label="reference", markersize=10)

plt.xlabel("SNR (dB)", fontsize='large')
plt.ylabel("NMSE (dB)", fontsize='large')
plt.ylim([-17, -4])
plt.grid(True)
plt.legend(loc='best', fontsize='x-large')
plt.title("")
plt.show()
