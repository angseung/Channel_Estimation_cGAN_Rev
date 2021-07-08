import numpy as np
from matplotlib import pyplot as plt

b2 = np.load("NMSE_2_NOEXT.npy").mean(axis=0)
b4 = np.load("NMSE_4_NOEXT.npy").mean(axis=0)
b10 = np.load("NMSE_10_NOEXT.npy").mean(axis=0)

b2_ext = np.load("NMSE_2_EXT.npy").mean(axis=0)
b4_ext = np.load("NMSE_4_EXT.npy").mean(axis=0)
b10_ext = np.load("NMSE_10_EXT.npy").mean(axis=0)

fig = plt.figure()
plt.plot(range(-10, 41, 5), b2, 'o-', label="BATCH SIZE = 2")
plt.plot(range(-10, 41, 5), b4, 's-', label="BATCH SIZE = 4")
plt.plot(range(-10, 41, 5), b10, 'x-', label="BATCH SIZE = 10")
plt.legend(loc='best')
plt.grid(True)

plt.xlabel("SNR (dB)", fontsize='large')
plt.ylabel("NMSE (dB)", fontsize='large')
plt.ylim([-16, -4])
plt.title("DCGAN Estimation Performance with 4641 Data Samples",
          fontsize='x-large')

plt.show()
fig.savefig("paper_prev_nmse.png")

fig = plt.figure()
plt.plot(range(-10, 41, 5), b2, 'o--', label="BATCH SIZE = 2, 4641 Samples")
plt.plot(range(-10, 41, 5), b4, 's--', label="BATCH SIZE = 4, 4641 Samples")
plt.plot(range(-10, 41, 5), b10, 'x--', label="BATCH SIZE = 10, 4641 Samples")
plt.plot(range(-10, 41, 5), b2_ext, 'o-', label="BATCH SIZE = 2, 13923 Samples")
plt.plot(range(-10, 41, 5), b4_ext, 's-', label="BATCH SIZE = 4, 13923 Samples")
plt.plot(range(-10, 41, 5), b10_ext, 'x-', label="BATCH SIZE = 10, 13923 Samples")
plt.legend(loc='best')
plt.grid(True)

plt.xlabel("SNR (dB)", fontsize='large')
plt.ylabel("NMSE (dB)", fontsize='large')
plt.ylim([-16, -4])
plt.title("DCGAN Estimation Performance with Test Data Samples",
          fontsize='x-large')

plt.show()
fig.savefig("paper_after_nmse.png")
