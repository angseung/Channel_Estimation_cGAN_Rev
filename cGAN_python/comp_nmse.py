import numpy as np
from matplotlib import pyplot as plt

nmse_ori = np.load("nmse_paper/nm_ori.npy")
nmse_bat1 = np.load("nmse_paper/nmse_2b.npy")
nmse_bat2 = np.load("nmse_paper/nmse_4b.npy")
nmse_bat4 = np.load("nmse_paper/nmse_10b.npy")
# nmse_bat10 = np.load("nmse_paper/nmse_dB_0.00020_0.00002_0.50_100.00_10.npy")

ep = range(1, nmse_ori.shape[0] + 1)

fig = plt.figure()
plt.plot(ep, nmse_ori, "b^-", label="Original cGAN")
plt.plot(ep, nmse_bat1, "k>-",  label="proposed GAN, BATCHSIZE=2")
plt.plot(ep, nmse_bat2, "r<-",  label="proposed GAN, BATCHSIZE=4")
plt.plot(ep, nmse_bat4, "yv-",  label="proposed GAN, BATCHSIZE=10")
# plt.plot(ep, nmse_bat10, "o-",  label="proposed GAN, BATCHSIZE=10")
plt.grid(True)
plt.legend(loc='best')
plt.xlabel("Epoch")
plt.ylabel("NMSE (dB)")
plt.title("GAN performance comparison")
plt.show()
fig.savefig("paper.png")