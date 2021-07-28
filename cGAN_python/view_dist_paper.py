import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from GAN.data_preprocess import load_image_test_y

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

path_ori = "../Data_Generation_matlab/Gan_Data/Gan_10_dB_3_path_Indoor2p5_64ant_32users_8pilot_r1.mat"
path_12 = "../Data_Generation_matlab/Gan_Data/Gan_10_dB_12_path_Indoor2p5_64ant_32users_8pilot_r2.mat"
path_25 = "../Data_Generation_matlab/Gan_Data/Gan_10_dB_25_path_Indoor2p5_64ant_32users_8pilot_r3.mat"
path_comb = "../Data_Generation_matlab/Gan_Data/Comb_3_12_25.mat"

(r_ori, _) = load_image_test_y(path_ori)
(r_12, _) = load_image_test_y(path_12)
(r_25, _) = load_image_test_y(path_25)
(r_comb, _) = load_image_test_y(path_comb)

num_ori = r_ori.shape[0]
num_comb = r_comb.shape[0]

# r_ori_real = r_ori[:, :, :, 0].flatten()
# r_ori_imag = r_ori[:, :, :, 1].flatten()
# r_12_real = r_12[:, :, :, 0].flatten()
# r_12_imag = r_12[:, :, :, 1].flatten()
# r_25_real = r_25[:, :, :, 0].flatten()
# r_25_imag = r_25[:, :, :, 1].flatten()
# r_comb_real = r_comb[:, :, :, 0].flatten()
# r_comb_imag = r_comb[:, :, :, 1].flatten()

r_ori_real_mean = 0.05
r_ori_imag_mean = r_ori[:, :, :, 1].mean(axis=None)
r_12_real_mean = -0.07
r_12_imag_mean = r_12[:, :, :, 1].mean(axis=None)
r_25_real_mean = 0.22
r_25_imag_mean = r_25[:, :, :, 1].mean(axis=None)
r_comb_real_mean = 0.03
r_comb_imag_mean = r_comb[:, :, :, 1].mean(axis=None)

r_ori_real_std = 0.25
r_ori_imag_std = r_ori[:, :, :, 1].std(axis=None)
r_12_real_std = 0.3
r_12_imag_std = r_12[:, :, :, 1].std(axis=None)
r_25_real_std = 0.8
r_25_imag_std = r_25[:, :, :, 1].std(axis=None)
r_comb_real_std = 0.5
r_comb_imag_std = r_comb[:, :, :, 1].std(axis=None)

# dist_3 = np.random.normal(loc=r_ori_real_mean, scale=r_ori_real_std, size=num_ori)
# dist_12 = np.random.normal(loc=r_12_real_mean, scale=r_12_real_std, size=num_ori)
# dist_25 = np.random.normal(loc=r_25_real_mean, scale=r_25_real_std, size=num_ori)
# dist_comb = np.random.normal(loc=r_comb_real_mean, scale=r_comb_real_std, size=num_comb)

dist_3 = np.linspace(
    r_ori_real_mean - 3 * r_ori_real_std, r_ori_real_mean + 3 * r_ori_real_std, 100
)

dist_12 = np.linspace(
    r_12_real_mean - 3 * r_12_real_std, r_12_real_mean + 3 * r_12_real_std, 100
)

dist_25 = np.linspace(
    r_25_real_mean - 3 * r_comb_real_std, r_25_real_mean + 3 * r_comb_real_std, 100
)

dist_comb = np.linspace(
    r_comb_real_mean - 3 * r_comb_real_std, r_comb_real_mean + 3 * r_comb_real_std, 100
)

fig = plt.figure()
plt.subplot(212)
# plt.hist(dist_3, bins=4000)
# plt.hist(dist_12, bins=4000)
# plt.hist(dist_25, bins=4000)
plt.plot(
    dist_3, stats.norm.pdf(dist_3, r_ori_real_mean, r_ori_real_std), "-", label="3 PATH"
)
plt.plot(
    dist_12,
    stats.norm.pdf(dist_12, r_12_real_mean, r_12_real_std),
    "-",
    label="12 PATH",
)
plt.plot(
    dist_25,
    stats.norm.pdf(dist_25, r_25_real_mean, r_25_real_mean),
    "-",
    label="25 PATH",
)
plt.axis("off")
plt.ylim([0.0, 2.0])
plt.xlim([-2.0, 2.0])
# plt.legend(loc='best')
# plt.show()

plt.subplot(211)
# fig = plt.figure()
plt.plot(
    dist_comb,
    stats.norm.pdf(dist_comb, r_comb_real_mean, r_comb_real_std),
    "k-",
    label="comb",
)
plt.axis("off")
plt.ylim([0.0, 2.0])
plt.xlim([-2.0, 2.0])
# plt.legend(loc='best')
plt.show()

# fig = plt.figure(figsize=(10, 10))
# plt.subplots_adjust(left=0.125, bottom=0.1,  right=0.9, top=0.9, wspace=0.2, hspace=0.35)
# plt.subplot(321)
# plt.hist(r_ori_real, 4000, label="MPCs = 3")
# plt.xlim([-3, 3])
# plt.ylim([0, 1000])
# plt.title("Real Part")
# plt.legend(loc='best')
#
# plt.subplot(322)
# plt.hist(r_ori_imag, 4000, label="MPCs = 3")
# plt.xlim([-3, 3])
# plt.ylim([0, 1000])
# plt.title("Imaginary Part")
# plt.legend(loc='best')
#
# plt.subplot(323)
# plt.hist(r_12_real, 4000, label="MPCs = 12")
# plt.xlim([-3, 3])
# plt.ylim([0, 1000])
# plt.title("Real Part")
# plt.legend(loc='best')
#
# plt.subplot(324)
# plt.hist(r_12_imag, 4000, label="MPCs = 12")
# plt.xlim([-3, 3])
# plt.ylim([0, 1000])
# plt.title("Imaginary Part")
# plt.legend(loc='best')
#
# plt.subplot(325)
# plt.hist(r_25_real, 4000, label="MPCs = 25")
# plt.xlim([-3, 3])
# plt.ylim([0, 1000])
# plt.title("Real Part")
# plt.legend(loc='best')
#
# plt.subplot(326)
# plt.hist(r_25_imag, 4000, label="MPCs = 25")
# plt.xlim([-3, 3])
# plt.ylim([0, 1000])
# plt.title("Imaginary Part")
# plt.legend(loc='best')
#
# plt.suptitle("Channel distribution under different MPC condition", fontsize='xx-large')
# plt.show()
