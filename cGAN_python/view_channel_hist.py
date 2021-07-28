import os
import numpy as np
import matplotlib.pyplot as plt
from GAN.data_preprocess import load_image_test_y

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path_ori = "../Data_Generation_matlab/Gan_Data/Gan_10_dB_3_path_Indoor2p5_64ant_32users_8pilot_r1.mat"
# path_ori = "../Data_Generation_matlab/Gan_Data/Comb_3_12_25.mat"
path_12 = "../Data_Generation_matlab/Gan_Data/Gan_10_dB_12_path_Indoor2p5_64ant_32users_8pilot_r2.mat"
path_25 = "../Data_Generation_matlab/Gan_Data/Gan_10_dB_25_path_Indoor2p5_64ant_32users_8pilot_r3.mat"

(r_ori, _) = load_image_test_y(path_ori)
(r_12, _) = load_image_test_y(path_12)
(r_25, _) = load_image_test_y(path_25)

r_ori_real = r_ori[:, :, :, 0].flatten()
r_ori_imag = r_ori[:, :, :, 1].flatten()
r_12_real = r_12[:, :, :, 0].flatten()
r_12_imag = r_12[:, :, :, 1].flatten()
r_25_real = r_25[:, :, :, 0].flatten()
r_25_imag = r_25[:, :, :, 1].flatten()

fig = plt.figure(figsize=(10, 10))
plt.subplots_adjust(left=0.125, bottom=0.1,  right=0.9, top=0.9, wspace=0.2, hspace=0.35)
plt.subplot(321)
plt.hist(r_ori_real, 4000, label="MPCs = 3")
plt.xlim([-3, 3])
plt.ylim([0, 1000])
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Real Part")
plt.legend(loc='best')

plt.subplot(322)
plt.hist(r_ori_imag, 4000, label="MPCs = 3")
plt.xlim([-3, 3])
plt.ylim([0, 1000])
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Imaginary Part")
plt.legend(loc='best')

plt.subplot(323)
plt.hist(r_12_real, 4000, label="MPCs = 12")
plt.xlim([-3, 3])
plt.ylim([0, 1000])
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Real Part")
plt.legend(loc='best')

plt.subplot(324)
plt.hist(r_12_imag, 4000, label="MPCs = 12")
plt.xlim([-3, 3])
plt.ylim([0, 1000])
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Imaginary Part")
plt.legend(loc='best')

plt.subplot(325)
plt.hist(r_25_real, 4000, label="MPCs = 25")
plt.xlim([-3, 3])
plt.ylim([0, 1000])
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Real Part")
plt.legend(loc='best')

plt.subplot(326)
plt.hist(r_25_imag, 4000, label="MPCs = 25")
plt.xlim([-3, 3])
plt.ylim([0, 1000])
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Imaginary Part")
plt.legend(loc='best')

plt.suptitle("Channel distributions under different MPC conditions", fontsize='xx-large')
plt.show()