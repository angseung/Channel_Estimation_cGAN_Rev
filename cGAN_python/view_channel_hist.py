import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sys import platform
import datetime
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from GAN.cGANGenerator import Generator
from GAN.cGANDiscriminator import Discriminator
from GAN.data_preprocess import load_image_train, load_image_test, load_image_test_y
from tempfile import TemporaryFile
from scipy.io import loadmat, savemat
import datetime
import h5py
import hdf5storage

# path_ori = "../Data_Generation_matlab/Gan_Data/Gan_10_dB_3_path_Indoor2p5_64ant_32users_8pilot_r1.mat"
path_ori = "../Data_Generation_matlab/Gan_Data/Comb_3_12_25.mat"
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
plt.subplot(321)
plt.hist(r_ori_real, 2000)
plt.xlim([-3, 3])
plt.ylim([0, 5000])
plt.title("3_real")

plt.subplot(322)
plt.hist(r_ori_imag, 2000)
plt.xlim([-3, 3])
plt.ylim([0, 5000])
plt.title("3_imag")

plt.subplot(323)
plt.hist(r_12_real, 2000)
plt.xlim([-3, 3])
plt.ylim([0, 5000])
plt.title("12_real")

plt.subplot(324)
plt.hist(r_12_imag, 2000)
plt.xlim([-3, 3])
plt.ylim([0, 5000])
plt.title("12_imag")

plt.subplot(325)
plt.hist(r_25_real, 2000)
plt.xlim([-3, 3])
plt.ylim([0, 5000])
plt.title("25_real")

plt.subplot(326)
plt.hist(r_25_imag, 2000)
plt.xlim([-3, 3])
plt.ylim([0, 5000])
plt.title("25_imag")

plt.show()