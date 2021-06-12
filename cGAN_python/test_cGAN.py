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

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)
layers = tf.keras.layers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

lr_gen = 2e-4
lr_dis = 2e-5
beta_1 = 0.5
l2_weight = 100.0

f_gen = "Models/Gen_%.5f_%.5f_%.2f_%.2f" % (lr_gen, lr_dis, beta_1, l2_weight)
# f_dis = "Models/Dis_%.5f_%.5f_%.2f_%.2f" % (lr_gen, lr_dis, beta_1, l2_weight)

## Load trained model
generator = tf.keras.models.load_model(f_gen)
generator.trainable = False
# discriminator = tf.keras.models.load_model(f_dis)

test_paths_list = [3, 12, 25]

for paths in test_paths_list:

    ## Load test data
    TestData = ("../Data_Generation_matlab/Gan_Data/Gan_10_dB_%d_path_Indoor2p5_64ant_32users_8pilot_testdat.mat"
                % paths)
    (realim, inpuim) = load_image_test_y(TestData)

    ## Estimate channel with generator model
    print("[%d PATH] Estimating Channel Coefficients with %d test samples." % (paths, realim.shape[0]))
    prediction = generator(inpuim)

    ## Calculate test NMSE score...
    error_ = np.sum((realim - prediction) ** 2, axis=None)
    real_ = np.sum(realim ** 2, axis=None)
    nmse_dB = 10 * np.log10(error_ / real_)

    print("[%d PATH] Estimation Performance : %2.4f with %d test samples..." % (paths, nmse_dB, realim.shape[0]))

