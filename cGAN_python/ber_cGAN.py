import os
import numpy as np
import tensorflow as tf
from signal_utils import base_mod, base_demod, awgn_noise
from GAN.data_preprocess import load_image_test_y

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

bit_len = 1024
mod_scheme = 2

(targetIm, inputIm) = load_image_test_y("../Data_Generation_matlab/Gan_Data/Comb_3_12_25_rev.mat")
bit_data = np.random.randint(low=0,
                             high=2,
                             size=(32, bit_len))
mod_data = base_mod(bit_data,
                    mod_scheme=mod_scheme)

targetIm = targetIm[:, :, :, 0] + 1j * targetIm[:, :, :, 1]
s = np.dot(targetIm[0, :, :], mod_data)
(y, _) = awgn_noise(s, 10)

generator = tf.keras.models.load_model("Models/Gen_0.00020_0.00002_0.50_100.00_10dB")
generator.trainable = False

est_H = generator(inputIm)
est_H = est_H.numpy()
est_H = est_H[:, :, :, 0] + 1j * est_H[:, :, :, 1]

est_h = np.linalg.pinv(est_H[0, :, :])

recovered_y = np.dot(est_h, y)
recovered_data = base_demod(recovered_y,
                            mod_scheme=mod_scheme)

error = np.logical_not(bit_data == recovered_data)
ber = np.mean(error, axis=-1)