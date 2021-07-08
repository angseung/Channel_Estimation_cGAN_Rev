import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from GAN.data_preprocess import load_image_test_y, view_channel_dist
from matplotlib import pyplot as plt

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)
layers = tf.keras.layers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

test_paths_list = list(range(3, 26))
test_snr_list = [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]

nmse_df = np.zeros((len(test_paths_list), len(test_snr_list)))

generator = tf.keras.models.load_model("Models_paper/Gen_in_re_Gan_10_dB_10_path_Indoor2p5_64ant_32users_8pilot_0707")
generator.trainable = False

for j, snr in enumerate(test_snr_list):
    for k, path in enumerate(test_paths_list):

        ## Load test data
        TestData = ("../Data_Generation_matlab/Gan_Val_Data/Gan_%d_dB_%d_path_Indoor2p5_64ant_32users_8pilot_r20.mat"
                    % (snr, path))
        (realim, inpuim) = load_image_test_y(TestData)

        ## Estimate channel with generator model
        print("[%d SNR, %d PATH]... Estimating Channel Coefficients with [%d] test samples."
              % (snr, path, realim.shape[0]))
        prediction = generator(inpuim)

        ## Calculate test NMSE score...
        error_ = np.sum((realim - prediction) ** 2, axis=None)
        real_ = np.sum(realim ** 2, axis=None)
        nmse_dB = 10 * np.log10(error_ / real_)
        nmse_df[k, j] = nmse_dB

        print("[%d SNR, %d PATH]... Estimation Performance : [%2.6fdB] with [%d] test samples..."
              % (snr, path, nmse_dB, realim.shape[0]))

        # view_channel_dist(TestData, IMAGE_SAVE_OPT=True)

# np.save("NMSE_refers", nmse_df)

# fig = plt.figure()
# plt.plot(range(-10, 41, 5), nmse_df[0, :], 'rx--', label="3 PATH")
# plt.plot(range(-10, 41, 5), nmse_df[1, :], 'bo--', label="12 PATH")
# plt.plot(range(-10, 41, 5), nmse_df[2, :], 'y^--', label="25 PATH")
# plt.plot(range(-10, 41, 5), nmse, 'kv-', label="averaged")
# plt.xlabel("SNR (dB)")
# plt.ylabel("NMSE (dB)")
# plt.title("BATCH SIZE = 2")
# plt.legend(loc='best')
# plt.grid(True)
# plt.show()
# fig.savefig("papaper.png")