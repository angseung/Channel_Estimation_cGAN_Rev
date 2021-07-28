import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
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

lr_gen = 2e-4
lr_dis = 2e-5
beta_1 = 0.5
l2_weight = 100.0
SNR = 10

# f_gen = "Models/Gen_%.5f_%.5f_%.2f_%.2f_%ddB_ext2" % (lr_gen, lr_dis, beta_1, l2_weight, SNR)
# f_gen = "Models/Gen_ori_2"
# f_dis = "Models/Dis_%.5f_%.5f_%.2f_%.2f" % (lr_gen, lr_dis, beta_1, l2_weight)

## Load trained model
# generator = tf.keras.models.load_model(f_gen)
# generator.trainable = False
# discriminator = tf.keras.models.load_model(f_dis)

test_paths_list = list(range(3, 26))
test_snr_list = [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]

fnames = [
    "Models/Gen_0.00020_0.00002_0.50_100.00_10dB_ext2",
    "Models/Gen_0.00020_0.00002_0.50_100.00_10dB_ext4",
    "Models/Gen_0.00020_0.00002_0.50_100.00_10dB_ext10",
]
nmse_df = np.zeros((len(fnames), len(test_paths_list), len(test_snr_list)))

for i, fname in enumerate(fnames):
    generator = tf.keras.models.load_model(fname)
    generator.trainable = False

    for j, snr in enumerate(test_snr_list):
        for k, path in enumerate(test_paths_list):

            ## Load test data
            TestData = (
                "../Data_Generation_matlab/Gan_Data/Gan_%d_dB_%d_path_Indoor2p5_64ant_32users_8pilot_r20.mat"
                % (snr, path)
            )
            (realim, inpuim) = load_image_test_y(TestData)

            ## Estimate channel with generator model
            print(
                "[%d SNR, %d PATH]... Estimating Channel Coefficients with [%d] test samples."
                % (snr, path, realim.shape[0])
            )
            prediction = generator(inpuim)

            ## Calculate test NMSE score...
            error_ = np.sum((realim - prediction) ** 2, axis=None)
            real_ = np.sum(realim ** 2, axis=None)
            nmse_dB = 10 * np.log10(error_ / real_)
            nmse_df[i, k, j] = nmse_dB

            print(
                "[%d SNR, %d PATH]... Estimation Performance : [%2.4fdB] with [%d] test samples..."
                % (snr, path, nmse_dB, realim.shape[0])
            )

            # view_channel_dist(TestData, IMAGE_SAVE_OPT=True)

nmse = nmse_df.mean(axis=0)
# np.save("NMSE_refers", nmse_df)

fig = plt.figure()
plt.plot(range(-10, 41, 5), nmse_df[0, :], "rx--", label="3 PATH")
plt.plot(range(-10, 41, 5), nmse_df[1, :], "bo--", label="12 PATH")
plt.plot(range(-10, 41, 5), nmse_df[2, :], "y^--", label="25 PATH")
plt.plot(range(-10, 41, 5), nmse, "kv-", label="averaged")
plt.xlabel("SNR (dB)")
plt.ylabel("NMSE (dB)")
plt.title("BATCH SIZE = 2")
plt.legend(loc="best")
plt.grid(True)
plt.show()
# fig.savefig("papaper.png")
