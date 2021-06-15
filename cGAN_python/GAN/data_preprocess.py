from sys import platform
import numpy as np
import matplotlib.pyplot as plt
import random
import h5py
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)
layers = tf.keras.layers


def load_image_train(path, batch_size = 1):
    """load, jitter, and normalize"""
    with h5py.File(path, 'r') as file:
        real_image = np.transpose(np.array(file['output_da']))

    with h5py.File(path, 'r') as file:
        input_image = np.transpose(np.array(file['input_da']))
        
    SIZE_IN= real_image.shape
    list_im=list(range(0, SIZE_IN[0]))

    batch_im = random.sample(list_im,SIZE_IN[0])
    real_image = real_image[batch_im, :, :, :]
    input_image = input_image[batch_im, :, :, :]
    
    n_batches = int(SIZE_IN[0] / batch_size)
    
    for i in range(n_batches - 1):
        imgs_A = real_image[i * batch_size : (i + 1) * batch_size]
        imgs_B = input_image[i * batch_size : (i + 1) * batch_size]

    
        yield (imgs_A, imgs_B)


def load_image_test(path, batch_size = 1):
       
    with h5py.File(path, 'r') as file:
        real_image = np.transpose(np.array(file['output_da_test']))

        
    with h5py.File(path, 'r') as file:
        input_image = np.transpose(np.array(file['input_da_test']))
        
    SIZE_IN= real_image.shape

    n_batches = int(SIZE_IN[0] / batch_size)
    
    for i in range(n_batches - 1):
        imgs_A = real_image[i * batch_size : (i + 1) * batch_size]
        imgs_B = input_image[i * batch_size : (i + 1) * batch_size]
        
    
        yield (imgs_A, imgs_B)


def load_image_train_y(path):
    with h5py.File(path, 'r') as file:
        real_image = np.transpose(np.array(file['output_da']))

    with h5py.File(path, 'r') as file:
        input_image = np.transpose(np.array(file['input_da']))

    return (real_image, input_image)

        
def load_image_test_y(path):
       
    with h5py.File(path, 'r') as file:
        real_image = np.transpose(np.array(file['output_da_test']))

        
    with h5py.File(path, 'r') as file:
        input_image = np.transpose(np.array(file['input_da_test']))
        

    return (real_image, input_image)

def view_channel_dist(path, TRAIN_VIEW_OPT = False, IMAGE_SAVE_OPT = False):

    if (TRAIN_VIEW_OPT):
        (channel, _) = load_image_train_y(path)
    else:
        (channel, _) = load_image_test_y(path)

    channel_r = channel[:, :, :, 0].flatten()
    channel_i = channel[:, :, :, 1].flatten()

    fig_hist = plt.figure(figsize=(10, 10))
    plt.subplot(211)
    plt.hist(channel_r, bins=1000)
    plt.grid(True)
    plt.title("Real Part")

    plt.subplot(212)
    plt.hist(channel_i, bins=1000)
    plt.grid(True)
    plt.title("Imaginary Part")
    plt.suptitle(path[35:])

    if (platform != 'linux'):
        plt.show()

    if (IMAGE_SAVE_OPT):
        fig_hist.savefig("hist_%s.png" % path[35 : -4])

    return None
