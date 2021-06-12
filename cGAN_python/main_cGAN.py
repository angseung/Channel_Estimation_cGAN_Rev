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

"""
Discriminator loss:
The discriminator loss function takes 2 inputs; real images, generated images
real_loss is a sigmoid cross entropy loss of the real images and an array of ones(since the real images)
generated_loss is a sigmoid cross entropy loss of the generated images and an array of zeros(since the fake images)
Then the total_loss is the sum of real_loss and the generated_loss

Generator loss:
It is a sigmoid cross entropy loss of the generated images and an array of ones.
The paper also includes L2 loss between the generated image and the target image.
This allows the generated image to become structurally similar to the target image.
The formula to calculate the total generator loss = gan_loss + LAMBDA * l2_loss, where LAMBDA = 100. 
This value was decided by the authors of the paper.
"""
is_not_linux = (platform != 'linux')

def discriminator_loss(disc_real_output, disc_generated_output):
    """disc_real_output = [real_target]
       disc_generated_output = [generated_target]
    """
    # real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    #         labels=tf.ones_like(disc_real_output), logits=disc_real_output)  # label=1
    #
    # generated_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    #         labels=tf.zeros_like(disc_generated_output), logits=disc_generated_output)  # label=0

    real_loss = cross_entropy(tf.ones_like(disc_real_output),
                              disc_real_output)
    generated_loss = cross_entropy(tf.zeros_like(disc_generated_output),
                                   disc_generated_output)

    total_disc_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(generated_loss)

    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target, l2_weight=100):
    """
        disc_generated_output: output of Discriminator when input is from Generator
        gen_output:  output of Generator (i.e., estimated H)
        target:  target image
        l2_weight: weight of L2 loss
    """
    # GAN loss
    # gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    #         labels=tf.ones_like(disc_generated_output), logits=disc_generated_output)
    gen_loss = cross_entropy(tf.ones_like(disc_generated_output),
                             disc_generated_output)

    # L2 loss
    l2_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = tf.reduce_mean(gen_loss) + (l2_weight * l2_loss)

    return total_gen_loss


def generated_image(model, test_input, tar, t=0):
    """Dispaly  the results of Generator"""
    prediction = model(test_input)
    #plt.figure(figsize=(15, 15))
    display_list = [np.squeeze(test_input[:,:,:,0]), np.squeeze(tar[:,:,:,0]), np.squeeze(prediction[:,:,:,0])]

    title = ['Input Y', 'Target H', 'Prediction H']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i]) 
        plt.axis("off")

    plt.savefig(os.path.join("generated_img", "img_" + str(t) + ".png"))


def train_step(input_image, target, l2_weight=100):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image)                      # input -> generated_target
        disc_real_output = discriminator(target)  # [input, target] -> disc output
        disc_generated_output = discriminator(gen_output)  # [input, generated_target] -> disc output
        # print("*", gen_output.shape, disc_real_output.shape, disc_generated_output.shape)

        # calculate loss
        gen_loss = generator_loss(disc_generated_output,
                                  gen_output,
                                  target,
                                  l2_weight=l2_weight)   # gen loss
        disc_loss = discriminator_loss(disc_real_output,
                                       disc_generated_output)  # disc loss

    # gradient
    generator_gradient = gen_tape.gradient(gen_loss,
                                           generator.trainable_variables)
    discriminator_gradient = disc_tape.gradient(disc_loss,
                                                discriminator.trainable_variables)

    # apply gradient
    generator_optimizer.apply_gradients(zip(generator_gradient,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradient,
                                                discriminator.trainable_variables))

    return (gen_loss, disc_loss)


def train(epochs, l2_weight=100):
    nm = []
    ep = []
    start_time = datetime.datetime.now()

    for epoch in range(epochs):
        print("-----\nEPOCH:", epoch)
        # train
        for (bi, (target, input_image)) in enumerate(load_image_train(path)):
            elapsed_time = datetime.datetime.now() - start_time
            (gen_loss, disc_loss) = train_step(input_image,
                                               target,
                                               l2_weight=l2_weight)

            print("B/E:", bi, '/' , epoch, ", Generator loss:", gen_loss.numpy(), ", Discriminator loss:", disc_loss.numpy(), ', time:',  elapsed_time)
        # generated and see the progress
        # for bii, (tar, inp) in enumerate(load_image_test(path)):
        #     if bii == 100:
        #         generated_image(generator, inp, tar, t=epoch+1  )

        # save checkpoint
        # if (epoch + 1) % 2 == 0:
        ep.append(epoch + 1)
        #generator.save_weights(os.path.join(BASE_PATH, "weights/generator_"+str(epoch)+".h5"))
        #discriminator.save_weights(os.path.join(BASE_PATH, "weights/discriminator_"+str(epoch)+".h5"))
        
        (realim, inpuim) = load_image_test_y(path)
        prediction = generator(inpuim)

        ## Calculate test NMSE score...
        error_ = np.sum((realim - prediction) ** 2, axis=None)
        real_ = np.sum(realim ** 2, axis=None)
        nmse_dB = 10 * np.log10(error_ / real_)
        nm.append(nmse_dB)
        
        # if (epoch == (epochs - 1)):
        #     nmse_epoch = TemporaryFile()
        #     np.save(nmse_epoch, nm)
        
        # Save the predicted Channel 
        # matfiledata = {} # make a dictionary to store the MAT data in
        # matfiledata[u'predict_Gan_0_dB_Indoor2p4_64ant_32users_8pilot'] = np.array(prediction) # *** u prefix for variable name = unicode format, no issues thru Python 3.5; advise keeping u prefix indicator format based on feedback despite docs ***
        # hdf5storage.write(matfiledata, '.', 'Results\Eest_cGAN_'+str(epoch + 1)+'_0db_Indoor2p4_64ant_32users_8pilot.mat', matlab_compatible=True)
        
        plt.figure()
        plt.plot(ep,nm,'^-r')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('NMSE')

        if (is_not_linux):
            plt.show()
    
    return nm, ep

#############
####START####
#############

# data path
path = "../Data_Generation_matlab/Gan_Data/Comb_3_12_25.mat"

# Set hyper params...
# beta_1_list = [0.5, 0.6, 0.7, 0.8, 0.9]
# l2_weight_list = [0.0, 1.0, 10.0, 50.0, 100.0]
# lr_gen_list = [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 1e-5]
# lr_dis_list = [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 1e-5]

beta_1_list = [0.5]
l2_weight_list = [100.0]
lr_gen_list = [2e-4]
lr_dis_list = [2e-5]

# batch = 1 produces good results on U-NET
BATCH_SIZE = 1
epochs = 25
NMSE_SAVE_OPT = True
MODEL_SAVE_OPT = True

for beta_1 in beta_1_list:
    for l2_weight in l2_weight_list:
        for lr_gen in lr_gen_list:
            for lr_dis in lr_dis_list:

                # model
                generator = Generator()
                discriminator = Discriminator()

                # optimizer
                # generator_optimizer = tf.compat.v1.train.AdamOptimizer(2e-4, beta1=0.5) ##
                # discriminator_optimizer = tf.compat.v1.train.RMSPropOptimizer(2e-5) ##
                #discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(2e-4, beta1=0.5) # Which is unused...
                discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_dis,
                                                                   beta_1=beta_1)
                generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_gen,
                                                               beta_1=beta_1)

                # train
                (nm, ep) = train(epochs=epochs,
                                 l2_weight=l2_weight)

                fig_nmse = plt.figure(figsize=(10, 10))
                plt.plot(ep, nm, '^-r')

                for (x, y) in zip(ep, nm):
                    if (x > 9):
                        plt.text(x=x,
                                 y=y + 0.5,
                                 s=("%.3f" % y),
                                 fontsize=9,
                                 color='black',
                                 horizontalalignment='center',
                                 verticalalignment='bottom',
                                 rotation=90)

                plt.xlabel('Epoch')
                plt.ylabel('NMSE(dB)')
                plt.title("[lr_gen : %.6f][lr_dis : %.6f][beta1 : %.3f][l2_weight : %.6f]"
                          % (lr_gen, lr_dis, beta_1, l2_weight))
                plt.grid(True)

                if (is_not_linux):
                    plt.show()

                timestr = time.strftime("%Y%m%d_%H%M%S")
                fig_nmse.savefig("fig_temp/nmse_score_%s_2epoch" % (timestr))

                fname = "nmse_dB_%.5f_%.5f_%.2f_%.2f" % (lr_gen, lr_dis, beta_1, l2_weight)

                if (NMSE_SAVE_OPT):
                    nm_np = np.array(nm)
                    np.save(fname, nm_np)

                MODEL_SAVE_COND = ((lr_gen == 2e-4) and
                                  (lr_dis == 2e-5) and
                                  (beta_1 == 0.5) and
                                  (l2_weight == 100.0))

                if (MODEL_SAVE_OPT and MODEL_SAVE_COND):
                    generator.save("Models/Gen_%.5f_%.5f_%.2f_%.2f"
                                   % (lr_gen, lr_dis, beta_1, l2_weight))
                    discriminator.save("Models/Dis_%.5f_%.5f_%.2f_%.2f"
                                       % (lr_gen, lr_dis, beta_1, l2_weight))

