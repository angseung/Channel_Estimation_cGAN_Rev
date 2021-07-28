import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
    """disc_real_output = [real_target]
    disc_generated_output = [generated_target]
    """
    # real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    #         labels=tf.ones_like(disc_real_output), logits=disc_real_output)  # label=1
    #
    # generated_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    #         labels=tf.zeros_like(disc_generated_output), logits=disc_generated_output)  # label=0

    real_loss = cross_entropy(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = cross_entropy(
        tf.zeros_like(disc_generated_output), disc_generated_output
    )

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
    gen_loss = cross_entropy(tf.ones_like(disc_generated_output), disc_generated_output)

    # L2 loss
    l2_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = tf.reduce_mean(gen_loss) + (l2_weight * l2_loss)

    return total_gen_loss
