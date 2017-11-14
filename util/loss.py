import tensorflow as tf

def cycle_consistency_loss(real, cycle):
    return tf.reduce_mean(tf.abs(real - cycle))

def generator_loss(fake):
    return tf.reduce_mean(tf.squared_difference(fake, 1))

def discriminator_loss(real, fake):
    return (tf.reduce_mean(tf.squared_difference(real, 1)) + 
            tf.reduce_mean(tf.squared_difference(fake, 0))) * 0.5

