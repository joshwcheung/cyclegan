import tensorflow as tf

def cyclic_loss(real, cycle):
    return tf.reduce_mean(tf.abs(real - cycle))

def lsgan_gen_loss(fake):
    return tf.reduce_mean(tf.squared_difference(fake, 1))

def lsgan_dis_loss(real, fake):
    return (tf.reduce_mean(tf.squared_difference(real, 1)) + 
            tf.reduce_mean(tf.squared_difference(fake, 0))) * 0.5

