import tensorflow as tf

from layers import *

def resnet_block(x, nf, scope='res'):
    with tf.variable_scope(scope):
        y = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        y = conv2d(y, nf, 3, 1, padding='VALID', scope='_conv1')
        y = instance_norm(y, scope='_norm1')
        y = relu(y, name='_relu1')
        y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        y = conv2d(y,nf, 3, 1, padding='VALID', scope='_conv2')
        y = instance_norm(y, scope='_norm2')
        return x + y

def generator(x, nf=32, c=1, scope='gen'):
    with tf.variable_scope(scope):
        #Convolutional layers
        g_p1 = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
        g_c1 = conv2d(g_p1, nf, 7, stride=1, padding='VALID', scope='conv1')
        g_n1 = instance_norm(g_c1, scope='norm1')
        g_r1 = relu(g_n1, name='relu1')
        
        g_c2 = conv2d(g_r1, nf * 2, 3, stride=2, scope='conv2')
        g_n2 = instance_norm(g_c2, scope='norm2')
        g_r2 = relu(g_n2, name='relu2')
        
        g_c3 = conv2d(g_r2, nf * 4, 3, stride=2, scope='conv3')
        g_n3 = instance_norm(g_c3, scope='norm3')
        g_r3 = relu(g_n3, name='relu3')
        
        #ResNet blocks
        res1 = resnet_block(g_r3, nf * 4, scope='res1')
        res2 = resnet_block(res1, nf * 4, scope='res2')
        res3 = resnet_block(res2, nf * 4, scope='res3')
        res4 = resnet_block(res3, nf * 4, scope='res4')
        res5 = resnet_block(res4, nf * 4, scope='res5')
        res6 = resnet_block(res5, nf * 4, scope='res6')
        res7 = resnet_block(res6, nf * 4, scope='res7')
        res8 = resnet_block(res7, nf * 4, scope='res8')
        res9 = resnet_block(res8, nf * 4, scope='res9')
        
        #Deconvolutional layers
        g_c4 = deconv2d(res9, nf * 2, 3, stride=2, scope='dconv4')
        g_n4 = instance_norm(g_c4, scope='norm4')
        g_r4 = relu(g_n4, name='relu4')
        
        g_c5 = deconv2d(g_r4, nf, 3, stride=2, scope='dconv5')
        g_n5 = instance_norm(g_c5, scope='norm5')
        g_r5 = relu(g_n5, name='relu5')
        
        g_p2 = tf.pad(g_r5, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
        g_c6 = conv2d(g_p2, c, 7, stride=1, padding='VALID', 
                      scope='conv6')
        
        y = tf.nn.tanh(g_c6, name='tanh1')
        return y

def discriminator(x, nf=64, scope='dis'):
    with tf.variable_scope(scope):
        #Convolutional layers
        d_c1 = conv2d(x, nf, 4, stride=2, scope='conv1')
        d_r1 = lrelu(d_c1, scope='lrelu1')
        
        d_c2 = conv2d(d_r1, nf * 2, 4, stride=2, scope='conv2')
        d_n2 = instance_norm(d_c2, scope='norm2')
        d_r2 = lrelu(d_n2, scope='lrelu2')
        
        d_c3 = conv2d(d_r2, nf * 4, 4, stride=2, scope='conv3')
        d_n3 = instance_norm(d_c3, scope='norm3')
        d_r3 = lrelu(d_n3, scope='lrelu3')
        
        d_c4 = conv2d(d_r3, nf * 8, 4, stride=1, scope='conv4')
        d_n4 = instance_norm(d_c4, scope='norm4')
        d_r4 = lrelu(d_n4, scope='lrelu4')
        
        d_c5 = conv2d(d_r4, 1, 4, stride=1, scope='conv5')
        
        return d_c5

