import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

from loss import *
from models import *

class CycleGan:
    def __init__(self):
        self.pool_size = 50
        self.input_w = 286
        self.input_h = 256
        self.w = 256
        self.h = 256
        self.c = 1
        self.lambda_a = 10
        self.lambda_b = 10
        
        self.fake_a = np.zeros((self.pool_size, 1, self.w, self.h, self.c))
        self.fake_b = np.zeros((self.pool_size, 1, self.w, self.h, self.c))
    
    def setup(self):
        self.real_a = tf.placeholder(tf.float32, [1, self.w, self.h, self.c], 
                                      name='input_a')
        self.real_b = tf.placeholder(tf.float32, [1, self.w, self.h, self.c], 
                                      name='input_b')
        self.fake_pool_a = tf.placeholder(tf.float32, 
                                          [None, self.w, self.h, self.c], 
                                          name='fake_pool_a')
        self.fake_pool_b = tf.placeholder(tf.float32, 
                                          [None, self.w, self.h, self.c], 
                                          name='fake_pool_b')
        
        self.global_step = slim.get_or_create_global_step()
        self.n_fake = 0
        self.lr = tf.placeholder(tf.float32, shape=[], name='lr')
        
        self.forward()
        
    def forward(self):
        with tf.variable_scope('CycleGAN') as scope:
            self.p_real_a = discriminator(self.real_a, scope='d_a')
            self.p_real_b = discriminator(self.real_b, scope='d_b')
            
            self.fake_b = generator(self.real_a, c=self.c, scope='g_a')
            self.fake_a = generator(self.real_b, c=self.c, scope='g_b')
            
            scope.reuse_variables()
            
            self.p_fake_a = discriminator(self.fake_a, scope='d_a')
            self.p_fake_b = discriminator(self.fake_b, scope='d_b')
            
            self.cycle_a = generator(self.fake_b, scope='g_b')
            self.cycle_b = generator(self.fake_a, scope='g_a')
            
            scope.reuse_variables()
            
            self.p_fake_pool_a = discriminator(self.fake_pool_a, 'd_a')
            self.p_fake_pool_b = discriminator(self.fake_pool_b, 'd_b')
    
