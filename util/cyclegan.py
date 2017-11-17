import numpy as np
import os
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim

from loss import *
from models import *

class CycleGan:
    def __init__(self):
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        #Image dimensions
        self.input_w = 286
        self.input_h = 286
        self.w = 256
        self.h = 256
        self.c = 1
        
        #TODO: Maybe make these adjustable? min/max voxels
        self.min = -137.284
        self.max = 17662.5
        
        #Cycle consistency loss weights
        self.lambda_a = 10.0
        self.lambda_b = 10.0
        
        #File paths
        self.name = 'gad'
        self.train_output = os.path.join('../train/', self.name, current_time)
        self.ckpt_dir = os.path.join(self.train_output, 'checkpoints')
        self.img_dir = os.path.join(self.train_output, 'images')
        
        #Training parameters
        self.restore_ckpt = False #TODO: Pass as variable
        self.pool_size = 50
        self.base_lr = 0.0002
        self.max_step = 200
        
        #Initalize fake images
        self.fake_a = np.zeros((self.pool_size, 1, self.w, self.h, self.c))
        self.fake_b = np.zeros((self.pool_size, 1, self.w, self.h, self.c))
    
    def input_setup(self):
        input_dir = os.path.join('../datasets/', self.name)
        train_a_dir = os.path.join(input_dir, 'trainA', '*.npy')
        train_b_dir = os.path.join(input_dir, 'trainB', '*.npy')
        
        train_a_names = tf.train.match_filenames_once(train_a_dir)
        train_b_names = tf.train.match_filenames_once(train_b_dir)
        
        self.n_train_a = tf.size(train_a_names)
        self.n_train_b = tf.size(train_b_names)
        
        train_a_queue = tf.train.string_input_producer(train_a_names)
        train_b_queue = tf.train.string_input_producer(train_b_names)
        
        reader = tf.WholeFileReader()
        _, train_a_file = reader.read(train_a_queue)
        _, train_b_file = reader.read(train_b_queue)
        
        self.image_a = tf.decode_raw(train_a_file, tf.float32)
        self.image_b = tf.decode_raw(train_b_file, tf.float32)
        
        #Resize without scaling
        self.input_a = tf.image.resize_image_with_crop_or_pad(image_a, 
                                                              self.input_h, 
                                                              self.input_w)
        self.input_b = tf.image.resize_image_with_crop_or_pad(image_b, 
                                                              self.input_h, 
                                                              self.input_w)
        
        #Randomly flip
        self.input_a = tf.image.random_flip_left_right(self.input_a)
        self.input_b = tf.image.random_flip_left_right(self.input_b)
        
        #Randomly crop
        self.input_a = tf.random_crop(input_a, [self.h, self.w, self.c])
        self.input_b = tf.random_crop(input_b, [self.h, self.w, self.c])
        
        #Normalize values: -1 to 1
        denom = (self.max - self.min) / 2
        self.input_a = tf.subtract(tf.divide(tf.subtract(self.input_a, 
                                                         self.min), denom), 1)
        self.input_b = tf.subtract(tf.divide(tf.subtract(self.input_b, 
                                                         self.min), denom), 1)
        
        #Shuffle batch
        self.batch_a, self.batch_b = tf.train.shuffle_batch([self.input_a, 
                                                             self.input_b], 
                                                            1, 5000, 100)
    
    def 3d_input_setup(self):
        input_dir = os.path.join('../datasets/', self.name)
        train_a_dir = os.path.join(input_dir, 'trainA_whole', '*.npy')
        train_b_dir = os.path.join(input_dir, 'trainB_whole', '*.npy')
        
        train_a_names = tf.train.match_filenames_once(train_a_dir)
        train_b_names = tf.train.match_filenames_once(train_b_dir)
        
        train_a_queue = tf.train.string_input_producer(train_a_names)
        train_b_queue = tf.train.string_input_producer(train_b_names)
        
        reader = tf.WholeFileReader()
        _, train_a_file = reader.read(train_a_queue)
        _, train_b_file = reader.read(train_b_queue)
        
        self.image_a = tf.decode_raw(train_a_file, tf.float32)
        self.image_b = tf.decode_raw(train_b_file, tf.float32)
        
        #Resize without scaling
        self.input_a = tf.image.resize_image_with_crop_or_pad(image_a, 
                                                              self.h, 
                                                              self.w)
        self.input_b = tf.image.resize_image_with_crop_or_pad(image_b, 
                                                              self.h, 
                                                              self.w)
        
        #Normalize values: -1 to 1
        denom = (self.max - self.min) / 2
        self.input_a = tf.subtract(tf.divide(tf.subtract(self.input_a, 
                                                         self.min), denom), 1)
        self.input_b = tf.subtract(tf.divide(tf.subtract(self.input_b, 
                                                         self.min), denom), 1)
        
        #Shuffle batch
        self.batch_a, self.batch_b = tf.train.shuffle_batch([self.input_a, 
                                                             self.input_b], 
                                                            1, 250, 5)
        
    
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
            #D(A), D(B)
            self.p_real_a = discriminator(self.real_a, scope='d_a')
            self.p_real_b = discriminator(self.real_b, scope='d_b')
            
            #G(A), G(B)
            self.fake_b = generator(self.real_a, c=self.c, scope='g_a')
            self.fake_a = generator(self.real_b, c=self.c, scope='g_b')
            
            scope.reuse_variables()
            
            #D(G(B)), D(G(A))
            self.p_fake_a = discriminator(self.fake_a, scope='d_a')
            self.p_fake_b = discriminator(self.fake_b, scope='d_b')
            
            #G(G(A)), G(G(B))
            self.cycle_a = generator(self.fake_b, scope='g_b')
            self.cycle_b = generator(self.fake_a, scope='g_a')
            
            scope.reuse_variables()
            
            #Fake pool for discriminator loss
            self.p_fake_pool_a = discriminator(self.fake_pool_a, 'd_a')
            self.p_fake_pool_b = discriminator(self.fake_pool_b, 'd_b')
    
    def loss(self):
        #Cycle consistency loss
        cyclic_loss_a = self.lambda_a * cyclic_loss(self.real_a, self.cycle_a)
        cyclic_loss_b = self.lambda_b * cyclic_loss(self.real_b, self.cycle_b)
        
        #LSGAN loss
        lsgan_loss_a = lsgan_gen_loss(self.p_fake_a)
        lsgan_loss_b = lsgan_gen_loss(self.p_fake_b)
        
        #Generator loss
        g_a_loss = cyclic_loss_a + cyclic_loss_b + lsgan_loss_b
        g_b_loss = cyclic_loss_b + cyclic_loss_a + lsgan_loss_a
        
        #Discriminator loss
        d_a_loss = lsgan_dis_loss(self.p_real_a, self.p_fake_pool_a)
        d_b_loss = lsgan_dis_loss(self.p_real_b, self.p_fake_pool_b)
        
        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)
        
        #Isolate variables
        self.vars = tf.trainable_variables()
        d_a_vars = [var for v in self.vars if 'd_a' in v.name]
        d_b_vars = [var for v in self.vars if 'd_b' in v.name]
        g_a_vars = [var for v in self.vars if 'g_a' in v.name]
        g_b_vars = [var for v in self.vars if 'g_b' in v.name]
        
        #Train while freezing other variables
        self.d_a_train = optimizer.minimize(d_a_loss, var_list=d_a_vars)
        self.d_b_train = optimizer.minimize(d_b_loss, var_list=d_b_vars)
        self.g_a_train = optimizer.minimize(g_a_loss, var_list=g_a_vars)
        self.g_b_train = optimizer.minimize(g_b_loss, var_list=g_b_vars)
        
        #Summary
        self.d_a_loss_summary = tf.summary.scalar('d_a_loss', d_a_loss)
        self.d_b_loss_summary = tf.summary.scalar('d_b_loss', d_b_loss)
        self.g_a_loss_summary = tf.summary.scalar('g_a_loss', g_a_loss)
        self.g_b_loss_summary = tf.summary.scalar('g_b_loss', g_b_loss)
    
    def fake_pool(self, fake, fake_pool):
        if self.n_fake < self.pool_size:
            fake_pool[self.n_fake] = fake
            return fake
        else:
            p = random.random()
            if p < 0.5:
                index = random.randint(0, self.pool_size - 1)
                temp = fake_pool[index]
                fake_pool[index] = fake
                return temp
            else:
                return fake
    
    def save_train_images(self, sess, epoch):
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        for i in range(0, self.
    
    def train(self):
        self.input_setup()
        self.setup()
        self.loss()
        init = (tf.global_variables_initializer(), 
                tf.local_variables_initializer())
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(init)
            if self.restore_ckpt:
                ckpt_name = tf.train.latest_checkpoint(self.ckpt_dir)
                saver.restore(sess, ckpt_name)
            if not os.path.exists(self.train_output):
                os.makedirs(self.train_output)
            writer = tf.summary.FileWriter(self.train_output)
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            
            for epoch in range(sess.run(global_step), self.max_step):
                saver.save(sess, self.ckpt_dir, global_step=epoch)
                
                if epoch < 100:
                    current_lr = self.base_lr
                else:
                    current_lr = self.base_lr - self.base_lr * (epoch - 100)/100
                
                #TODO: Save training images
                
                for i in range(0, self.n_train):
                    print('Epoch {} Image {}/{}'.format(epoch, i, self.n_train))
                
                #TODO: run inputs in sess
                
