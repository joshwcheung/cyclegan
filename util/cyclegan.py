import numpy as np
import os
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim

from datetime import datetime

from loss import *
from models import *
from nifti_to_tfrecord import read_from_tfrecord

class CycleGAN:
    def __init__(self):
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        #Image dimensions
        self.input_w = 286
        self.input_h = 286
        self.w = 256
        self.h = 256
        self.d = 256
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
        self.n_save = 1
        
        #Initalize fake images
        self.fake_a = np.zeros((self.pool_size, 1, self.w, self.h, self.c))
        self.fake_b = np.zeros((self.pool_size, 1, self.w, self.h, self.c))
    
    def input_setup(self):
        input_dir = os.path.join('../datasets/', self.name)
        train_a_dir = os.path.join(input_dir, 'trainA', '*.tfrecord')
        train_b_dir = os.path.join(input_dir, 'trainB', '*.tfrecord')
        
        train_a_names = tf.train.match_filenames_once(train_a_dir)
        train_b_names = tf.train.match_filenames_once(train_b_dir)
        
        self.n_train_a = tf.size(train_a_names)
        self.n_train_b = tf.size(train_b_names)
        self.n_train = tf.maximum(self.n_train_a, self.n_train_b)
        
        #train_a_queue = tf.train.string_input_producer(train_a_names)
        #train_b_queue = tf.train.string_input_producer(train_b_names)
        
        #reader = tf.WholeFileReader()
        #_, train_a_file = reader.read(train_a_queue)
        #_, train_b_file = reader.read(train_b_queue)
        
        #image_a = tf.reshape(tf.decode_raw(train_a_file, tf.float32), 
        #                     [self.w, self.h, self.c])
        #image_b = tf.reshape(tf.decode_raw(train_b_file, tf.float32), 
        #                     [self.w, self.h, self.c])
        
        image_a = read_from_tfrecord(train_a_names)
        image_b = read_from_tfrecord(train_b_names)
        
        #train_a_queue = tf.train.string_input_producer(train_a_names)
        #train_b_queue = tf.train.string_input_producer(train_b_names)
        
        #reader = tf.TFRecordReader()
        #_, train_a_file = reader.read(train_a_queue)
        #_, train_b_file = reader.read(train_b_queue)
        
        #features = {'shape': tf.FixedLenFeature([], tf.string), 
        #            'array': tf.FixedLenFeature([], tf.string)}
        #features_a = tf.parse_single_example(train_a_file, features=features, 
        #                                     name='features_a')
        #features_b = tf.parse_single_example(train_b_file, features=features, 
        #                                     name='features_b')
        
        #image_a = tf.decode_raw(features_a['array'], tf.float32)
        #image_b = tf.decode_raw(features_b['array'], tf.float32)
        
        #shape_a = tf.decode_raw(features_a['shape'], tf.int32)
        #shape_b = tf.decode_raw(features_b['shape'], tf.int32)
        
        #image_a = tf.reshape(image_a, shape_a)
        #image_b = tf.reshape(image_b, shape_b)
        
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
        self.input_a = tf.random_crop(self.input_a, [self.h, self.w, self.c])
        self.input_b = tf.random_crop(self.input_b, [self.h, self.w, self.c])
        
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
    
    def input_setup_3d(self):
        input_dir = os.path.join('../datasets/', self.name)
        train_a_dir = os.path.join(input_dir, 'trainA_whole', '*.tfrecord')
        train_b_dir = os.path.join(input_dir, 'trainB_whole', '*.tfrecord')
        
        train_a_names = tf.train.match_filenames_once(train_a_dir)
        train_b_names = tf.train.match_filenames_once(train_b_dir)
        
        train_a_queue = tf.train.string_input_producer(train_a_names)
        train_b_queue = tf.train.string_input_producer(train_b_names)
        
        reader = tf.WholeFileReader()
        _, train_a_file = reader.read(train_a_queue)
        _, train_b_file = reader.read(train_b_queue)
        
        image_a = tf.reshape(tf.decode_raw(train_a_file, tf.float32), 
                             [self.d, self.w, self.h, self.c])
        image_b = tf.reshape(tf.decode_raw(train_b_file, tf.float32), 
                             [self.d, self.w, self.h, self.c])
        
        #Resize without scaling
        self.input_a_3d = tf.image.resize_image_with_crop_or_pad(image_a, 
                                                                 self.h, 
                                                                 self.w)
        self.input_b_3d = tf.image.resize_image_with_crop_or_pad(image_b, 
                                                                 self.h, 
                                                                 self.w)
        
        #Normalize values: -1 to 1
        denom = (self.max - self.min) / 2
        self.input_a_3d = \
            tf.subtract(tf.divide(tf.subtract(self.input_a_3d, 
                                              self.min), denom), 1)
        self.input_b_3d = \
            tf.subtract(tf.divide(tf.subtract(self.input_b_3d, 
                                              self.min), denom), 1)
        
        #Shuffle batch
        self.batch_a_3d, self.batch_b_3d = \
            tf.train.shuffle_batch([self.input_a_3d, self.input_b_3d], 
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
        
        #self.global_step = slim.get_or_create_global_step()
        self.global_step = 0
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
            self.p_fake_pool_a = discriminator(self.fake_pool_a, scope='d_a')
            self.p_fake_pool_b = discriminator(self.fake_pool_b, scope='d_b')
    
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
        d_a_vars = [v for v in self.vars if 'd_a' in v.name]
        d_b_vars = [v for v in self.vars if 'd_b' in v.name]
        g_a_vars = [v for v in self.vars if 'g_a' in v.name]
        g_b_vars = [v for v in self.vars if 'g_b' in v.name]
        
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
    
    def fake_pool(self, fake, pool):
        if self.n_fake < self.pool_size:
            pool[self.n_fake] = fake
            return fake
        else:
            p = random.random()
            if p < 0.5:
                index = random.randint(0, self.pool_size - 1)
                temp = pool[index]
                pool[index] = fake
                return temp
            else:
                return fake
    
    def save_train_images(self, sess, epoch):
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        for i in range(0, self.n_save):
            inputs = sess.run([self.input_a_3d, self.input_b_3d, 
                               self.batch_a_3d, self.batch_b_3d])
            _, _, batch_a, batch_b = inputs
            fake_a_temp, fake_b_temp, cyc_a_temp, cyc_b_temp = \
                sess.run([self.fake_a, self.fake_b, self.cycle_a, self.cycle_b],
                         feed_dict={self.real_a: batch_a, self.real_b: batch_b})
            tensors = [batch_a, batch_b, fake_b_temp, fake_a_temp, cyc_a_temp, 
                       cyc_b_temp]
            #TODO: Save the image
    
    def train(self):
        self.input_setup()
        self.input_setup_3d()
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
                #TODO: parse checkpoint file for initial global_step
            if not os.path.exists(self.train_output):
                os.makedirs(self.train_output)
            writer = tf.summary.FileWriter(self.train_output)
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            
            #for epoch in range(sess.run(self.global_step), self.max_step):
            for epoch in range(self.global_step, self.max_step):
                saver.save(sess, self.ckpt_dir, global_step=epoch)
                
                if epoch < 100:
                    current_lr = self.base_lr
                else:
                    current_lr = self.base_lr - self.base_lr * (epoch - 100)/100
                
                #TODO: Save training images
                
                total = sess.run(self.n_train)
                for i in range(0, total):
                    print('Epoch {}: Image {}/{}'.format(epoch, i, total))
                    inputs = sess.run([self.input_a, self.input_b, 
                                       self.batch_a, self.batch_b])
                    input_a, input_b, batch_a, batch_b = inputs
                    
                    #Optimize G_A
                    run_list = [self.g_a_train, self.fake_b, 
                                self.g_a_loss_summary]
                    feed_dict = {self.real_a: batch_a, self.real_b: batch_b, 
                                 self.lr: current_lr}
                    _, fake_b_temp, summary = sess.run(run_list, feed_dict)
                    writer.add_summary(summary, epoch * total + i)
                    
                    #Sample from fake B pool
                    fake_b_sample = self.fake_pool(fake_b_temp, self.fake_b)
                    
                    #Optimize D_B
                    run_list = [self.d_b_train, self.d_b_loss_summary]
                    feed_dict = {self.real_a: batch_a, self.real_b: batch_b, 
                                 self.lr: current_lr, 
                                 self.fake_pool_b: fake_b_sample}
                    _, summary = sess.run(run_list, feed_dict)
                    writer.add_summary(summary, epoch * total + i)
                    
                    #Optimize G_B
                    run_list = [self.g_b_train, self.fake_a, 
                                self.g_b_loss_summary]
                    feed_dict = {self.real_a: batch_a, self.real_b: batch_b, 
                                 self.lr: current_lr}
                    _, fake_a_temp, summary = sess.run(run_list, feed_dict)
                    writer.add_summary(summary, epoch * total + i)
                    
                    #Sample from fake A pool
                    fake_a_sample = self.fake_pool(fake_a_temp, self.fake_a)
                    
                    #Optimize D_A
                    run_list = [self.d_a_train, self.d_a_loss_summary]
                    feed_dict = {self.real_a: batch_a, self.real_b: batch_b, 
                                 self.lr: current_lr, 
                                 self.fake_pool_a: fake_a_sample}
                    _, summary = sess.run(run_list, feed_dict)
                    writer.add_summary(summary, epoch * total + i)
                    
                    writer.flush()
                    self.n_fake += 1
                    
                #sess.run(tf.assign(self.global_step, epoch + 1))
                self.global_step += 1
                
            coord.request_stop()
            coord.join(threads)
            writer.add_graph(sess.graph)
    
    def test(self, filename):
        self.input_setup_3d()
        self.setup()
        
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            ckpt_name = tf.train.latest_checkpoint(self.ckpt_dir)
            saver.restore(sess, ckpt_name)
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            
            
            
            coord.request_stop()
            coord.join(threads)

