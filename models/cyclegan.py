import numpy as np
import os
import random
import tensorflow as tf

from datetime import datetime
from glob import glob

from models import *
from util.loss import *
from util.nifti_to_binary import npy_to_nifti, read_from_tfrecord

class CycleGAN:
    def __init__(self, input_w, input_h, min_vox, max_vox, name, lambda_a, 
                 lambda_b, pool_size, base_lr, max_step, n_save, batch_size, 
                 is_train, restore_ckpt, train_dir, test_dir, input_dir, 
                 timestamp, test_ids_a, test_ids_b):
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        
        #Image dimensions
        self.input_w, self.input_h = input_w, input_h
        self.w, self.h, self.c = 256, 256, 1
        
        #Min/max voxels
        self.min, self.max = min_vox, max_vox
        
        #Dataset name
        self.name = name
        
        #Training parameters
        self.restore_ckpt = restore_ckpt
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.pool_size = pool_size
        self.base_lr = base_lr
        self.max_step = max_step
        self.n_save = n_save
        self.save_training_images = n_save > 0
        self.batch_size = 1
        
        #Training/testing file paths
        self.input_dir = input_dir
        if is_train:
            if restore_ckpt:
                self.train_output = os.path.join(train_dir, timestamp)
            else:
                self.train_output = os.path.join(train_dir, current_time)
            
            self.ckpt_dir = os.path.join(self.train_output, 'checkpoints')
            self.npy_dir = os.path.join(self.train_output, 'npy')
            self.img_dir = os.path.join(self.train_output, 'images')
        else:
            self.ckpt_dir = os.path.join(train_dir, timestamp)
            self.test_output = os.path.join(test_dir, current_time)
            self.npy_dir = os.path.join(self.test_output, 'npy')
            self.img_dir = os.path.join(self.test_output, 'images')
        
        #Initalize fake images
        self.fake_a = np.zeros((self.pool_size, 1, self.h, self.w, self.c))
        self.fake_b = np.zeros((self.pool_size, 1, self.h, self.w, self.c))
        
        #Randomly select training subjects to save
        ids_a = glob(os.path.join(input_dir, 'trainA', '*.tfrecord'))
        ids_b = glob(os.path.join(input_dir, 'trainB', '*.tfrecord'))
        ids_a = set([os.path.basename(x).split('-')[0] for x in ids_a])
        ids_b = set([os.path.basename(x).split('-')[0] for x in ids_b])
        ids_a = random.sample(ids_a, n_save)
        ids_b = random.sample(ids_b, n_save)
        self.train_ids = {'A': ids_a, 'B': ids_b}
        
        #Test IDs
        self.test_ids = {'A': test_ids_a, 'B': test_ids_b}
        
    def input_setup(self):
        train_a_dir = os.path.join(self.input_dir, 'trainA', '*.tfrecord')
        train_b_dir = os.path.join(self.input_dir, 'trainB', '*.tfrecord')
        
        train_a_names = tf.train.match_filenames_once(train_a_dir)
        train_b_names = tf.train.match_filenames_once(train_b_dir)
        
        self.n_train_a = tf.size(train_a_names)
        self.n_train_b = tf.size(train_b_names)
        self.n_train = tf.maximum(self.n_train_a, self.n_train_b)
        
        image_a = read_from_tfrecord(train_a_names)
        image_b = read_from_tfrecord(train_b_names)
        
        #Resize without scaling
        image_a = tf.image.resize_image_with_crop_or_pad(image_a, self.input_h, 
                                                         self.input_w)
        image_b = tf.image.resize_image_with_crop_or_pad(image_b, self.input_h, 
                                                         self.input_w)
        
        #Randomly flip
        image_a = tf.image.random_flip_left_right(image_a)
        image_b = tf.image.random_flip_left_right(image_b)
        
        #Randomly crop
        image_a = tf.random_crop(image_a, [self.h, self.w, self.c])
        image_b = tf.random_crop(image_b, [self.h, self.w, self.c])
        
        #Reshape to batch_size, h, w, c
        self.input_a = tf.reshape(image_a, [1, self.h, self.w, self.c])
        self.input_b = tf.reshape(image_b, [1, self.h, self.w, self.c])
    
    def save_images_setup(self, is_train):
        if is_train:
            path_a = [os.path.join(self.input_dir, 'trainA', 
                                   '{:s}*.tfrecord'.format(x)) 
                      for x in self.train_ids['A']]
            path_b = [os.path.join(self.input_dir, 'trainB', 
                                   '{:s}*.tfrecord'.format(x)) 
                      for x in self.train_ids['B']]
        else: 
            path_a = [os.path.join(self.input_dir, 'testA', 
                                   '{:s}*.tfrecord'.format(x)) 
                      for x in self.test_ids['A']]
            path_b = [os.path.join(self.input_dir, 'testB', 
                                   '{:s}*.tfrecord'.format(x)) 
                      for x in self.test_ids['B']]
        
        a_names = tf.train.match_filenames_once(path_a)
        b_names = tf.train.match_filenames_once(path_b)
        
        img_a = read_from_tfrecord(a_names, shuffle=False)
        img_b = read_from_tfrecord(b_names, shuffle=False)
        
        #Resize without scaling
        img_a = tf.image.resize_image_with_crop_or_pad(img_a, self.h, self.w)
        img_b = tf.image.resize_image_with_crop_or_pad(img_b, self.h, self.w)
        
        #Normalize values: -1 to 1
        denom = (self.max - self.min) / 2
        img_a = tf.subtract(tf.divide(tf.subtract(img_a, self.min), denom), 1)
        img_b = tf.subtract(tf.divide(tf.subtract(img_b, self.min), denom), 1)
        
        #Reshape to 1, h, w, c
        self.img_a = tf.reshape(img_a, [1, self.h, self.w, self.c])
        self.img_b = tf.reshape(img_b, [1, self.h, self.w, self.c])
        
        #Paths to affines of original images
        self.affine_a = os.path.join(self.input_dir, 'affineA')
        self.affine_b = os.path.join(self.input_dir, 'affineB')
        
        return a_names, b_names
    
    def setup(self):
        self.real_a = tf.placeholder(tf.float32, 
                                     [self.batch_size, self.w, self.h, self.c], 
                                     name='input_a')
        self.real_b = tf.placeholder(tf.float32, 
                                     [self.batch_size, self.w, self.h, self.c], 
                                     name='input_b')
        self.fake_pool_a = tf.placeholder(tf.float32, 
                                          [None, self.w, self.h, self.c], 
                                          name='fake_pool_a')
        self.fake_pool_b = tf.placeholder(tf.float32, 
                                          [None, self.w, self.h, self.c], 
                                          name='fake_pool_b')
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.n_fake = 0
        self.lr = tf.placeholder(tf.float32, shape=[], name='lr')
        
        self.forward()
    
    def forward(self):
        with tf.variable_scope('CycleGAN') as scope:
            #D(A), D(B)
            self.p_real_a = discriminator(self.real_a, scope='d_a')
            self.p_real_b = discriminator(self.real_b, scope='d_b')
            
            #G(A), G(B)
            self.fake_img_b = generator(self.real_a, c=self.c, scope='g_a')
            self.fake_img_a = generator(self.real_b, c=self.c, scope='g_b')
            
            scope.reuse_variables()
            
            #D(G(B)), D(G(A))
            self.p_fake_a = discriminator(self.fake_img_a, scope='d_a')
            self.p_fake_b = discriminator(self.fake_img_b, scope='d_b')
            
            #G(G(A)), G(G(B))
            self.cycle_a = generator(self.fake_img_b, c=self.c, scope='g_b')
            self.cycle_b = generator(self.fake_img_a, c=self.c, scope='g_a')
            
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
    
    def save_images(self, sess, epoch, a_names, b_names, ids):
        if not os.path.exists(self.npy_dir):
            os.makedirs(self.npy_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        
        run_list = [a_names, b_names, tf.size(a_names), tf.size(b_names)]
        a_names_list, b_names_list, n_a, n_b = sess.run(run_list)
        
        #Save G(A), G(G(A))
        for i in range(n_a):
            a_name = os.path.splitext(os.path.basename(a_names_list[i]))[0]
            in_a = sess.run(self.img_a)
            run_list = [self.fake_img_b, self.cycle_a]
            feed_dict = {self.real_a: in_a}
            fake_b_tmp, cyc_a_tmp = sess.run(run_list, feed_dict=feed_dict)
            
            #De-normalize
            fake_b_tmp = ((fake_b_tmp + 1) / 2 * (self.max - self.min) + 
                          self.min)
            cyc_a_tmp = ((cyc_a_tmp + 1) / 2 * (self.max - self.min) + 
                         self.min)
            
            #Save as .npy
            fake_b_name = 'epoch_{:d}_fake_b_{:s}'.format(epoch, a_name)
            cyc_a_name = 'epoch_{:d}_cyc_a_{:s}'.format(epoch, a_name)
            fake_b_path = os.path.join(self.npy_dir, fake_b_name)
            cyc_a_path = os.path.join(self.npy_dir, cyc_a_name)
            np.save(fake_b_path, fake_b_tmp)
            np.save(cyc_a_path, cyc_a_tmp)
        
        #Save G(B), G(G(B))
        for i in range(n_b):
            b_name = os.path.splitext(os.path.basename(b_names_list[i]))[0]
            in_b = sess.run(self.img_b)
            run_list = [self.fake_img_a, self.cycle_b]
            feed_dict = {self.real_b: in_b}
            fake_a_tmp, cyc_b_tmp = sess.run(run_list, feed_dict=feed_dict)
            
            #De-normalize
            fake_a_tmp = ((fake_a_tmp + 1) / 2 * (self.max - self.min) + 
                          self.min)
            cyc_b_tmp = ((cyc_b_tmp + 1) / 2 * (self.max - self.min) + 
                         self.min)
            
            #Save as .npy
            fake_a_name = 'epoch_{:d}_fake_a_{:s}'.format(epoch, b_name)
            cyc_b_name = 'epoch_{:d}_cyc_b_{:s}'.format(epoch, b_name)
            fake_a_path = os.path.join(self.npy_dir, fake_a_name)
            cyc_b_path = os.path.join(self.npy_dir, cyc_b_name)
            np.save(fake_a_path, fake_a_tmp)
            np.save(cyc_b_path, cyc_b_tmp)
        
        #Save as nifti
        for subject in ids['A']:
            npy_to_nifti(subject, self.npy_dir, self.affine_a, self.img_dir, 
                         'epoch_{:d}_fake_b_{:s}'.format(epoch, subject))
            npy_to_nifti(subject, self.npy_dir, self.affine_a, self.img_dir, 
                         'epoch_{:d}_cyc_a_{:s}'.format(epoch, subject))
            
        for subject in ids['B']:
            npy_to_nifti(subject, self.npy_dir, self.affine_b, self.img_dir, 
                         'epoch_{:d}_fake_a_{:s}'.format(epoch, subject))
            npy_to_nifti(subject, self.npy_dir, self.affine_b, self.img_dir, 
                         'epoch_{:d}_cyc_b_{:s}'.format(epoch, subject))
    
    def train(self):
        self.input_setup()
        self.setup()
        self.loss()
        
        if self.save_training_images:
            a_names, b_names = self.save_images_setup(is_train=True)
        
        init = (tf.global_variables_initializer(), 
                tf.local_variables_initializer())
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(init)
            
            if self.restore_ckpt:
                ckpt_name = tf.train.latest_checkpoint(self.train_output)
                saver.restore(sess, ckpt_name)
            if not os.path.exists(self.train_output):
                os.makedirs(self.train_output)
            writer = tf.summary.FileWriter(self.train_output)
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            
            for epoch in range(sess.run(self.global_step), self.max_step):
                saver.save(sess, self.ckpt_dir, global_step=epoch)
                
                if epoch < 100:
                    current_lr = self.base_lr
                else:
                    current_lr = self.base_lr - self.base_lr * (epoch - 100)/100
                
                if self.save_training_images:
                    print('Saving images for epoch {}...'.format(epoch))
                    self.save_images(sess, epoch, a_names, b_names, 
                                     self.train_ids)
                
                total = sess.run(self.n_train)
                for i in range(0, total):
                    if i % 100 == 0:
                        print('Epoch {}: Image {}/{}'.format(epoch, i, total))
                    
                    input_a, input_b = sess.run([self.input_a, self.input_b])
                    
                    #Optimize G_A
                    run_list = [self.g_a_train, self.fake_img_b, 
                                self.g_a_loss_summary]
                    feed_dict = {self.real_a: input_a, self.real_b: input_b, 
                                 self.lr: current_lr}
                    _, fake_b_temp, summary = sess.run(run_list, feed_dict)
                    writer.add_summary(summary, epoch * total + i)
                    
                    #Sample from fake B pool
                    fake_b_sample = self.fake_pool(fake_b_temp, self.fake_b)
                    
                    #Optimize D_B
                    run_list = [self.d_b_train, self.d_b_loss_summary]
                    feed_dict = {self.real_a: input_a, self.real_b: input_b, 
                                 self.lr: current_lr, 
                                 self.fake_pool_b: fake_b_sample}
                    _, summary = sess.run(run_list, feed_dict)
                    writer.add_summary(summary, epoch * total + i)
                    
                    #Optimize G_B
                    run_list = [self.g_b_train, self.fake_img_a, 
                                self.g_b_loss_summary]
                    feed_dict = {self.real_a: input_a, self.real_b: input_b, 
                                 self.lr: current_lr}
                    _, fake_a_temp, summary = sess.run(run_list, feed_dict)
                    writer.add_summary(summary, epoch * total + i)
                    
                    #Sample from fake A pool
                    fake_a_sample = self.fake_pool(fake_a_temp, self.fake_a)
                    
                    #Optimize D_A
                    run_list = [self.d_a_train, self.d_a_loss_summary]
                    feed_dict = {self.real_a: input_a, self.real_b: input_b, 
                                 self.lr: current_lr, 
                                 self.fake_pool_a: fake_a_sample}
                    _, summary = sess.run(run_list, feed_dict)
                    writer.add_summary(summary, epoch * total + i)
                    
                    writer.flush()
                    self.n_fake += 1
                    
                sess.run(tf.assign(self.global_step, epoch + 1))
                
            coord.request_stop()
            coord.join(threads)
            writer.add_graph(sess.graph)
    
    def test(self):
        self.setup()
        
        a_names, b_names = self.save_images_setup(is_train=False)
        
        saver = tf.train.Saver()
        init = (tf.global_variables_initializer(), 
                tf.local_variables_initializer())
        
        with tf.Session() as sess:
            sess.run(init)
            ckpt_name = tf.train.latest_checkpoint(self.ckpt_dir)
            saver.restore(sess, ckpt_name)
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            
            self.save_images(sess, 0, a_names, b_names, self.test_ids)
            
            coord.request_stop()
            coord.join(threads)

