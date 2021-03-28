import tensorflow as tf
import numpy as np
import time
import os
from scipy.io import loadmat
from utils import *
import scipy.misc
from ops import *

class Model(object):

    def __init__(self,
                 sess,
                 is_train,
                 image_size,
                 c_dim,
                 batch_size,
                 D,
                 C,
                 G,
                 G0,
                 kernel_size
                 ):

        self.sess = sess
        self.is_train = is_train
        self.image_size = image_size
        self.c_dim = c_dim
        self.batch_size = batch_size
        self.D = D
        self.C = C
        self.G = G
        self.G0 = G0
        self.kernel_size = kernel_size

        images_shape = [None, self.image_size, self.image_size, self.c_dim]
        labels_shape = [None, self.image_size, self.image_size, self.c_dim]

        self.input_image = tf.placeholder('float32', images_shape, name='input_image_to_generator')
        self.label_image = tf.placeholder('float32', labels_shape, name='label_image_to_generator')

        self.predict_image = self.generator(self.input_image, is_train=True, reuse=False)

        self.digit_real = self.discriminator(self.label_image, is_train=True, reuse=False)
        self.digit_false = self.discriminator(self.predict_image, is_train=True, reuse=True)
        
        scale = 1.0
        hp = np.zeros([4096, 8, 8, 1], dtype=np.float32) #120*120*16/64
        hp[:, :, :, 0] = np.array([[0.0, scale, scale, scale, scale, scale, scale, scale],
                             [scale, scale, scale, scale, scale, scale, scale, scale],
                             [scale, scale, scale, scale, scale, scale, scale, scale],
                             [scale, scale, scale, scale, scale, scale, scale, scale],
                             [scale, scale, scale, scale, scale, scale, scale, scale],
                             [scale, scale, scale, scale, scale, scale, scale, scale],
                             [scale, scale, scale, scale, scale, scale, scale, scale],
                             [scale, scale, scale, scale, scale, scale, scale, scale]], dtype=np.float32)
        self.mask = tf.Variable(hp, trainable=False, name='Mask')
        
        hp_1 = np.zeros([225, 8, 8, 1], dtype=np.float32)
        hp_1[:, :, :, 0] = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, scale],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, scale, scale],
                             [0.0, 0.0, 0.0, 0.0, 0.0, scale, scale, scale],
                             [0.0, 0.0, 0.0, 0.0, scale, scale, scale, scale],
                             [0.0, 0.0, 0.0, scale, scale, scale, scale, scale],
                             [0.0, 0.0, scale, scale, scale, scale, scale, scale],
                             [0.0, scale, scale, scale, scale, scale, scale, scale]], dtype=np.float32)
        self.mask_1 = tf.Variable(hp_1, trainable=False, name='Mask')

        self.saver = tf.train.Saver(max_to_keep=0)

    def RDBs(self, input_layer, is_train=True, reuse=False):

        rdb_concat = list()
        rdb_in = input_layer
        w_init = tf.random_normal_initializer(stddev=0.02)

        with tf.variable_scope("rdbs", reuse=reuse) as vs:
            for i in range(1, self.D+1):
                x = rdb_in  
                for j in range(1, self.C+1):
                      
                    tmp = tf.layers.conv2d(x, self.G, self.kernel_size, padding='same', activation=tf.nn.relu, kernel_initializer=w_init, 
	            		                         name='con_1_%d_%d' % (i, j), reuse=reuse)
                    x = tf.concat([x, tmp], axis=3)

                x = tf.layers.conv2d(x, self.G, 1, padding='same', kernel_initializer=w_init, 
	            	                    name='con_2_%d_%d' % (i, self.C + 1), reuse=reuse)
	            
                rdb_in = tf.add(x, rdb_in)
                rdb_concat.append(rdb_in)

        return tf.concat(rdb_concat, axis=3)

    def divide_block(self, input_mtx, img_size, block_size): # tested [1,256,256,1]->[1024,8,8,1]

        step = int(img_size/block_size)

        bs = tf.shape(input_mtx)
        batch_size = bs[0]
        output = tf.slice(input_mtx, [0, 0, 0, 0], [batch_size, block_size, block_size, self.c_dim])
        

        for ix in range(step):
            for iy in range(step):
                if ix == 0 and iy == 0:
                    continue
                tmp = tf.slice(input_mtx, [0, ix*block_size, iy*block_size, 0], [batch_size, block_size, block_size, self.c_dim])

                output = tf.concat([output, tmp], 0)

        return output

    def dct_trans(self, input_tensor):# tested [?,8,8]->[?,8,8]

        tmp_tensor = tf.einsum('ij,ljkm->likm', self.dctmtx, input_tensor, name='dct_mul_1')
        tmp_tensor = tf.einsum('ijkl,km->ijml', tmp_tensor, tf.transpose(self.dctmtx),name='dct_mul_2')

        return tmp_tensor


    def fun_DCT(self, cover_tensor, img_size): #tested [batch_size,256,256,1]->[batch_size,256,256,1]

        crop_tmp = self.divide_block(cover_tensor, img_size, 8)
        combine_tmp = self.dct_trans(crop_tmp)
	    
        return combine_tmp
    
    
    def calibrated_feature(self, x, x_cal):
        x_dct = self.fun_DCT(x, 120)
        x_cal_dct = self.fun_DCT(x_cal, 120)
        
        x_dct = x_dct * self.mask_1
        x_cal_dct = x_cal_dct * self.mask_1
        
        x_dct_mean = tf.reduce_mean(x_dct, axis=0)
        x_cal_dct_mean = tf.reduce_mean(x_cal_dct, axis=0)
        
        x_dct = x_dct - x_dct_mean
        x_cal_dct = x_cal_dct - x_cal_dct_mean
        
        x_dct = tf.square(x_dct)
        x_cal_dct = tf.square(x_cal_dct)
        
        x_dct_var = tf.reduce_mean(x_dct, axis=0)
        x_cal_dct_var = tf.reduce_mean(x_cal_dct, axis=0)
        
        cali_fea = tf.abs(x_dct_var - x_cal_dct_var)
        
        cali_fea = tf.reduce_mean(cali_fea)
        return cali_fea
        
    
    def compute_calibrated_feature(self, t_image):
        x = tf.slice(t_image, [0, 0, 0, 0], [self.batch_size, 120, 120, self.c_dim])
        x_cal = tf.slice(t_image, [0, 4, 4, 0], [self.batch_size, 120, 120, self.c_dim])
        
        x_list = []
        x_cal_list = []
        
        for i in range(self.batch_size):
            x_list.append(tf.slice(x, [i, 0, 0, 0], [1, 120, 120, self.c_dim]))
            x_cal_list.append(tf.slice(x_cal, [i, 0, 0, 0], [1, 120, 120, self.c_dim]))
        
        cali_feature_sum = 0
        
        for i in range(self.batch_size):
            cali_feature_sum += self.calibrated_feature(x_list[i], x_cal_list[i])
        
        cali_feature = cali_feature_sum / self.batch_size
        
        return cali_feature
    def google_attention(self, x, channels, scope='attention'):
        with tf.variable_scope(scope):
            batch_size, height, width, num_channels = x.get_shape().as_list()
            f = conv(x, channels // 8, kernel=1, stride=1, sn=True, scope='f_conv')  # [bs, h, w, c']
            f = max_pooling(f)

            g = conv(x, channels // 8, kernel=1, stride=1, sn=True, scope='g_conv')  # [bs, h, w, c']

            h = conv(x, channels // 2, kernel=1, stride=1, sn=True, scope='h_conv')  # [bs, h, w, c]
            h = max_pooling(h)

            # N = h * w
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

            beta = tf.nn.softmax(s)  # attention map

            o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=[batch_size, height, width, num_channels // 2])  # [bs, h, w, C]
            o = conv(o, channels, kernel=1, stride=1, sn=True, scope='attn_conv')
            x = gamma * o + x

        return x

    def attention(self, x, channels, scope='attention'):
        with tf.variable_scope(scope):
            f = conv(x, channels // 8, kernel=1, stride=1, sn=True, scope='f_conv') # [bs, h, w, c']
            g = conv(x, channels // 8, kernel=1, stride=1, sn=True, scope='g_conv') # [bs, h, w, c']
            h = conv(x, channels, kernel=1, stride=1, sn=True, scope='h_conv') # [bs, h, w, c]

            # N = h * w
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

            beta = tf.nn.softmax(s)  # attention map

            o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            x_shape = tf.shape(x)
            o = tf.reshape(o, x_shape) # [bs, h, w, C]
            o = conv(o, channels, kernel=1, stride=1, sn=True, scope='attn_conv')

            x = gamma * o + x

        return x

    def generator(self, t_image, is_train=True, reuse=False):

        w_init = tf.random_normal_initializer(stddev=0.02)
        with tf.variable_scope("gen", reuse=reuse) as vs:
            F_1 = tf.layers.conv2d(t_image, self.G0, 9, padding='same', kernel_initializer=w_init, 
	    		                          name='F_1_conv', reuse=reuse)
            
            F0 = tf.layers.conv2d(F_1, self.G, 5, padding='same', kernel_initializer=w_init, 
	    		                         name='F_0_conv', reuse=reuse)


            FD = self.RDBs(F0)
            FGF1 = tf.layers.conv2d(FD, self.G0, 1, padding='same', kernel_initializer=w_init,
            	                       name='FGF1_conv', reuse=reuse)

            FGF2 = tf.layers.conv2d(FGF1, self.G0, self.kernel_size, padding='same', kernel_initializer=w_init,
            	                       name='FGF2_conv', reuse=reuse)
            #FGF3 = self.attention(FGF2, channels=self.G0, scope='self_attention')
            #FGF3 = non_local.sn_non_local_block_sim(FGF2, None, name='g_non_local')

            FDF = tf.add(FGF2, F_1)

            """output = tf.layers.conv2d(FDF, self.c_dim, self.kernel_size, padding='same', kernel_initializer=w_init, 
	        	                           name='output_conv', reuse=reuse)"""
            output = conv(FDF, channels=self.c_dim, kernel=3, stride=1, pad=1, pad_type='reflect', scope='g_logit')
            
            output = tf.nn.tanh(output)

        return output

    def discriminator(self, t_image, is_train=True, reuse=False):

        #t_image = self.divide_block(t_image, self.image_size, 16)
        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init=tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope("dis", reuse=reuse) as vs:
            t_image = tf.layers.conv2d(t_image, 32, 3, (2, 2), padding='same', kernel_initializer=w_init, 
    			                              name='con1', reuse=reuse)
            t_image = tf.contrib.layers.layer_norm(t_image, trainable=is_train)
            t_image = tf.nn.relu(t_image) #64

            t_image = tf.layers.conv2d(t_image, 32, 3, (2, 2), padding='same', kernel_initializer=w_init, 
    			                              name='con2', reuse=reuse)
            t_image = tf.contrib.layers.layer_norm(t_image, trainable=is_train)
            t_image = tf.nn.relu(t_image) #32
            #t_image = non_local.sn_non_local_block_sim(t_image, None, name='d_non_local')
            #t_image = self.attention(t_image, channels=32, scope='self_attention')

            t_image = tf.layers.conv2d(t_image, 64, 3, (2, 2), padding='same', kernel_initializer=w_init, 
    			                              name='con3', reuse=reuse)
            t_image = tf.contrib.layers.layer_norm(t_image, trainable=is_train)
            t_image = tf.nn.relu(t_image) #16

            t_image = tf.layers.conv2d(t_image, 64, 3, (2, 2), padding='same', kernel_initializer=w_init, 
    			                              name='con4', reuse=reuse)
            t_image = tf.contrib.layers.layer_norm(t_image, trainable=is_train)
            t_image = tf.nn.relu(t_image) #8

            t_image = tf.layers.conv2d(t_image, 128, 3, (2, 2), padding='same', kernel_initializer=w_init, 
    			                              name='con5', reuse=reuse)
            t_image = tf.contrib.layers.layer_norm(t_image, trainable=is_train)
            #t_image = tf.nn.relu(t_image) #4
            x = lrelu(t_image, 0.2)

            x = global_sum_pooling(x)

            x = fully_connected(x, units=1, sn=True, scope='d_logit')

            
            """t_image = tf.layers.conv2d(t_image, 128, 3, (2, 2), padding='same', kernel_initializer=w_init, 
    			                              name='con6', reuse=reuse)
            t_image = tf.contrib.layers.layer_norm(t_image, trainable=is_train)
            t_image = tf.nn.relu(t_image) #2

            t_image = tf.layers.conv2d(t_image, 256, 3, padding='same', kernel_initializer=w_init, 
    			                              name='con7', reuse=reuse)
            t_image = tf.contrib.layers.layer_norm(t_image, trainable=is_train)
            t_image = tf.nn.relu(t_image)

            t_image = tf.layers.conv2d(t_image, 1, 2, padding='valid', kernel_initializer=w_init, 
    			                               name='con8', reuse=reuse)

            output = tf.reshape(t_image, [-1, 1], name='output')"""

        return x

    def train(self, config):

        print("\nPrepare Data...\n")


        data_dir = get_data_dir(config.checkpoint_dir, True)
        data_num = get_data_num(data_dir)

        self.dctmtx = tf.convert_to_tensor(loadmat('./dctmtx8.mat')['ans'], dtype=tf.float32)     
         
        self.label_dct = self.fun_DCT(self.label_image, config.image_size)
        self.predict_dct = self.fun_DCT(self.predict_image, config.image_size)
        
        self.label_dct = self.label_dct * self.mask
        self.predict_dct = self.predict_dct * self.mask
     
        self.label_mean = tf.reduce_mean(self.label_dct, axis=0)
        self.predict_mean = tf.reduce_mean(self.predict_dct, axis=0)

        self.label_dct = self.label_dct - self.label_mean
        self.predict_dct = self.predict_dct - self.predict_mean

        self.eps = tf.random_uniform([1], minval=-1., maxval=1.)
        self.x_inter = self.eps * self.label_image + (1. - self.eps) * self.predict_image
        self.grad = tf.gradients(self.discriminator(self.x_inter, is_train=True, reuse=True), [self.x_inter])[0]
        self.grad_norm = tf.sqrt(tf.reduce_sum((self.grad)**2, axis=1))
        self.grad_pen = 10 * tf.reduce_mean(tf.nn.relu(self.grad_norm - 1.))

        #self.d_loss = tf.reduce_mean(self.digit_false - self.digit_real) + self.grad_pen
        self.real_loss = tf.reduce_mean(relu(1.0 - self.digit_real))
        self.fake_loss = tf.reduce_mean(relu(1.0 + self.digit_false))
        self.d_loss = self.real_loss + self.fake_loss

        alpha = 1.0
        beta = 1.0
        sigma = 50.0
        gama = 1e-5

        self.mse = alpha * tf.reduce_mean(tf.square(self.label_image - self.predict_image))
        self.dct_loss = beta * tf.reduce_mean(tf.square(self.label_dct - self.predict_dct))
        self.cali_loss = sigma * self.compute_calibrated_feature(self.predict_image)
        self.adv_loss = -gama * tf.reduce_mean(self.digit_false)

        self.g_loss_1 = self.mse + self.dct_loss 
        self.g_loss_2 = self.mse + self.dct_loss  + self.adv_loss
        self.g_loss = self.g_loss_2

        self.g_vars = [var for var in tf.global_variables() if 'gen' in var.name]
        self.d_vars = [var for var in tf.global_variables() if 'dis' in var.name]

        self.g_optim_1 = tf.train.AdamOptimizer(config.g_lr).minimize(self.g_loss_1, var_list=self.g_vars)
        self.g_optim_2 = tf.train.AdamOptimizer(config.g_lr).minimize(self.g_loss_2, var_list=self.g_vars)
        self.d_optim = tf.train.AdamOptimizer(config.d_lr).minimize(self.d_loss, var_list=self.d_vars)


        self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d_vars]
        tf.global_variables_initializer().run(session=self.sess)

        ep = 1
        ite = 0
        self.load(config.checkpoint_mse_dctFullHpCov_noCali_noBlurTrain_Hinge, ite)
        print("\nNow Start Training...\n")
        ite += 1
        
        d_loss_sum = 0
        g_loss_sum, mse_sum, dct_loss_sum, cali_loss_sum, adv_loss_sum = 0, 0, 0, 0, 0
        
        while ep < config.epoch:
            # Run by batch images
            print('\n')
            batch_idxs = data_num // config.batch_size
            
            for idx in range(0, batch_idxs):
                batch_input_images, batch_label_images = get_batch(data_dir, data_num, config.batch_size)
                
                if ite > 10000:
                    _, d_loss_temp = self.sess.run([self.d_optim, self.d_loss], 
                	                            feed_dict={self.input_image: batch_input_images, self.label_image: batch_label_images})
                    

                    _, g_loss_temp, mse_temp, dct_loss_temp, cali_loss_temp, adv_loss_temp = self.sess.run([self.g_optim_2, self.g_loss, self.mse, self.dct_loss, self.cali_loss, self.adv_loss],
                     	                                                                                     feed_dict={self.input_image: batch_input_images, self.label_image: batch_label_images})
                else:
                    _, g_loss_temp, mse_temp, dct_loss_temp, cali_loss_temp, adv_loss_temp = self.sess.run([self.g_optim_1, self.g_loss, self.mse, self.dct_loss, self.cali_loss, self.adv_loss],
                     	                                                                                     feed_dict={self.input_image: batch_input_images, self.label_image: batch_label_images})
                    
                    d_loss_temp = 0


                d_loss_sum += d_loss_temp
                g_loss_sum += g_loss_temp
                mse_sum += mse_temp
                dct_loss_sum += dct_loss_temp
                cali_loss_sum += cali_loss_temp
                adv_loss_sum += adv_loss_temp
                
                if ite % 50 == 0:
                    print("iteration: [%d], d_loss: [%8f], g_loss: [%.8f], mse: [%.8f], dct_loss: [%.8f], cali_loss: [%8f], adv_loss: [%8f]" % 
                           (ite, d_loss_sum/50.0, g_loss_sum/50.0, mse_sum/50.0, dct_loss_sum/50.0, cali_loss_sum/50.0, adv_loss_sum/50.0))
                    
                    f = open("train_mse_dctFullHpCov_noCali_noBlurTrain_Hinge.txt", "a")
                    print("iteration: [%d], d_loss: [%8f], g_loss: [%.8f], mse: [%.8f], dct_loss: [%.8f], cali_loss: [%8f], adv_loss: [%8f]" % (ite, d_loss_sum/50.0, g_loss_sum/50.0, mse_sum/50.0, dct_loss_sum/50.0, cali_loss_sum/50.0, adv_loss_sum/50.0),file=f)
                    f.close()
                    d_loss_sum = 0
                    g_loss_sum, mse_sum, dct_loss_sum, cali_loss_sum, adv_loss_sum = 0, 0, 0, 0, 0
                
                if ite % 500 == 0:
                    self.save(config.checkpoint_mse_dctFullHpCov_noCali_noBlurTrain_Hinge, ite)

                    val_data_dir = get_data_dir(config.checkpoint_dir, False)
                    val_data_num = get_data_num(val_data_dir)

                    val_batch_idxs = val_data_num // config.batch_size
            
                    val_d_loss_sum = 0
                    val_g_loss_sum, val_mse_sum, val_dct_sum, val_cali_sum, val_adv_sum = 0, 0, 0, 0, 0
                    
                    #val_batch_idxs = 5
                    for idx in range(0, val_batch_idxs):
                        val_batch_input_images, val_batch_label_images = get_batch_val(val_data_dir, val_data_num, config.batch_size, idx * config.batch_size)

                        val_d_loss_temp, val_g_loss_temp, val_mse_temp, val_dct_temp, val_cali_temp, val_adv_temp = self.sess.run([self.d_loss, self.g_loss, self.mse, self.dct_loss, self.cali_loss, self.adv_loss], 
                	                                                                                                  feed_dict={self.input_image: val_batch_input_images, self.label_image: val_batch_label_images})
                        val_d_loss_sum += val_d_loss_temp
                        val_g_loss_sum += val_g_loss_temp
                        val_mse_sum += val_mse_temp
                        val_dct_sum += val_dct_temp
                        val_cali_sum += val_cali_temp
                        val_adv_sum += val_adv_temp

                    print('\n')
                    print("Validation: iteration: [%d], d_loss: [%8f], g_loss: [%.8f], mse: [%.8f], dct_loss: [%.8f], cali_loss: [%8f], adv_loss: [%8f]" % 
            	             (ite, val_d_loss_sum/val_batch_idxs, val_g_loss_sum/val_batch_idxs, val_mse_sum/val_batch_idxs, val_dct_sum/val_batch_idxs, val_cali_sum/val_batch_idxs, val_adv_sum/val_batch_idxs))
                    print('\n')
            
                    f = open("val_mse_dctFullHpCov_noCali_noBlurTrain_Hinge.txt", "a")
                    print("Validation: iteration: [%d], d_loss: [%8f], g_loss: [%.8f], mse: [%.8f], dct_loss: [%.8f], cali_loss: [%8f], adv_loss: [%8f]" %  (ite, val_d_loss_sum/val_batch_idxs, val_g_loss_sum/val_batch_idxs, val_mse_sum/val_batch_idxs, val_dct_sum/val_batch_idxs, val_cali_sum/val_batch_idxs, val_adv_sum/val_batch_idxs),file=f)
                    f.close()
                
                ite += 1
            
            ep += 1


    def test(self, config):

        print("\nPrepare Data...\n")
        data_dir = os.path.join(os.getcwd(), "jpeg_25/size512/jpeg/BossBase/")

        print("\nNow Start Testing...\n")
        tf.global_variables_initializer().run(session=self.sess)
        self.load(config.checkpoint_dir, 54000)
        for idx in range(8001, 10001):
            input_ = imread(data_dir + str(idx) + '.jpg')
            input_ = (input_ / 255.0) * 2.0 - 1.0
            input_ = input_[np.newaxis, :, :, np.newaxis]


            result = self.sess.run([self.predict_image], feed_dict={self.input_image: input_})


            x = (np.squeeze(result) + 1.0) / 2.0 * 255 
            x = np.clip(x, 0, 255)
            x = np.uint8(x)

            if not os.path.isdir(os.path.join(os.getcwd(), "jpeg_25/size512/new_gan/BossBase")):
                os.makedirs(os.path.join(os.getcwd(), "jpeg_25/size512/new_gan/BossBase"))
            result_dir = os.path.join(os.getcwd(), "jpeg_25/size512/new_gan/BossBase/")
            scipy.misc.imsave(result_dir + str(idx) + '.png', x)
        
        self.sess.close()



    def load(self, checkpoint_dir, ep):
        print("\nReading Checkpoints.....\n")
        model_dir = "%s_%s" % ("newgan", ep)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
 
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            print("\nCheckpoint Loading Success! %s\n" % ckpt_path)
        else:
            print("\nCheckpoint Loading Failed! \n")


    def save(self, checkpoint_dir, ep):
        model_dir = "%s_%s" % ("newgan", ep)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, 'model.ckpt'))



