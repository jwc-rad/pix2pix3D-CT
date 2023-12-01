import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import random 

import tensorflow as tf
from keras import backend as K
from keras.layers import InputSpec
from keras.initializers import RandomNormal
from keras.layers import Layer
from keras.layers import Input, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, GaussianNoise
from keras.layers import LeakyReLU
from keras.layers import UpSampling3D, Conv3D
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

class My3dResize(Layer):
    def __init__(self, sizes, nn=False, **kwargs):
        super(My3dResize, self).__init__(**kwargs)
        self.sizes = sizes
        self.nn = nn

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1] = output_shape[1]*self.sizes[0]
        output_shape[2] = output_shape[2]*self.sizes[1]
        output_shape[3] = output_shape[3]*self.sizes[2]
        return tuple(output_shape)
    
    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()
        output_shape = input_shape.copy()
        output_shape[1] = output_shape[1]*self.sizes[0]
        output_shape[2] = output_shape[2]*self.sizes[1]
        output_shape[3] = output_shape[3]*self.sizes[2]
        
        # resize rows and columns
        
        u = tf.reshape(inputs, shape=[-1]+input_shape[1:-2]+[input_shape[-2]*input_shape[-1]])
        if self.nn:
            u = tf.image.resize(u, size=output_shape[1:3], method=tf.image.ResizeMethod.BILINEAR)
        else:
            u = tf.image.resize(u, size=output_shape[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        u = tf.reshape(u, shape=[-1]+output_shape[1:-2]+input_shape[-2:])
        
        # repeat depth-wise
        outputs = K.repeat_elements(u, self.sizes[2], axis=3)        
        return outputs
 
## modified from https://github.com/eriklindernoren/Keras-GAN/tree/master/pix2pix

class My3dPix2Pix():
    def __init__(self, data_loader, savepath='result/pilot', L_weights=(1,100), opt='adam', lrs=(0.0002,0.0), \
                 randomshift=0.1, resoutput=0.0, dropout=0.0, smoothlabel=True, gennoise=0,\
                 fmloss=False, coordconv=False, resizeconv=False, multigpu=None):
        '''
        PatchGAN's receptive field = 70x70x4
        '''
        # Configure data loader
        self.data_loader = data_loader
        self.savepath = savepath
        self.coordconv = coordconv
        self.resizeconv = resizeconv
        self.opt = opt
        self.lr_ini, self.lr_decay = lrs
        self.smoothlabel = smoothlabel
        self.gennoise = gennoise
        self.resoutput=resoutput
        self.dropout = dropout
        self.fmloss = fmloss    # discarded
        self.randomshift=randomshift
        
        self.reswindow = (self.data_loader.window2[0][0]*self.resoutput,\
                          self.data_loader.window2[0][1]-0.5*(1-self.resoutput)*self.data_loader.window2[0][0])
        
        ###  from data_loader
        # Input shape --- x,y >=64, depth >=16.
        self.img_shape = tuple(data_loader.img_shape) + (len(data_loader.window1),)
        self.img_rows, self.img_cols, self.depth = data_loader.img_shape
        self.channels = len(data_loader.window1)

        # Calculate output shape of D (PatchGAN)
        patch = max(int(self.img_rows / 2**3), 1)
        dpatch = max(int(self.depth / 2**5), 2)
        self.disc_patch = (patch, patch, dpatch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)
        if self.opt=='adam':
            optimizer = Adam(self.lr_ini, 0.5)

        # Build and compile the discriminator
        self.discriminator, self.discriminator_feat = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer, loss_weights=[0.5],
            metrics=['accuracy'])
        self.discriminator_feat.compile(loss='mae',
            optimizer=optimizer)

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        self.discriminator_feat.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        if self.fmloss:
            # discarded
            '''
            valid = self.discriminator([fake_A, img_B])
            valid_feat = self.discriminator_feat([fake_A, img_B])

            self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A, valid_feat])
            if multigpu is not None:
                self.combined = multi_gpu_model(self.combined, gpus=multigpu)

            self.combined.compile(loss=['binary_crossentropy', 'mae', 'mae'],
                                  loss_weights=list(L_weights),
                                  optimizer=optimizer)
            '''
            valid = self.discriminator([fake_A, img_B])

            self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
            if multigpu is not None:
                self.combined = multi_gpu_model(self.combined, gpus=multigpu)

            self.combined.compile(loss=['binary_crossentropy', ssim_mae_loss],
                                  loss_weights=list(L_weights),
                                  optimizer=optimizer)
            
            
        else:
            valid = self.discriminator([fake_A, img_B])

            self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
            if multigpu is not None:
                self.combined = multi_gpu_model(self.combined, gpus=multigpu)

            self.combined.compile(loss=['binary_crossentropy', 'mae'],
                                  loss_weights=list(L_weights),
                                  optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv3d(layer_input, filters, kernel_size=(4,4,2), strides=(2,2,2), bn=True):
            """Layers used during downsampling"""
            init = RandomNormal(stddev=0.02)
            d = Conv3D(filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=init)(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                #d = BatchNormalization(momentum=0.8)(d)
                d = InstanceNormalization()(d)
            return d

        def deconv3d(layer_input, skip_input, filters, kernel_size=(4,4,2), strides=(2,2,2), dropout_rate=0, bn=True):
            """Layers used during upsampling"""
            if self.resizeconv:
                u = My3dResize(strides)(layer_input)
            else:
                u = UpSampling3D(size=strides)(layer_input)
            init = RandomNormal(stddev=0.02)
            u = Conv3D(filters, kernel_size=kernel_size, strides=1, padding='same', kernel_initializer=init, activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            if bn:
                #u = BatchNormalization(momentum=0.8)(u)
                u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            u = Activation('relu')(u)
            return u

        # Image input
        d00 = Input(shape=self.img_shape)
                
        if self.coordconv:
            d0 = CoordinateChannel3D()(d00)
        else:
            d0 = d00
        
        n_layers = 7
        encoders = []
        decoders = []
        
        # Downsampling
        for i in range(n_layers):
            z = 1
            if i < self.depth.bit_length()-1: z = 2
            if i==0:
                encoders.append(conv3d(d0, self.gf, kernel_size=(4,4,z), strides=(2,2,z), bn=False))
            else:
                encoders.append(conv3d(encoders[-1], self.gf*(2**min(i,3)), kernel_size=(4,4,z), strides=(2,2,z)))
                
        # Upsampling
        for i in range(n_layers-1):
            z = 1
            if i+self.depth.bit_length()>n_layers: z = 2
            if i==0:
                decoders.append(deconv3d(encoders[-(i+1)], encoders[-(i+2)], self.gf*(2**min(n_layers-2-i,3)), 
                                         kernel_size=(4,4,z), strides=(2,2,z)))
            else:
                decoders.append(deconv3d(decoders[-1], encoders[-(i+2)], self.gf*(2**min(n_layers-2-i,3)), 
                                         kernel_size=(4,4,z), strides=(2,2,z), dropout_rate=self.dropout))

        if self.resizeconv:
            u7 = My3dResize((2,2,2))(decoders[-1])
        else:
            u7 = UpSampling3D(size=2)(decoders[-1])
        init = RandomNormal(stddev=0.02)
        output_img = Conv3D(self.channels, kernel_size=(4,4,2), strides=1, padding='same', kernel_initializer=init, activation='tanh')(u7)

        return Model(d00, output_img)


        # d1=Conv3D(filters=32,kernel_size=(3,3,1),strides=(1,1,1),padding='same')(d00)
        # d1=InstanceNormalization()(d1)
        # d1=LeakyReLU(alpha=0.2)(d1)
        # d1=Conv3D(filters=32,kernel_size=(3,3,1),strides=(2,2,2),padding='same')(d1)
        # d1=InstanceNormalization()(d1)
        # d1=LeakyReLU(alpha=0.2)(d1)

        # d2=Conv3D(filters=64,kernel_size=(3,3,3),strides=(2,2,2),padding='same')(d1)
        # d2=InstanceNormalization()(d2)
        # d2=LeakyReLU(alpha=0.2)(d2)
        # d2=Conv3D(filters=64,kernel_size=(3,3,3),strides=(1,1,1),padding='same')(d2)
        # d2=InstanceNormalization()(d2)
        # d2=LeakyReLU(alpha=0.2)(d2)

        # d3=Conv3D(filters=128,kernel_size=(3,3,3),strides=(2,2,2),padding='same')(d2)
        # d3=InstanceNormalization()(d3)
        # d3=LeakyReLU(alpha=0.2)(d3)
        # d3=Conv3D(filters=128,kernel_size=(3,3,3),strides=(1,1,1),padding='same')(d3)
        # d3=InstanceNormalization()(d3)
        # d3=LeakyReLU(alpha=0.2)(d3)

        # d4=Conv3D(filters=256,kernel_size=(3,3,3),strides=(2,2,2),padding='same')(d3)
        # d4=InstanceNormalization()(d4)
        # d4=LeakyReLU(alpha=0.2)(d4)
        # d4=Conv3D(filters=256,kernel_size=(3,3,3),strides=(1,1,1),padding='same')(d4)
        # d4=InstanceNormalization()(d4)
        # d4=LeakyReLU(alpha=0.2)(d4)

        # d5=Conv3D(filters=512,kernel_size=(3,3,3),strides=(2,2,1),padding='same')(d4)
        # d5=InstanceNormalization()(d5)
        # d5=LeakyReLU(alpha=0.2)(d5)
        # d5=Conv3D(filters=512,kernel_size=(3,3,3),strides=(1,1,1),padding='same')(d5)
        # d5=InstanceNormalization()(d5)
        # d5=LeakyReLU(alpha=0.2)(d5)

        # d6=Conv3D(filters=512,kernel_size=(3,3,3),strides=(2,2,1),padding='same')(d5)
        # d6=InstanceNormalization()(d6)
        # d6=LeakyReLU(alpha=0.2)(d6)
        # d6=Conv3D(filters=512,kernel_size=(3,3,3),strides=(1,1,1),padding='same')(d6)
        # d6=InstanceNormalization()(d6)
        # d6=LeakyReLU(alpha=0.2)(d6)

        # d7=Conv3D(filters=512,kernel_size=(3,3,3),strides=(2,2,1),padding='same')(d6)
        # d7=InstanceNormalization()(d7)
        # d7=LeakyReLU(alpha=0.2)(d7)
        # d7=Conv3D(filters=512,kernel_size=(3,3,3),strides=(1,1,1),padding='same')(d7)
        # d7=InstanceNormalization()(d7)
        # d7=LeakyReLU(alpha=0.2)(d7)

        # u1=UpSampling3D(size=(2,2,1))(d7)
        # u1=Concatenate()([u1, d6])
        # u1=Conv3D(filters=512,kernel_size=(3,3,3),strides=(1,1,1),padding='same')(u1)
        # u1=InstanceNormalization()(u1)
        # u1=LeakyReLU(alpha=0.2)(u1)
        # u1=Conv3D(filters=512,kernel_size=(3,3,3),strides=(1,1,1),padding='same')(u1)
        # u1=InstanceNormalization()(u1)
        # u1=LeakyReLU(alpha=0.2)(u1)

        # u2=UpSampling3D(size=(2,2,1))(u1)
        # u2=Concatenate()([u2, d5])
        # u2=Conv3D(filters=512,kernel_size=(3,3,3),strides=(1,1,1),padding='same')(u2)
        # u2=InstanceNormalization()(u2)
        # u2=LeakyReLU(alpha=0.2)(u2)
        # u2=Conv3D(filters=512,kernel_size=(3,3,3),strides=(1,1,1),padding='same')(u2)
        # u2=InstanceNormalization()(u2)
        # u2=LeakyReLU(alpha=0.2)(u2)

        # u3=UpSampling3D(size=(2,2,1))(u2)
        # u3=Concatenate()([u3, d4])
        # u3=Conv3D(filters=512,kernel_size=(3,3,3),strides=(1,1,1),padding='same')(u3)
        # u3=InstanceNormalization()(u3)
        # u3=LeakyReLU(alpha=0.2)(u3)
        # u3=Conv3D(filters=512,kernel_size=(3,3,3),strides=(1,1,1),padding='same')(u3)
        # u3=InstanceNormalization()(u3)
        # u3=LeakyReLU(alpha=0.2)(u3)

        # u4=UpSampling3D(size=(2,2,2))(u3)
        # u4=Concatenate()([u4, d3])
        # u4=Conv3D(filters=512,kernel_size=(3,3,3),strides=(1,1,1),padding='same')(u4)
        # u4=InstanceNormalization()(u4)
        # u4=LeakyReLU(alpha=0.2)(u4)
        # u4=Conv3D(filters=512,kernel_size=(3,3,3),strides=(1,1,1),padding='same')(u4)
        # u4=InstanceNormalization()(u4)
        # u4=LeakyReLU(alpha=0.2)(u4)

        # u5=UpSampling3D(size=(2,2,2))(u4)
        # u5=Concatenate()([u5, d2])
        # u5=Conv3D(filters=256,kernel_size=(3,3,3),strides=(1,1,1),padding='same')(u5)
        # u5=InstanceNormalization()(u5)
        # u5=LeakyReLU(alpha=0.2)(u5)
        # u5=Conv3D(filters=256,kernel_size=(3,3,3),strides=(1,1,1),padding='same')(u5)
        # u5=InstanceNormalization()(u5)
        # u5=LeakyReLU(alpha=0.2)(u5)

        # u6=UpSampling3D(size=(2,2,2))(u5)
        # u6=Concatenate()([u6, d1])
        # u6=Conv3D(filters=128,kernel_size=(3,3,3),strides=(1,1,1),padding='same')(u6)
        # u6=InstanceNormalization()(u6)
        # u6=LeakyReLU(alpha=0.2)(u6)
        # u6=Conv3D(filters=128,kernel_size=(3,3,3),strides=(1,1,1),padding='same')(u6)
        # u6=InstanceNormalization()(u6)
        # u6=LeakyReLU(alpha=0.2)(u6)

        # u7=UpSampling3D(size=(2,2,2))(u6)
        # u7=Concatenate()([u7, d00])
        # u7=Conv3D(filters=64,kernel_size=(3,3,3),strides=(1,1,1),padding='same')(u7)
        # u7=InstanceNormalization()(u7)
        # u7=LeakyReLU(alpha=0.2)(u7)
        # u7=Conv3D(filters=64,kernel_size=(3,3,3),strides=(1,1,1),padding='same')(u7)
        # u7=InstanceNormalization()(u7)
        # u7=LeakyReLU(alpha=0.2)(u7)

        # output_img = Conv3D(3, kernel_size=(1,1,1), strides=(1,1,1), padding='same',activation='tanh')(u7)

        # return Model(d00, output_img)

    def build_discriminator(self):
        def d_layer(layer_input, filters, kernel_size=(4,4,2), strides=(2,2,2), bn=True, dropout=False):
            """Discriminator layer"""
            init = RandomNormal(stddev=0.02)
            d = Conv3D(filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=init)(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
          
            if bn:
                #d = BatchNormalization(momentum=0.8)(d)
                d = InstanceNormalization()(d)
            if dropout:
                d = Dropout(0.3)(d)
            return d
        
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])
        
        ## testing
        #combined_imgs = GaussianNoise(0.02)(combined_imgs)
        
        if self.coordconv:
            combined_imgs = CoordinateChannel3D()(combined_imgs)
            
        n_d_layer = 4
        d_layers = []
        df = self.df
        dout = self.depth
        for i in range(n_d_layer):
            if i==0:
                d_layers.append(d_layer(combined_imgs, df, kernel_size=(4,4,2), strides=(2,2,2)))
            else:
                z = 2
                s = 2
                if dout == 2:
                    z = 1
                if i == n_d_layer-1:
                    s = 1
                df = min(df*2, self.df*8)
                if i==1 or i==2:
                   d_layers.append(d_layer(d_layers[-1], df, kernel_size=(4,4,z), strides=(s,s,z), dropout=True))
                else:
                   d_layers.append(d_layer(d_layers[-1], df, kernel_size=(4,4,z), strides=(s,s,z)))                  
            dout = max(2, int(0.5*dout))

        init = RandomNormal(stddev=0.02)
        features = Conv3D(1, kernel_size=(4,4,1), strides=1, padding='same', kernel_initializer=init)(d_layers[-1])
        validity = Activation('sigmoid')(features)
                
        return Model([img_A, img_B], validity), Model([img_A, img_B], features)

    def convert_resoutput(self, imgs_A, imgs_B):
        ## imgs 5D tensor (batch, rows, cols, depth, channels)
        if self.resoutput:
            new_A = []
            for i in range(imgs_B.shape[0]):
                a = rWND(255*(0.5*imgs_A[i]+0.5),self.data_loader.window2)
                b = rWND(255*(0.5*imgs_B[i]+0.5),self.data_loader.window1)
                c = a-b
                #c[b>100] = 0
                #c = WND(c,self.reswindow)
                M = np.max(c)
                m = np.min(c)
                if M==m:
                    c = np.zeros(c.shape)
                else:
                    c = 2.*(c-m)/(M-m) - 1
                new_A.append(c)
            imgs_A = np.array(new_A)
        return imgs_A
    
    def invert_resoutput(self, fake_A, imgs_B):
        if self.resoutput:
            '''
            new_A = []
            for i in range(fake_A.shape[0]):
                a = rWND(255*(0.5*fake_A[i]+0.5),self.reswindow)
                b = rWND(255*(0.5*imgs_B[i]+0.5),self.data_loader.window1)
                c = a+b
                c = WND(c,self.data_loader.window2)
                new_A.append(c/127.5 - 1.)
            fake_A = np.array(new_A)
            '''
            pass
        return fake_A
    
    def generate_noise(self, mode, imgs_A, imgs_B):
        if mode==0:
            pass
        elif mode==1:
            r = np.random.uniform(0,1)
            if r>=0.5:
                row,col,dep,ch= imgs_A.shape[1:]
                mean = 0
                sigma = 0.05
                gauss = np.random.normal(mean,sigma,(row,col,dep,ch))
                gauss = gauss.reshape(row,col,dep,ch)
                gauss = np.expand_dims(gauss, axis=0)
                imgs_A = imgs_A + gauss
                imgs_B = imgs_B + gauss
            else:
                pass
        elif mode==2:
            r = np.random.uniform(0,1)
            r2 = np.random.uniform(0,1)
            if r>=0.5:
                row,col,dep,ch= imgs_A.shape[1:]
                mean = 0
                sigma = 0.05
                gauss = np.random.normal(mean,sigma,(row,col,dep,ch))
                gauss = gauss.reshape(row,col,dep,ch)
                gauss = np.expand_dims(gauss, axis=0)
                imgs_A = imgs_A + gauss
                imgs_B = imgs_B + gauss
            else:
                pass
            if r2>=0.75:
                imgs_A = gaussian_filter(imgs_A, sigma=0.05)
                imgs_B = gaussian_filter(imgs_B, sigma=0.05)
            else:
                pass
        else:
            pass
            
        return imgs_A, imgs_B
    
    def train(self, epochs, batch_size=1, sample_interval=200, model_interval=-1):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        patch_shape = (batch_size,) + self.disc_patch
        if self.smoothlabel:
            valid = np.ones(patch_shape) - 0.1*np.random.rand(*patch_shape)
        else:
            valid = np.ones(patch_shape)
            
        fake = np.zeros(patch_shape)
        
        # log file
        f = open(os.path.join(self.savepath, 'log.txt'), 'w')
        f.close()
        
        for epoch in range(epochs):
            if epoch<10:
                lr = self.lr_ini
            else:
                lr = self.lr_ini/(1+(epoch-10)*self.lr_decay)
            K.set_value(self.discriminator.optimizer.lr, lr)
            K.set_value(self.combined.optimizer.lr, lr)
            
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size=batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------                
                imgs_A = self.convert_resoutput(imgs_A, imgs_B) #何も変わらない
                
                # randomshift
                non = lambda s: s if s<0 else None
                mom = lambda s: max(0,s)

                shift_A = np.full(imgs_A.shape, -1.)
                shift_B = np.full(imgs_A.shape, -1.)
                sx = int(self.randomshift*shift_A.shape[2])
                sy = int(self.randomshift*shift_A.shape[1])
                for i in range(shift_A.shape[0]):
                    ox = np.random.randint(2*sx+1, size=1)[0] - sx
                    oy = np.random.randint(2*sy+1, size=1)[0] - sy
                    shift_A[i,mom(oy):non(oy),mom(ox):non(ox),:,:] = imgs_A[i,mom(-oy):non(-oy),mom(-ox):non(-ox),:,:]
                    shift_B[i,mom(oy):non(oy),mom(ox):non(ox),:,:] = imgs_B[i,mom(-oy):non(-oy),mom(-ox):non(-ox),:,:]
                imgs_A = shift_A
                imgs_B = shift_B
                
                imgs_A, imgs_B = self.generate_noise(self.gennoise, imgs_A, imgs_B)
                
                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # 一時的な変数を定義
                temp_imgs_A = imgs_A
                temp_fake_A = fake_A

                coin = random.random()

                # realとfakeを入れ替える
                if coin < 0.35:
                    temp_imgs_A, temp_fake_A = fake_A, imgs_A

                d_loss_real = self.discriminator.train_on_batch([temp_imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([temp_fake_A, imgs_B], fake)
                d_loss = np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                '''
                if self.fmloss:
                    valid_feat = self.discriminator_feat.predict([imgs_A, imgs_B])
                    g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A, valid_feat])
                else:
                    g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
                '''
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                
                # Plot the progress
                newlog = "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, D fake: %f] [G loss: %f, gan loss: %f, L1 loss: %f] time: %s" % (
                    epoch+1, epochs, batch_i+1, self.data_loader.n_batches,
                    d_loss[0], d_loss_fake[0], #100*d_loss[1],
                    g_loss[0], g_loss[1], g_loss[2],
                    elapsed_time
                )
                
                print(newlog)
                f = open(os.path.join(self.savepath, 'log.txt'), 'a')
                f.write(newlog+'\n')
                f.close()    
                
                # If at save interval => save generated image samples
                if (batch_i+1) % sample_interval == 0:#設定したsample_interval毎に画像を保存
                    self.sample_images(epoch, batch_i+1)
            # save weights
            if model_interval>0:
                if (epoch+1) % model_interval == 0:#設定したmodel_interval毎に重みを保存
                    self.save_weights('{}'.format(str(epoch+1)))
                    
        # final sample image & save weights
        self.sample_images(epochs, 'final')
        if model_interval>0:
            self.save_weights('final{}'.format(str(epochs)))
        
    def predict_on_batch(self, imgs_A, imgs_B):
        fake_A = self.generator.predict(imgs_B)
        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])
        return gen_imgs

    def sample_images(self, epoch, batch_i):
        r, c = 3, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3)
        fake_A = np.concatenate([self.generator.predict(np.expand_dims(x,axis=0)) for x in imgs_B], axis=0)
        fake_A = self.invert_resoutput(fake_A, imgs_B)

        gen_imgs = np.concatenate([imgs_B[:,:,:,0,-1], fake_A[:,:,:,0,-1], imgs_A[:,:,:,0,-1]])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        plt.style.use('default')
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            dr = imgs_B.shape[1]//4+imgs_B.shape[1]//8
            dc = imgs_B.shape[2]//4+imgs_B.shape[2]//8
            pr = np.random.choice(imgs_B.shape[1]//2-dr)+imgs_B.shape[1]//4
            pc = np.random.choice(imgs_B.shape[2]//2-dc)+imgs_B.shape[2]//4
            
            for j in range(c):
                axs[j,i].imshow(gen_imgs[cnt], cmap='gray', vmin=0, vmax=1)
                axs[j,i].set_title(titles[i])
                axs[i,j].axis([pc,pc+dc,pr+dr,pr])
                cnt += 1
        samplepath = self.make_directory('samples')
        fig.savefig(os.path.join(samplepath, '{}_{}.png'.format(epoch, batch_i)))
        plt.close()
    
    def make_directory(self, dirname):
        dirpath = os.path.join(self.savepath, dirname)
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)
        return dirpath
    
    def load_weights(self, weightfile, summary=True):
        loadweightspath = os.path.join(self.savepath, 'models', '{}.h5'.format(weightfile))
        self.combined.load_weights(loadweightspath)
        if summary:
            self.combined.summary()
            
    def load_final_weights(self, *args, **kwargs):
        wdir = os.path.join(self.savepath, 'models')
        wlist = [os.path.splitext(x)[0] for x in os.listdir(wdir) if x.lower().endswith('.h5')]
        wlist.sort()

        s = None
        for x in wlist:
            if 'final' in x:
                s = x
        if s is None:
            s = wlist[-1]        
        
        self.load_weights(s,*args, **kwargs)
        
    def save_weights(self, weightfile):      
        modelpath = self.make_directory('models')
        self.combined.save_weights(os.path.join(modelpath, '{}.h5'.format(weightfile)))
        print('saved {}.h5 to {}'.format(weightfile, modelpath))
        
## not used
def ssim_mae_loss(y_true, y_pred):
    return 0.15*losses.mean_absolute_error(y_true, y_pred) + 0.85*ssim_loss(y_true, y_pred)

def ssim_loss(y_true, y_pred):
    # ssim for 3d data 구현
    ashape = y_true.get_shape().as_list()

    ut = tf.transpose(y_true[:,:,:,:,0], perm=[0,3,1,2])
    #ut = tf.reshape(ut, shape=[-1]+ashape[1:3])
    ut = tf.expand_dims(ut, axis=-1)
    up = tf.transpose(y_pred[:,:,:,:,0], perm=[0,3,1,2])
    #up = tf.reshape(up, shape=[-1]+ashape[1:3])
    up = tf.expand_dims(up, axis=-1)
    
    return tf.reduce_mean(1-tf.image.ssim_multiscale(ut, up, 2.0))


# modified from "An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution" from Uber research
# https://arxiv.org/abs/1807.03247

class _CoordinateChannel(Layer):
    def __init__(self, rank, 
                 data_format=None,
                 **kwargs):
        super(_CoordinateChannel, self).__init__(**kwargs)
    
        self.rank = rank
        self.data_format = K.image_data_format() if data_format is None else data_format
        self.axis = 1 if K.image_data_format() == 'channels_first' else -1

        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[self.axis]

        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={self.axis: input_dim})
        self.built = True
    
    def call(self, inputs):
        input_shape = K.shape(inputs)

        if self.rank == 2:

            input_shape = [input_shape[i] for i in range(4)]
            batch_shape, dim1, dim2, channels = input_shape

            xx_ones = K.ones(K.stack([batch_shape, dim2]), dtype='int32')
            xx_ones = K.expand_dims(xx_ones, axis=-1)

            xx_range = tf.tile(K.expand_dims(K.arange(0, dim1), axis=0),
                              K.stack([batch_shape, 1]))
            xx_range = K.expand_dims(xx_range, axis=1)
            xx_channels = K.batch_dot(xx_ones, xx_range, axes=[2, 1])
            xx_channels = K.expand_dims(xx_channels, axis=-1)
            xx_channels = K.permute_dimensions(xx_channels, [0, 2, 1, 3])

            yy_ones = K.ones(K.stack([batch_shape, dim1]), dtype='int32')
            yy_ones = K.expand_dims(yy_ones, axis=1)

            yy_range = tf.tile(K.expand_dims(K.arange(0, dim2), axis=0),
                              K.stack([batch_shape, 1]))
            yy_range = K.expand_dims(yy_range, axis=-1)

            yy_channels = K.batch_dot(yy_range, yy_ones, axes=[2, 1])
            yy_channels = K.expand_dims(yy_channels, axis=-1)
            yy_channels = K.permute_dimensions(yy_channels, [0, 2, 1, 3])

            xx_channels = K.cast(xx_channels, K.floatx())
            xx_channels = xx_channels / K.cast(dim1 - 1, K.floatx())
            xx_channels = (xx_channels * 2) - 1.

            yy_channels = K.cast(yy_channels, K.floatx())
            yy_channels = yy_channels / K.cast(dim2 - 1, K.floatx())
            yy_channels = (yy_channels * 2) - 1.

            outputs = K.concatenate([inputs, xx_channels, yy_channels], axis=-1)

        if self.rank == 3:

            input_shape = [input_shape[i] for i in range(5)]
            batch_shape, dim1, dim2, dim3, channels = input_shape

            xx_ones = K.ones(K.stack([batch_shape, dim3]), dtype='int32')
            xx_ones = K.expand_dims(xx_ones, axis=-1)

            xx_range = tf.tile(K.expand_dims(K.arange(0, dim2), axis=0),
                              K.stack([batch_shape, 1]))
            xx_range = K.expand_dims(xx_range, axis=1)

            xx_channels = K.batch_dot(xx_ones, xx_range, axes=[2, 1])
            xx_channels = K.expand_dims(xx_channels, axis=-1)
            xx_channels = K.permute_dimensions(xx_channels, [0, 2, 1, 3])

            xx_channels = K.expand_dims(xx_channels, axis=1)
            xx_channels = tf.tile(xx_channels,
                                 [1, dim1, 1, 1, 1])

            yy_ones = K.ones(K.stack([batch_shape, dim2]), dtype='int32')
            yy_ones = K.expand_dims(yy_ones, axis=1)

            yy_range = tf.tile(K.expand_dims(K.arange(0, dim3), axis=0),
                              K.stack([batch_shape, 1]))
            yy_range = K.expand_dims(yy_range, axis=-1)

            yy_channels = K.batch_dot(yy_range, yy_ones, axes=[2, 1])
            yy_channels = K.expand_dims(yy_channels, axis=-1)
            yy_channels = K.permute_dimensions(yy_channels, [0, 2, 1, 3])

            yy_channels = K.expand_dims(yy_channels, axis=1)
            yy_channels = tf.tile(yy_channels,
                                 [1, dim1, 1, 1, 1])

            
            zz_range = tf.tile(K.expand_dims(K.arange(0, dim1), axis=0),
                              K.stack([batch_shape, 1]))
            zz_range = K.expand_dims(zz_range, axis=-1)
            zz_range = K.expand_dims(zz_range, axis=-1)

            zz_channels = tf.tile(zz_range,
                                 [1, 1, dim2, dim3])
            zz_channels = K.expand_dims(zz_channels, axis=-1)
            

            xx_channels = K.cast(xx_channels, K.floatx())
            xx_channels = xx_channels / K.cast(dim2 - 1, K.floatx())
            xx_channels = xx_channels * 2 - 1.

            yy_channels = K.cast(yy_channels, K.floatx())
            yy_channels = yy_channels / K.cast(dim3 - 1, K.floatx())
            yy_channels = yy_channels * 2 - 1.

            zz_channels = K.cast(zz_channels, K.floatx())
            zz_channels = zz_channels / K.cast(dim1 - 1, K.floatx())
            zz_channels = zz_channels * 2 - 1.

            outputs = K.concatenate([inputs, zz_channels, xx_channels, yy_channels],axis=-1)
            
        return outputs
    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[self.axis]

        channel_count = self.rank

        output_shape = list(input_shape)
        output_shape[self.axis] = input_shape[self.axis] + channel_count
        return tuple(output_shape)
    
    def get_config(self):
        config = {
            'rank': self.rank,
            'data_format': self.data_format
        }
        base_config = super(_CoordinateChannel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
class CoordinateChannel3D(_CoordinateChannel):
    def __init__(self,
                 data_format=None,
                 **kwargs):
        super(CoordinateChannel3D, self).__init__(
            rank=3,
            data_format=data_format,
            **kwargs
        )
        
    def get_config(self):
        config = super(CoordinateChannel3D, self).get_config()
        config.pop('rank')
        return config
