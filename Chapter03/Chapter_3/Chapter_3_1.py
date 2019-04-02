import sys
import numpy as np
from IPython.core.debugger import Tracer
from keras.datasets import mnist
from keras.datasets import cifar100
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt

from data_loader import DataLoader
plt.switch_backend('agg')

class GAN(object):
    def generator(self):
        model = Sequential()
        model.add(Dense(256, input_shape=(100,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.WIDTH * self.HEIGHT * self.CHANNELS, activation='tanh'))
        model.add(Reshape((self.WIDTH, self.HEIGHT, self.CHANNELS)))
        return model

    def discriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.SHAPE))
        model.add(Dense((self.WIDTH * self.HEIGHT * self.CHANNELS), input_shape=self.SHAPE))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(int((self.WIDTH * self.HEIGHT * self.CHANNELS)/2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        return model

    def stacked_G_D(self):
        self.D.trainable = False
        model = Sequential()
        model.add(self.G)
        model.add(self.D)
        return model

    def __init__(self, width = 28, height= 28, channels = 1):
        self.WIDTH = width
        self.HEIGHT = height
        self.CHANNELS = channels
        self.SHAPE = (self.WIDTH, self.HEIGHT, self.CHANNELS)
        self.OPTIMIZER = Adam(lr=0.0002, decay=8e-9)
        self.noise_gen = np.random.normal(0,1,(100,))
        self.G = self.generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)
        self.D = self.discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER, metrics=['accuracy'] )
        self.stacked_G_D = self.stacked_G_D()        
        self.stacked_G_D.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)

    def train(self, X_train, epochs=2000000, batch = 32, save_interval = 200):
        for cnt in range(epochs):
            ## train discriminator
            random_index =  np.random.randint(0, len(X_train) - batch/2)
            legit_images = X_train[random_index : random_index + int(batch/2)].reshape(int(batch/2), self.WIDTH, self.HEIGHT, self.CHANNELS)
            gen_noise = np.random.normal(0, 1, (int(batch/2),100))
            syntetic_images = self.G.predict(gen_noise)
            x_combined_batch = np.concatenate((legit_images, syntetic_images))
            y_combined_batch = np.concatenate((np.ones((int(batch/2), 1)), np.zeros((int(batch/2), 1))))
            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)
            # train generator
            noise = np.random.normal(0, 1, (batch,100))
            y_mislabled = np.ones((batch, 1))
            g_loss = self.stacked_G_D.train_on_batch(noise, y_mislabled)
            print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))
            
            if cnt % save_interval == 0 : 
                self.plot_images(save2file=True, step=cnt)


    def plot_images(self, save2file=False,  samples=16, step=0):
        filename = "./images/mnist_%d.png" % step
        noise = np.random.normal(0, 1, (samples,100))
        images = self.G.predict(noise)
        
        plt.figure(figsize=(10,10))
    
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [ self.HEIGHT, self.WIDTH, self.CHANNELS ])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.tight_layout()

        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

if __name__ == '__main__':
    (X_train, y), (_, _) = cifar100.load_data(label_mode='fine')
    Z_train = []
    cnt = 0
    for i in range(0,len(y)):
        if y[i] == 19:
            cnt = cnt + 1            
            z = X_train[i]
            Z_train.append(z)            

# Rescale -1 to 1
    #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    #X_train = np.expand_dims(X_train, axis=3)
    Z_train = np.reshape(Z_train, [500, 32, 32, 3])
    Z_train = (Z_train.astype(np.float32) - 127.5) / 127.5
    Z_train = np.expand_dims(Z_train, axis=3)


gan = GAN(width=32, height=32,channels=3 )
gan.train(Z_train)
