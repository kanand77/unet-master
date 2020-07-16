import numpy as np 
import os
from tensorflow.python.keras.utils.data_utils import Sequence
import skimage.io as io

class DataGenerator(Sequence):
    def __init__(self, im_IDs, train_path, truth_path, 
                 train_prefix = 'input', truth_prefix = 'truth',
                 N_t = 25, batch_size=4, shuffle=True, 
                 dim = (256,256), n_channels=2, do_fft = False):

        self.im_IDs = im_IDs
        self.N_t = N_t
        self.dim = dim
        self.n_channels = n_channels
        
        self.train_path = train_path
        self.truth_path = truth_path
        self.train_prefix = train_prefix
        self.truth_prefix = truth_prefix 

        self.do_fft = do_fft

        self.list_IDs = []
        for im_ID in (self.im_IDs):
            for j in range(self.N_t):
                self.list_IDs.append('%03d_t%02d' % (im_ID, j))

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

        print('Initialized with {} total IDs'.format(len(self.list_IDs)))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            for k in range(2):
                
                save_str = ''
                if k == 0:
                    save_str = 'r'
                elif k == 1:
                    save_str = 'i'

                # Load training data
                f = '%s_%s_%s' % (self.train_prefix, ID, save_str)
                img = io.imread(os.path.join(self.train_path, "%s.png" % f), as_gray = True).astype(np.float)
                img = (img / 255.0) - 0.5
                
                # Maybe normalize these images (img)
                X[i,:,:,k] = img

                # Loading the truth data
                f = '%s_%s_%s' % (self.truth_prefix, ID, save_str)
                img = io.imread(os.path.join(self.truth_path, "%s.png" % f), as_gray = True).astype(np.float)
                img = (img / 255.0) - 0.5

                # Maybe normalize these images (img)
                y[i,:,:,k] = img

        if self.do_fft:
            for i in range(self.batch_size):
                img = X[i,:,:,0] + 1j * X[i,:,:,1]
                img = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img)))
                X[i,:,:,0] = img.real
                X[i,:,:,1] = img.imag

                img = y[i,:,:,0] + 1j * y[i,:,:,1]
                img = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img)))
                y[i,:,:,0] = img.real
                y[i,:,:,1] = img.imag

        return X, y