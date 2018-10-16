import keras
import os
import numpy as np
from keras.models import load_model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from config.config import BATCH_SIZE, NUM_CLASSES, EPOCHS, IMG_ROWS, IMG_COLS
import logging

# Setting logging file path
try:
    log_path = os.stat(os.path.join(os.getcwd(), 'log'))
except:
    log_path = os.mkdir(os.path.join(os.getcwd(), 'log'))
    

# Logging object configrations
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='logger.log',
                    filemode='w')


class ImageClassifier:
    """
    Trains a simple convnet on the MNIST dat,'mni')aset.
    """
    def __init__(self):
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, IMG_ROWS, IMG_COLS)
            x_test = x_test.reshape(x_test.shape[0], 1, IMG_ROWS, IMG_COLS)
            input_shape = (1, IMG_ROWS, IMG_COLS)
        else:
            x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
            x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)
            input_shape = (IMG_ROWS, IMG_COLS, 1)
    
        
        # create a sequential convnet model in Keras
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASSES, activation='softmax'))

        # define loss, optimizer and evaluation metric
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        self.best_model = None
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        """
        Train the model
        :return:
        """
        # convert types
        x_train = self.x_train.astype('float32')
        x_test = self.x_test.astype('float32')

        # scale
        x_train /= 255
        x_test /= 255

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(self.y_train, NUM_CLASSES)
        y_test = keras.utils.to_categorical(self.y_test, NUM_CLASSES)
        
        # setting up path
        try:
            os.stat(os.path.join(os.getcwd()  ,'SAVED_MODELS'))
        except:
            os.makedirs(os.path.join(os.getcwd()  ,'SAVED_MODELS'))
        
        # define the callback function
        self.best_model = ModelCheckpoint(os.path.join(os.getcwd()  ,'SAVED_MODELS', 'mnist_model.h5'), save_best_only=True)
        
        # finally fit the model on the data

        self.model.fit(x_train, y_train,
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS,
                       verbose=1,
                       validation_data=(x_test, y_test), callbacks= [self.best_model])
        
        # saving the model
        
        logging.info("Saving the model: {}". format(os.path.join(os.getcwd()  ,'SAVED_MODELS')))

        
        return True
    
    # Function to train data coming from user
    
    def batch_train_online(self, train, label):
        """
        Trains the model from the point it was left at previously
        """
              
        self.model.fit(train, label,
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS,
                       verbose=1,
                       validation_split= 0.1, callbacks= [self.best_model])
            
        logging.info('Model Updated with new training data...Training completed. Saving the model now')

        #self.model.save(os.path.join(os.getcwd()  ,'SAVED_MODELS', 'mnist_model.h5'))
        
        logging.info("Model Saved Successfully at: {}".format(os.path.join(os.getcwd()  ,'SAVED_MODELS', 'mnist_model.h5')))
    
        
    
    ### Finding available memory
    
    def get_model_memory_usage(self):
        '''
	Approximate the DNN model's memory usage.
	:return: total_mem: total memeory usage
	'''
        total_mem = 4*self.model.count_params()
        
        return total_mem
    
    
    def load_model(self):
        """
        Loads pretrained model
        """
        self.model = load_model(os.path.join(os.getcwd()  ,'SAVED_MODELS', 'mnist_model.h5'))
        

    def predict(self, newdata):
        """
        Predict class from the features
        :param newdata:
        :return: predicted output
        """
        return self.model.predict(newdata)
