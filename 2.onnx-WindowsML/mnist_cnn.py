# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import print_function

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import Callback
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.utils import multi_gpu_model
from keras import backend as K

from azureml.core import Run
from utils import load_data, one_hot_encode

# parse parameters
optimizer_types = {
    'SGD': lambda lr: SGD(lr=lr),
    'RMSprop': lambda lr: RMSprop(lr=lr),
    'Adagrad': lambda lr: Adagrad(lr=lr),
    'Adadelta': lambda lr: Adadelta(lr=lr),
    'Adam': lambda lr: Adam(lr=lr),
    'Adamax': lambda lr: Adamax(lr=lr),
    'Nadam': lambda lr: Nadam(lr=lr)
}

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=50, help='mini batch size for training')
parser.add_argument('--epoch', type=int, dest='epoch', default=20, help='epoch size for training')
parser.add_argument('--neurons-1', type=int, dest='neurons_1', default=32, help='# of neurons in the first layer')
parser.add_argument('--neurons-2', type=int, dest='neurons_2', default=64,help='# of neurons in the second layer')
parser.add_argument('--neurons-3', type=int, dest='neurons_3', default=128, help='# of neurons in the third layer')
parser.add_argument('--kernel-size-1', type=int, dest='kernel_size_1', default=3, help='kernel size of first layer')
parser.add_argument('--kernel-size-2', type=int, dest='kernel_size_2', default=3, help='kernel size of second layer')
parser.add_argument('--pool-size', type=int, dest='pool_size', default=2, help='# of neurons in the third layer')                    
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001, help='learning rate')
parser.add_argument('--activation', type=str, dest='activation', default='relu', help='activation function')
parser.add_argument('--optimizer', type=str, dest='optimizer', default='RMSprop', help='Optimzers to use for training. Defaults to RMSProp for initial training and SGD for subsequent.')
parser.add_argument('--loss', type=str, dest='loss', default='categorical_crossentropy', help='loss function.')
parser.add_argument('--dropout-1', type=float, dest='dropout_1', default=0.25, help='Drop Out rate 1st')
parser.add_argument('--dropout-2', type=float, dest='dropout_2', default=0.5, help='Drop Out rate 2nd')
parser.add_argument('--gpu', type=int, dest='gpu', default=1, help='The count of GPU')
parser.add_argument('--auto_mixed_precision', type=int, dest='auto_mixed_precision', default=1, help='Enable Automatic Mixed Precision to use Tensor Core')

args = parser.parse_args()

data_folder = args.data_folder

img_rows, img_cols = 28, 28
n_classes = 10
neurons_1 = args.neurons_1
neurons_2 = args.neurons_2
neurons_3 = args.neurons_3
kernel_size_1 = args.kernel_size_1
kernel_size_2 = args.kernel_size_2
pool_size = args.pool_size
dropout_1 = args.dropout_1
dropout_2 = args.dropout_2

epochs = args.epoch
batch_size = args.batch_size
learning_rate = args.learning_rate
activation = args.activation
optimizer = args.optimizer
loss = args.loss
gpu = args.gpu

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = str(args.auto_mixed_precision)

# the data, split between train and test sets
x_train = load_data(os.path.join(data_folder, 'train-images.gz'), False) / 255.0
x_test = load_data(os.path.join(data_folder, 'test-images.gz'), False) / 255.0

y_train = load_data(os.path.join(data_folder, 'train-labels.gz'), True).reshape(-1)
y_test = load_data(os.path.join(data_folder, 'test-labels.gz'), True).reshape(-1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


K.set_image_data_format('channels_first')

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
print(input_shape, 'input_shape')
    
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

model = Sequential()
model.add(Conv2D(neurons_1, kernel_size=(kernel_size_1, kernel_size_1),
                activation=activation,
                input_shape=input_shape))
model.add(Conv2D(neurons_2, kernel_size=(kernel_size_2, kernel_size_2), 
                activation=activation))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(dropout_1))
model.add(Flatten())
model.add(Dense(neurons_3, 
                activation=activation))
model.add(Dropout(dropout_2))
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss=loss,
            optimizer=optimizer_types[optimizer](learning_rate),
            metrics=['accuracy'])

# start an Azure ML run
run = Run.get_context()

class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        run.log('Loss', log['loss'])
        run.log('Accuracy', log['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[LogRunMetrics()])

score = model.evaluate(x_test, y_test, verbose=0)

run.log("Final test loss", score[0])
run.log('Final test accuracy', score[1])

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# log an image
plt.figure(figsize=(6, 3))
plt.title('MNIST with Keras MLP ({} epochs)'.format(epochs), fontsize=14)
plt.plot(history.history['accuracy'], 'b-', label='Accuracy', lw=4, alpha=0.5)
plt.plot(history.history['loss'], 'r--', label='Loss', lw=4, alpha=0.5)
plt.legend(fontsize=12)
plt.grid(True)

run.log_image('Accuracy vs Loss', plot=plt)

# save model
os.makedirs('./outputs/model', exist_ok=True)
model.save('./outputs/model/mnist.h5', include_optimizer=False)

# serialize NN architecture to JSON
model_json = model.to_json()
# save model JSON
with open('./outputs/model/mnist.json', 'w') as f:
    f.write(model_json)
# save model weights
model.save_weights('./outputs/model/mnist_weights.h5')

print("model saved in ./outputs/model folder")