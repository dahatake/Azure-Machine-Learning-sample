# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import argparse
import os

import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import Callback
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.utils import multi_gpu_model

import tensorflow as tf

from azureml.core import Run
from utils import load_data, one_hot_encode

print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)

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
parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=100,
                    help='# of neurons in the first layer')
parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=100,
                    help='# of neurons in the second layer')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001, help='learning rate')
parser.add_argument('--activation', type=str, dest='activation', default='relu', help='activation function')
parser.add_argument('--optimizer', type=str, dest='optimizer', default='RMSprop', help='Optimzers to use for training. Defaults to RMSProp for initial training and SGD for subsequent.')
parser.add_argument('--loss', type=str, dest='loss', default='categorical_crossentropy', help='loss function.')
parser.add_argument('--dropout', type=float, dest='dropout', default=0.2, help='Drop Out rate')
parser.add_argument('--gpu', type=int, dest='gpu', default=1, help='The count of GPU')
parser.add_argument('--auto_mixed_precision', type=int, dest='auto_mixed_precision', default=1, help='Enable Automatic Mixed Precision to use Tensor Core')

args = parser.parse_args()

# Control Automatic Mixed Precision to use NVIDIA Tensor Core
# https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = str(args.auto_mixed_precision)

data_folder = args.data_folder

print('training dataset is stored here:', data_folder)

X_train = load_data(os.path.join(data_folder, 'train-images.gz'), False) / 255.0
X_test = load_data(os.path.join(data_folder, 'test-images.gz'), False) / 255.0

y_train = load_data(os.path.join(data_folder, 'train-labels.gz'), True).reshape(-1)
y_test = load_data(os.path.join(data_folder, 'test-labels.gz'), True).reshape(-1)

training_set_size = X_train.shape[0]

n_inputs = 28 * 28
n_h1 = args.n_hidden_1
n_h2 = args.n_hidden_2
n_outputs = 10
n_epochs = args.epoch
batch_size = args.batch_size
learning_rate = args.learning_rate
activation = args.activation
optimizer = args.optimizer
dropout = args.dropout
loss = args.loss
gpu = args.gpu

y_train = one_hot_encode(y_train, n_outputs)
y_test = one_hot_encode(y_test, n_outputs)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

# Build a simple MLP model
model = Sequential()
# first hidden layer
model.add(Dense(n_h1, activation=activation, input_shape=(n_inputs,)))
model.add(Dropout(dropout))
# second hidden layer
model.add(Dense(n_h2, activation=activation))
model.add(Dropout(dropout))
# output layer
model.add(Dense(n_outputs, activation='softmax'))

model.summary()

# optimize multi_gpu
if gpu > 1:
	model = multi_gpu_model(model, gpus=gpu)
	batch_size = batch_size * gpu

model.compile(loss=loss,
              optimizer=optimizer_types[optimizer](learning_rate),
              metrics=['accuracy'])

# start an Azure ML run
run = Run.get_context()

class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        run.log('Loss', log['val_loss'])
        run.log('Accuracy', log['val_accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    verbose=2,
                    validation_data=(X_test, y_test),
                    callbacks=[LogRunMetrics()])

score = model.evaluate(X_test, y_test, verbose=0)

# log a single value
run.log("Final test loss", score[0])
print('Test loss:', score[0])

run.log('Final test accuracy', score[1])
print('Test accuracy:', score[1])

plt.figure(figsize=(6, 3))
plt.title('MNIST with Keras MLP ({} epochs)'.format(n_epochs), fontsize=14)
plt.plot(history.history['val_accuracy'], 'b-', label='Accuracy', lw=4, alpha=0.5)
plt.plot(history.history['val_loss'], 'r--', label='Loss', lw=4, alpha=0.5)
plt.legend(fontsize=12)
plt.grid(True)

# log an image
run.log_image('Accuracy vs Loss', plot=plt)

# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs/model', exist_ok=True)

# serialize NN architecture to JSON
model_json = model.to_json()
# save model JSON
with open('./outputs/model/model.json', 'w') as f:
    f.write(model_json)
# save model weights
model.save_weights('./outputs/model/model.h5')
# save model itself
model.save('./outputs/model/mnist.h5')
print("model saved in ./outputs/model folder")