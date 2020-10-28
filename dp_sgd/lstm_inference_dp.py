from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.stats import truncnorm
from tensorflow.contrib import rnn

import numpy as np
import math, json
import dp_optimizer



import tensorflow as tf
sess = tf.Session()

with open('config.json') as config_file:
    config = json.load(config_file)

dataType = config['data_type']
TIMESTEPS = config['hist_len']
Hidden = config['Hidden']

NUM_LAYERS = config['NUM_LAYERS']
DROP_OUT = config['DROP_OUT'] # prob to keep


def init_weights(size):
    # we truncate the normal distribution at two times the standard deviation (which is 2)
    # to account for a smaller variance (but the same mean), we multiply the resulting matrix with he desired std
    return np.float32(truncnorm.rvs(-2, 2, size=size)*1.0/math.sqrt(float(size[0])))

def get_a_cell(lstm_size, keep_prob, bTrain):
  lstm = rnn.BasicLSTMCell(lstm_size)
#  if bTrain: lstm = rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob) # avoid dropout in test
  return lstm


def inference_lstm(bTrain, inputData, num_hidden):
  print('num_hidden', num_hidden, 'NUM_CLASSES', NUM_CLASSES)
  
  # Define weights
  weights = {
    'out': tf.Variable(tf.random_normal([Hidden, NUM_CLASSES]))
#    'out': tf.Variable(np.zeros([Hidden, NUM_CLASSES]), dtype=tf.float32)
  }
  biases = {
    'out': tf.Variable(tf.random_normal([NUM_CLASSES]))
#    'out': tf.Variable(np.zeros([NUM_CLASSES]), dtype=tf.float32)
  }

  print('for static rnn, shape of input', inputData.shape)
  inputData = tf.unstack(inputData, TIMESTEPS, 1)
  lstm_cell = rnn.MultiRNNCell([get_a_cell(Hidden, DROP_OUT, bTrain) for i in range(num_hidden)])
  outputs, states = rnn.static_rnn(lstm_cell, inputData, dtype=tf.float32)
  # Linear activation, using rnn inner loop last output
  logits = tf.matmul(outputs[-1], weights['out']) + biases['out']

  print('logits shape', tf.shape(logits))
  return logits

def prediction(logits):
  return tf.nn.softmax(logits)

def loss(logits, labels):
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')

  vector_loss = cross_entropy
  scalar_loss = tf.reduce_mean(vector_loss)
  return vector_loss, scalar_loss


def training(vector_loss, scalar_loss, learning_rate, dpsgd=False, l2_norm_clip=0, noise_multiplier=0, microbatches=0, num_examples=0):

  if dpsgd:
    optimizer = dp_optimizer.DPAdamGaussianOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=microbatches,
        learning_rate=learning_rate,
        unroll_microbatches=True,
        population_size=num_examples)
    opt_loss = vector_loss
  else:
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate)
    opt_loss = scalar_loss
#  global_step = tf.train.get_global_step()
  train_op = optimizer.minimize(loss=opt_loss) #, global_step=global_step)

  return train_op


def evaluation(logits, labels):
  print('in evaluation, logits.shape', logits.shape, 'labels.shape', labels.shape)
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))


def placeholder_inputs(oneHot):
    if oneHot:
        input_placeholder = tf.placeholder(tf.float32, shape=(None, TIMESTEPS, NUM_CLASSES), name='images_placeholder')
    else:
        input_placeholder = tf.placeholder(tf.float32, shape=(None, TIMESTEPS, 1), name='images_placeholder')
    labels_placeholder = tf.placeholder(tf.int32, (None), name='labels_placeholder')
    return input_placeholder, labels_placeholder


def lstm_model(oneHot, bTrain=True, num_classes=20, num_hidden=2, learning_rate=0.001, dpsgd=False, l2_norm_clip=0, noise_multiplier=0, microbatches=0, num_examples=0):
    global NUM_CLASSES
    NUM_CLASSES = num_classes
    # - placeholder for the input Data (in our case MNIST), depends on the batch size specified in C
    data_placeholder, labels_placeholder = placeholder_inputs(oneHot)

    # - logits : output of the fully connected neural network when fed with images. The NN's architecture is
    #           specified in '
    logits = inference_lstm(bTrain, data_placeholder, num_hidden)

    # - loss : when comparing logits to the true labels.
    vector_loss, scalar_loss = loss(logits, labels_placeholder)

    # - eval_correct: When run, returns the amount of labels that were predicted correctly.
    eval_correct = evaluation(logits, labels_placeholder)

    # - global_step :          A Variable, which tracks the amount of steps taken by the clients:
    global_step = tf.Variable(0, dtype=tf.float32, trainable=False, name='global_step')

    # - learning_rate : A tensorflow learning rate, dependent on the global_step variable.
#    learning_rate = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step, decay_steps=27000, decay_rate=0.1, staircase=False, name='learning_rate')

    train_op = training(vector_loss, scalar_loss, learning_rate, dpsgd, l2_norm_clip, noise_multiplier, microbatches, num_examples)

    pred = prediction(logits)

    return train_op, eval_correct, scalar_loss, data_placeholder, labels_placeholder, pred
