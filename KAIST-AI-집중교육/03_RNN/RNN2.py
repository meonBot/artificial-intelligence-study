from __future__ import absolute_import, division, print_function

import numpy as np
import os
import shutil
import tensorflow as tf
import models
import matplotlib.pylab as plt

### load data and split it to training and test data###
data = np.loadtxt('stock_data.txt')
data_train_X = data[:450,:]
data_train_Y = data[1:451,:]
data_test_X  = data[450:-1,:]
data_test_Y  = data[451:,:]

### Hyperparameters ### 
INPUT_SIZE    = 9
OUTPUT_SIZE   = 9
HIDDEN_SIZE   = 100
LEARNING_RATE = 0.005
nEpoch 	      = 500

### Grapch ### 
# placehoders for inputs and outputs
inputs   = tf.placeholder(tf.float32, (None, INPUT_SIZE))  
outputs  = tf.placeholder(tf.float32, (None, OUTPUT_SIZE)) 
inputs_  = tf.expand_dims(inputs,0)
outputs_ = tf.expand_dims(outputs,0)

# RNN/LSTM layer
#cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
#cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
#cell = models.BasicRNNCell(HIDDEN_SIZE)
#cell = models.BasicLSTMCell(HIDDEN_SIZE)
#cell = models.GRUCell(HIDDEN_SIZE)
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs_, dtype=tf.float32)

# Output layer 
pred = tf.layers.dense(rnn_outputs, OUTPUT_SIZE)

# objective function 
mse = tf.reduce_mean(tf.square(pred - outputs_))

tf.summary.scalar("loss", mse)
merge_op = tf.summary.merge_all()

#optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
minimize = optimizer.minimize(mse)

# graph information
#size = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
#n = sum(size(v) for v in tf.trainable_variables())
#print("Model size: %d" % n)

### run ### 
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if os.path.exists('train'):
  shutil.rmtree('train')
if os.path.exists('val'):
  shutil.rmtree('val')
os.mkdir('train')
os.mkdir('val')
train_writer = tf.summary.FileWriter('train', sess.graph)
val_writer = tf.summary.FileWriter('val', sess.graph)

for i in range(nEpoch):
  _, train_mse, summary = sess.run([minimize, mse, merge_op], {inputs: data_train_X, outputs: data_train_Y})
  train_writer.add_summary(summary, i)
  test_mse, summary = sess.run([mse, merge_op], {inputs: data_test_X, outputs: data_test_Y})
  val_writer.add_summary(summary, i)
  print("Epoch: %d Train MSE: %.5f Test MSE: %.5f " % (i+1, train_mse, test_mse))

pred_train = sess.run(pred, {inputs:data_train_X, outputs:data_train_Y})
pred_test = sess.run(pred, {inputs:data_test_X, outputs:data_test_Y})
