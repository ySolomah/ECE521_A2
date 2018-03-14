import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import sys

with np.load("notMNIST.npz") as data :
    Data, Target = data ["images"], data["labels"]
    posClass = 2
    negClass = 9
    dataIndx = (Target==posClass) + (Target==negClass)
    Data = Data[dataIndx]/255.
    Target = Target[dataIndx].reshape(-1, 1)
    Target[Target==posClass] = 1
    Target[Target==negClass] = 0
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data, Target = Data[randIndx], Target[randIndx]
    trainData, trainTarget = Data[:3500], Target[:3500]
    validData, validTarget = Data[3500:3600], Target[3500:3600]
    testData, testTarget = Data[3600:], Target[3600:]

sess = tf.InteractiveSession()

# 3500 x 784
trainDataReshaped = np.reshape(trainData, [trainData.shape[0], trainData.shape[1] * trainData.shape[2]])
validDataReshaped = np.reshape(validData, [validData.shape[0], validData.shape[1] * validData.shape[2]])
testDataReshaped = np.reshape(testData, [testData.shape[0], testData.shape[1] * testData.shape[2]])
# Add x0 = 1, so 785 columns
trainDataReshaped = np.insert(trainDataReshaped, 0, 1, axis=1)
validDataReshaped = np.insert(validDataReshaped, 0, 1, axis=1)
testDataReshaped = np.insert(testDataReshaped, 0, 1, axis=1)
# None x 785
X = tf.placeholder(tf.float32, [None, trainData.shape[1] * trainData.shape[2] + 1], name='input_x')
y = tf.placeholder(tf.float32, [None, 1], name='target_y')

# 1 x 785
# Normal equation (X^T*X)^-1*X^T*y
W = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(X), X)), tf.transpose(X)), y)

# Graph definition
predY = tf.matmul(X, W)

# Error definition
MSE = tf.reduce_mean(tf.reduce_sum((predY - y)**2, 1))
# Classification threshold is 0.5. Cast bool Tensor output by tf.greater to get a
# tensor of 1.'s and 0.'s, i.e. the classifications yhat, then subtract y and
# count nonzeros to get # of failures.
ACC = 1-tf.count_nonzero(tf.cast(tf.greater(predY, 0.5), tf.float32) - y)/tf.shape(y, out_type=tf.int64)[0]

init = tf.global_variables_initializer()
sess.run(init)

start = time.time()
sess.run(W, feed_dict={X: trainDataReshaped, y: trainTarget})
end = time.time()

error = sess.run(MSE, feed_dict={X: trainDataReshaped, y: trainTarget})
accuracy = sess.run(ACC, feed_dict={X: validDataReshaped, y: validTarget}) # 1.3
print("took " + str(end - start))
print("final training MSE" + str(error))
print("validation set accuracy " + str(accuracy))
accuracy = sess.run(ACC, feed_dict={X: testDataReshaped, y: testTarget})
print("test set accuracy " + str(accuracy))
accuracy = sess.run(ACC, feed_dict={X: trainDataReshaped, y: trainTarget})
print("training set accuracy " + str(accuracy))

"""
took 0.15718817710876465
final training MSE0.018783608
validation set accuracy 0.48
test set accuracy 0.496551724137931
training set accuracy 0.9934285714285714
"""
