import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import sys

with np.load("notMNIST.npz") as data:
	Data, Target = data ["images"], data["labels"]
	np.random.seed(521)
	randIndx = np.arange(len(Data))
	np.random.shuffle(randIndx)
	Data = Data[randIndx]/255.
	Target = Target[randIndx]
	trainData, trainTarget = Data[:15000], Target[:15000]
	validData, validTarget = Data[15000:16000], Target[15000:16000]
	testData, testTarget = Data[16000:], Target[16000:]

sess = tf.InteractiveSession()

# 15000 x 784
trainDataReshaped = np.reshape(trainData, [trainData.shape[0], trainData.shape[1] * trainData.shape[2]])
validDataReshaped = np.reshape(validData, [validData.shape[0], validData.shape[1] * validData.shape[2]])
testDataReshaped = np.reshape(testData, [testData.shape[0], testData.shape[1] * testData.shape[2]])
# 10 x 784
W = tf.Variable(tf.truncated_normal([10, trainData.shape[1] * trainData.shape[2]], stddev=1.0, name='weights'))
b = tf.Variable(0.0, name='biases')
print(np.shape(trainData))
# None x 784
X = tf.placeholder(tf.float32, [None, trainData.shape[1] * trainData.shape[2]], name='input_x')
y = tf.placeholder(tf.float32, [None, 10], name='target_y')

# Graph definition
predY = tf.matmul(X, tf.transpose(W)) + b

# Error definition
#MSE = tf.reduce_mean(tf.reduce_sum((predY - y)**2, 1))
CE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predY))
# Find the index of largest probability along predY axis 1. Find the index of 1 along y axis 1. Check if equal.
CORRECT = tf.count_nonzero(tf.cast(tf.equal(tf.argmax(predY, axis=1), tf.argmax(y, axis=1)), tf.float32))
ACC = CORRECT/tf.shape(y, out_type=tf.int64)[0]

print(trainTarget[:5])
print(tf.one_hot(trainTarget[:5], 10).eval())
trainTargetOneHot = tf.one_hot(trainTarget, 10).eval()
testTargetOneHot = tf.one_hot(testTarget, 10).eval()
validTargetOneHot = tf.one_hot(validTarget, 10).eval()
learningRates = [0.005, 0.001, 0.0001] # best is 0.001
for learningRate in learningRates:
    B = 500
    lam = 0.01
    start = time.time()
    epochCounter = 0
    epoch = []
    trainLoss = []
    validLoss = []
    trainAccuracies = []
    validAccuracies = []
    WD = lam/2*(tf.norm(W)**2)

    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    train = optimizer.minimize(loss=tf.add(CE, WD))

    init = tf.global_variables_initializer()
    sess.run(init)
    for iteration in range(1, 5001):
        # Shuffle data
        minibatch = random.sample(list(zip(trainDataReshaped, trainTargetOneHot)), B)
        minibatchData, minibatchTarget = zip(*minibatch)
        _ = sess.run(train, feed_dict={X: minibatchData, y: minibatchTarget})
        epochCounter += B
        if epochCounter >= trainData.shape[0]:
            epochCounter -= trainData.shape[0]
            epoch.append(B*iteration/trainData.shape[0])
            error = sess.run(CE, feed_dict={X: trainDataReshaped, y: trainTargetOneHot})
            error += sess.run(WD, feed_dict={X: trainDataReshaped, y: trainTargetOneHot})
            trainLoss.append(error)
            validError = sess.run(CE, feed_dict={X: validDataReshaped, y: validTargetOneHot})
            validError += sess.run(WD, feed_dict={X: validDataReshaped, y: validTargetOneHot})
            validLoss.append(validError)
            trainAccuracies.append(sess.run(ACC, feed_dict={X: trainDataReshaped, y: trainTargetOneHot}))
            validAccuracies.append(sess.run(ACC, feed_dict={X: validDataReshaped, y: validTargetOneHot}))
            end = time.time()
    print("learning rate " + str(learningRate) + " batch size " + str(B) + " converged to loss " + str(trainLoss[-1]) + " after " + str(end-start))
    correct = sess.run(CORRECT, feed_dict={X: validDataReshaped, y: validTargetOneHot})
    accuracy = sess.run(ACC, feed_dict={X: validDataReshaped, y: validTargetOneHot})
    print("validation set accuracy " + str(accuracy) + " correct " + str(correct))
    correct = sess.run(CORRECT, feed_dict={X: testDataReshaped, y: testTargetOneHot})
    accuracy = sess.run(ACC, feed_dict={X: testDataReshaped, y: testTargetOneHot})
    print("test set accuracy " + str(accuracy) + " correct " + str(correct))
    correct = sess.run(CORRECT, feed_dict={X: trainDataReshaped, y: trainTargetOneHot})
    accuracy = sess.run(ACC, feed_dict={X: trainDataReshaped, y: trainTargetOneHot})
    print("training set accuracy " + str(accuracy) + " correct " + str(correct))

    fig, ax1 = plt.subplots()
    ax1.plot(epoch, trainAccuracies, 'b.', mew=0.0, label='training accuracy')
    ax1.set_ylabel("Accuracy %", color='b')
    ax1.set_xlabel('Epoch')
    vals = ax1.get_yticks()
    ax1.set_yticklabels(['{:d}%'.format(int(x*100)) for x in vals])
    ax2 = ax1.twinx()
    ax2.plot(epoch, trainLoss, 'r.', mew=0.0, label='training loss')
    ax2.set_ylabel("Loss", color='r')
    plt.show()
    fig, ax1 = plt.subplots()
    ax1.plot(epoch, validAccuracies, 'b.', mew=0.0, label='validation accuracy')
    ax1.set_ylabel("Accuracy %", color='b')
    ax1.set_xlabel('Epoch')
    vals = ax1.get_yticks()
    ax1.set_yticklabels(['{:d}%'.format(int(x*100)) for x in vals])
    ax2 = ax1.twinx()
    ax2.plot(epoch, validLoss, 'r.', mew=0.0, label='validation loss')
    ax2.set_ylabel("Loss", color='r')
    plt.show()

"""
learning rate 0.005 batch size 500 converged to loss 0.49875438 after 71.35069012641907
validation set accuracy 0.895 correct 895
test set accuracy 0.8942731277533039 correct 2436
training set accuracy 0.8969333333333334 correct 13454
learning rate 0.001 batch size 500 converged to loss 0.48137113 after 73.3856885433197
validation set accuracy 0.899 correct 899
test set accuracy 0.8909691629955947 correct 2427
training set accuracy 0.9004666666666666 correct 13507
learning rate 0.0001 batch size 500 converged to loss 19.264252 after 72.94785666465759
validation set accuracy 0.713 correct 713
test set accuracy 0.7430249632892805 correct 2024
training set accuracy 0.737 correct 11055
"""
