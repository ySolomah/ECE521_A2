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
learningRates = [0.002]
for learningRate in learningRates:
    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    train = optimizer.minimize(loss=CE)

    init = tf.global_variables_initializer()
    sess.run(init)
    initialW = sess.run(W)  
    initialb = sess.run(b)

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
    ax1.plot(epoch, trainLoss, 'b.', mew=0.0, label='training loss')
    ax1.plot(epoch, validLoss, 'm.', mew=0.0, label='validation loss')
    ax1.legend(loc="best")
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel("epoch")
    ax2 = ax1.twinx() 
    ax2.plot(epoch, trainAccuracies, 'g.', mew=0.0, label='training accuracy')
    ax2.plot(epoch, validAccuracies, 'r.', mew=0.0, label='validation accuracy')
    ax2.legend(loc="best")
    ax2.tick_params('y', colors='r')
    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{:d}%'.format(int(x*100)) for x in vals])
    ax2.set_ylabel("accuracy")
    fig.tight_layout()
    plt.show()

"""
learning rate 0.005 batch size 500 converged to loss 21.130325 after 64.43282628059387
validation set accuracy 0.877 correct 877
test set accuracy 0.8638032305433186 correct 2353
training set accuracy 0.9368 correct 14052
learning rate 0.001 batch size 500 converged to loss 23.909683 after 67.57670474052429
validation set accuracy 0.866 correct 866
test set accuracy 0.8582966226138032 correct 2338
training set accuracy 0.8873333333333333 correct 13310
learning rate 0.0001 batch size 500 converged to loss 31.071321 after 66.96525382995605
validation set accuracy 0.778 correct 778
test set accuracy 0.7749632892804699 correct 2111
training set accuracy 0.7778666666666667 correct 11668

### LEARNING RATE 0.002

learning rate 0.002 batch size 500 converged to loss 21.987534 after 65.70072555541992
validation set accuracy 0.87 correct 870
test set accuracy 0.8524229074889867 correct 2322
training set accuracy 0.9078 correct 13617
"""
