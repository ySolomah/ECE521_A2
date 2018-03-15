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
# 1 x 784
W = tf.Variable(tf.truncated_normal([1, trainData.shape[1] * trainData.shape[2]], stddev=1.0, name='weights'))
b = tf.Variable(0.0, name='biases')
# None x 784
X = tf.placeholder(tf.float32, [None, trainData.shape[1] * trainData.shape[2]], name='input_x')
y = tf.placeholder(tf.float32, [None, 1], name='target_y')

# Graph definition
predY = tf.matmul(X, tf.transpose(W)) + b

# Error definition
#MSE = tf.reduce_mean(tf.reduce_sum((predY - y)**2, 1))
CE = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=predY))
# Classification threshold is 0.5. Cast bool Tensor output by tf.greater to get a
# tensor of 1.'s and 0.'s, i.e. the classifications yhat, then subtract y and
# count nonzeros to get # of failures.
ACC = 1-tf.count_nonzero(tf.cast(tf.greater(predY, 0.5), tf.float32) - y)/tf.shape(y, out_type=tf.int64)[0]

learningRate = 0.001
optimizers = [tf.train.GradientDescentOptimizer(learning_rate=learningRate), tf.train.AdamOptimizer(learning_rate=learningRate)]
labels = ['SGD', 'Adam']
for optimizer, label, colour in zip(optimizers, labels, ['b.', 'r.']):
    # Training mechanism
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
    WD = lam/2*(tf.norm(W)**2)
    for iteration in range(1, 5001):
        # Shuffle data
        minibatch = random.sample(list(zip(trainDataReshaped, trainTarget)), B)
        minibatchData, minibatchTarget = zip(*minibatch)
        _ = sess.run(train, feed_dict={X: minibatchData, y: minibatchTarget})
        epochCounter += B
        if epochCounter >= trainData.shape[0]:
            epochCounter -= trainData.shape[0]
            epoch.append(B*iteration/trainData.shape[0])
            error = sess.run(CE, feed_dict={X: trainDataReshaped, y: trainTarget})
            error += sess.run(WD, feed_dict={X: trainDataReshaped, y: trainTarget})
            trainLoss.append(error)
            end = time.time()
    plt.plot(epoch, trainLoss, colour, mew=0.0, label=label)
plt.legend()
plt.show()