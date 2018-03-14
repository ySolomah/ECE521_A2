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

learningRates = [0.005] #, 0.001, 0.0001] # 2.1.1 post-tuning
for learningRate, colour in zip(learningRates, ['b.']): #, 'r.', 'g.']):
    # Training mechanism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
    train = optimizer.minimize(loss=CE)

    init = tf.global_variables_initializer()
    sess.run(init)
    initialW = sess.run(W)  
    initialb = sess.run(b)

    Bs = [500] # 2.1.1
    lams = [0.01] # 2.1.1
    for B in Bs:
        for lam in lams:
            start = time.time()
            epochCounter = 0
            epoch = []
            trainLoss = []
            validLoss = []
            trainAccuracies = []
            validAccuracies = []
            WD = lam/2*(tf.norm(W)**2) # 2.1.1
            for iteration in range(1, 5001):
                # Shuffle data
                minibatch = random.sample(list(zip(trainDataReshaped, trainTarget)), B)
                minibatchData, minibatchTarget = zip(*minibatch)
                _ = sess.run(train, feed_dict={X: minibatchData, y: minibatchTarget})
                error = sess.run(CE, feed_dict={X: trainDataReshaped, y: trainTarget})
                error += sess.run(WD, feed_dict={X: trainDataReshaped, y: trainTarget})
                trainAccuracy = sess.run(ACC, feed_dict={X: trainDataReshaped, y: trainTarget})
                validAccuracy = sess.run(ACC, feed_dict={X: validDataReshaped, y: validTarget})
                epochCounter += B
                if epochCounter >= trainData.shape[0]:
                    epochCounter -= trainData.shape[0]
                    epoch.append(B*iteration/trainData.shape[0])
                    trainLoss.append(error)
                    validError = sess.run(CE, feed_dict={X: validDataReshaped, y: validTarget})
                    validError += sess.run(WD, feed_dict={X: validDataReshaped, y: validTarget})
                    validLoss.append(validError)
                    trainAccuracies.append(sess.run(ACC, feed_dict={X: trainDataReshaped, y: trainTarget}))
                    validAccuracies.append(sess.run(ACC, feed_dict={X: validDataReshaped, y: validTarget}))
                    end = time.time()
            print("learning rate " + str(learningRate) + " batch size " + str(B) + " lambda " + str(lam) + " converged to loss " + str(trainLoss[-1]) + " after " + str(end-start))
            accuracy = sess.run(ACC, feed_dict={X: validDataReshaped, y: validTarget})
            print("validation set accuracy " + str(accuracy))
            accuracy = sess.run(ACC, feed_dict={X: testDataReshaped, y: testTarget})
            print("test set accuracy " + str(accuracy))
            accuracy = sess.run(ACC, feed_dict={X: trainDataReshaped, y: trainTarget})
            print("training set accuracy " + str(accuracy))

            fig, ax1 = plt.subplots()
            ax1.plot(epoch, trainLoss, 'b.', mew=0.0, label='training loss')
            ax1.plot(epoch, validLoss, 'm.', mew=0.0, label='validation loss')
            ax1.legend(loc="center left")
            ax1.tick_params('y', colors='b')
            ax1.set_xlabel("epoch")
            ax2 = ax1.twinx() 
            ax2.plot(epoch, trainAccuracies, 'g.', mew=0.0, label='training accuracy')
            ax2.plot(epoch, validAccuracies, 'r.', mew=0.0, label='validation accuracy')
            ax2.legend(loc="center right")
            ax2.tick_params('y', colors='r')
            vals = ax2.get_yticks()
            ax2.set_yticklabels(['{:d}%'.format(int(x*100)) for x in vals])
            ax2.set_ylabel("accuracy")
            fig.tight_layout()
            plt.show()

"""
learning rate 0.005 batch size 500 lambda 0.01 converged to loss 3.1157613 after 58.167991638183594
validation set accuracy 0.96
test set accuracy 0.9586206896551724
training set accuracy 0.9577142857142857
learning rate 0.001 batch size 500 lambda 0.01 converged to loss 3.430304 after 56.885133028030396
validation set accuracy 0.9
test set accuracy 0.9172413793103449
training set accuracy 0.9165714285714286
learning rate 0.0001 batch size 500 lambda 0.01 converged to loss 4.6389065 after 57.90668272972107
validation set accuracy 0.6699999999999999
test set accuracy 0.6413793103448275
training set accuracy 0.6374285714285715
"""
"""
learning rate 0.005 batch size 500 lambda 0.01 converged to loss 3.2479236 after 99.36951732635498
validation set accuracy 0.98
test set accuracy 0.9586206896551724
training set accuracy 0.9622857142857143
"""