import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import sys

def data_segmentation(data_path, target_path, task):
    # task = 0 >> select the name ID targets for face recognition task
    # task = 1 >> select the gender ID targets for gender recognition task
    data = np.load(data_path)/255
    data = np.reshape(data, [-1, 32*32])
    
    target = np.load(target_path)
    
    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    
    trBatch = int(0.8*len(rnd_idx))
    validBatch = int(0.1*len(rnd_idx))
    
    trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
            data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
            data[rnd_idx[trBatch + validBatch+1:-1],:]
    
    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
            target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
            target[rnd_idx[trBatch + validBatch + 1:-1], task]
    
    return trainData, validData, testData, trainTarget, validTarget, testTarget

trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation('data.npy', 'target.npy', 0)

sess = tf.InteractiveSession()

# 6 x 1024
W = tf.Variable(tf.truncated_normal([6, trainData.shape[1]], stddev=1.0, name='weights'))
b = tf.Variable(0.0, name='biases')
print(np.shape(trainData))
# None x 1024
X = tf.placeholder(tf.float32, [None, trainData.shape[1]], name='input_x')
y = tf.placeholder(tf.float32, [None, 6], name='target_y')

# Graph definition
predY = tf.matmul(X, tf.transpose(W)) + b

# Error definition
#MSE = tf.reduce_mean(tf.reduce_sum((predY - y)**2, 1))
CE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predY))
# Find the index of largest probability along predY axis 1. Find the index of 1 along y axis 1. Check if equal.
CORRECT = tf.count_nonzero(tf.cast(tf.equal(tf.argmax(predY, axis=1), tf.argmax(y, axis=1)), tf.float32))
ACC = CORRECT/tf.shape(y, out_type=tf.int64)[0]

print(trainTarget[:5])
print(tf.one_hot(trainTarget[:5], 6).eval())
trainTargetOneHot = tf.one_hot(trainTarget, 6).eval()
testTargetOneHot = tf.one_hot(testTarget, 6).eval()
validTargetOneHot = tf.one_hot(validTarget, 6).eval()
"""
BEST IS LEARNING RATE 0.005 WITH LAMBDA 0.1 validation set accuracy 0.8804347826086957
"""
learningRates = [0.005] #, 0.001, 0.001, 0.0001]
lams = [0.1]
# lams = [0, 0.001, 0.1, 1, .015]
for learningRate in learningRates:
    for lam in lams:
        # Training mechanism
        optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
        train = optimizer.minimize(loss=CE)

        init = tf.global_variables_initializer()
        sess.run(init)
        initialW = sess.run(W)  
        initialb = sess.run(b)

        B = 300
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
            minibatch = random.sample(list(zip(trainData, trainTargetOneHot)), B)
            minibatchData, minibatchTarget = zip(*minibatch)
            _ = sess.run(train, feed_dict={X: minibatchData, y: minibatchTarget})
            epochCounter += B
            if epochCounter >= trainData.shape[0]:
                epochCounter -= trainData.shape[0]
                epoch.append(B*iteration/trainData.shape[0])
                error = sess.run(CE, feed_dict={X: trainData, y: trainTargetOneHot})
                error += sess.run(WD, feed_dict={X: trainData, y: trainTargetOneHot})
                trainLoss.append(error)
                validError = sess.run(CE, feed_dict={X: validData, y: validTargetOneHot})
                validError += sess.run(WD, feed_dict={X: validData, y: validTargetOneHot})
                validLoss.append(validError)
                trainAccuracies.append(sess.run(ACC, feed_dict={X: trainData, y: trainTargetOneHot}))
                validAccuracies.append(sess.run(ACC, feed_dict={X: validData, y: validTargetOneHot}))
                end = time.time()
        print("learning rate " + str(learningRate) + " batch size " + str(B) + " lambda " + str(lam) + " converged to loss " + str(trainLoss[-1]) + " after " + str(end-start))
        correct = sess.run(CORRECT, feed_dict={X: validData, y: validTargetOneHot})
        accuracy = sess.run(ACC, feed_dict={X: validData, y: validTargetOneHot})
        print("validation set accuracy " + str(accuracy) + " correct " + str(correct))
        correct = sess.run(CORRECT, feed_dict={X: testData, y: testTargetOneHot})
        accuracy = sess.run(ACC, feed_dict={X: testData, y: testTargetOneHot})
        print("test set accuracy " + str(accuracy) + " correct " + str(correct))
        correct = sess.run(CORRECT, feed_dict={X: trainData, y: trainTargetOneHot})
        accuracy = sess.run(ACC, feed_dict={X: trainData, y: trainTargetOneHot})
        print("training set accuracy " + str(accuracy) + " correct " + str(correct))

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
BEST IS LEARNING RATE 0.005 WITH LAMBDA 0.1 validation set accuracy 0.8804347826086957
"""
"""
learning rate 0.005 batch size 300 lambda 0 converged to loss 0.012132303 after 26.280330896377563
validation set accuracy 0.8586956521739131 correct 79
test set accuracy 0.8279569892473119 correct 77
training set accuracy 1.0 correct 747
learning rate 0.005 batch size 300 lambda 0.001 converged to loss 2.7980819 after 26.051892280578613
validation set accuracy 0.8043478260869565 correct 74
test set accuracy 0.8602150537634409 correct 80
training set accuracy 1.0 correct 747
learning rate 0.005 batch size 300 lambda 0.1 converged to loss 274.8433 after 26.017375230789185
validation set accuracy 0.8804347826086957 correct 81
test set accuracy 0.8387096774193549 correct 78
training set accuracy 1.0 correct 747
learning rate 0.005 batch size 300 lambda 1 converged to loss 2791.954 after 25.761048555374146
validation set accuracy 0.8152173913043478 correct 75
test set accuracy 0.8494623655913979 correct 79
training set accuracy 1.0 correct 747

learning rate 0.001 batch size 300 lambda 0 converged to loss 0.12513088 after 25.555497407913208
validation set accuracy 0.8260869565217391 correct 76
test set accuracy 0.8279569892473119 correct 77
training set accuracy 0.9852744310575636 correct 736
learning rate 0.001 batch size 300 lambda 0.001 converged to loss 2.40627 after 25.42415690422058
validation set accuracy 0.782608695652174 correct 72
test set accuracy 0.7956989247311828 correct 74
training set accuracy 0.9892904953145917 correct 739
learning rate 0.001 batch size 300 lambda 0.1 converged to loss 232.86537 after 25.9399573802948
validation set accuracy 0.7934782608695652 correct 73
test set accuracy 0.7419354838709677 correct 69
training set accuracy 0.9825970548862115 correct 734
learning rate 0.001 batch size 300 lambda 1 converged to loss 2313.949 after 25.8808331489563
validation set accuracy 0.8369565217391305 correct 77
test set accuracy 0.8279569892473119 correct 77
training set accuracy 0.9892904953145917 correct 739

learning rate 0.0001 batch size 300 lambda 0 converged to loss 1.6798323 after 25.77893042564392
validation set accuracy 0.391304347826087 correct 36
test set accuracy 0.4731182795698925 correct 44
training set accuracy 0.5314591700133868 correct 397
learning rate 0.0001 batch size 300 lambda 0.001 converged to loss 3.7303348 after 25.624298095703125
validation set accuracy 0.5543478260869565 correct 51
test set accuracy 0.6451612903225806 correct 60
training set accuracy 0.5943775100401606 correct 444
learning rate 0.0001 batch size 300 lambda 0.1 converged to loss 234.81752 after 25.55628991127014
validation set accuracy 0.4673913043478261 correct 43
test set accuracy 0.5591397849462365 correct 52
training set accuracy 0.6037483266398929 correct 451
learning rate 0.0001 batch size 300 lambda 1 converged to loss 2338.626 after 26.00483727455139
validation set accuracy 0.41304347826086957 correct 38
test set accuracy 0.4731182795698925 correct 44
training set accuracy 0.5676037483266398 correct 424

### LEARNING RATE 0.002 + LAMBDA 0.015
learning rate 0.002 batch size 300 lambda 0 converged to loss 0.035655938 after 25.133418083190918
validation set accuracy 0.7717391304347826 correct 71
test set accuracy 0.7849462365591398 correct 73
training set accuracy 1.0 correct 747
learning rate 0.002 batch size 300 lambda 0.001 converged to loss 2.5622768 after 25.22196865081787
validation set accuracy 0.7934782608695652 correct 73
test set accuracy 0.8279569892473119 correct 77
training set accuracy 1.0 correct 747
learning rate 0.002 batch size 300 lambda 0.1 converged to loss 248.78593 after 25.15253233909607
validation set accuracy 0.7934782608695652 correct 73
test set accuracy 0.8064516129032258 correct 75
training set accuracy 0.998661311914324 correct 746
learning rate 0.002 batch size 300 lambda 1 converged to loss 2495.9768 after 25.65945029258728
validation set accuracy 0.8043478260869565 correct 74
test set accuracy 0.7849462365591398 correct 73
training set accuracy 0.9973226238286479 correct 745
learning rate 0.002 batch size 300 lambda 0.015 converged to loss 37.1731 after 25.17229437828064
validation set accuracy 0.8152173913043478 correct 75
test set accuracy 0.8387096774193549 correct 78
training set accuracy 1.0 correct 747
"""
