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
BEST IS LEARNING RATE 0.005 WITH LAMBDA 0.1 validation set accuracy 0.8152173913043478 
"""
learningRates = [0.005, 0.001, 0.0001]
lams = [0]
lams = [0, 0.001, 0.1, 1]
for learningRate in learningRates:
    for lam in lams:
        B = 300
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
            minibatch = random.sample(list(zip(trainData, trainTargetOneHot)), B)
            minibatchData, minibatchTarget = zip(*minibatch)
            _ = sess.run(train, feed_dict={X: minibatchData, y: minibatchTarget})
            epochCounter += B
            if epochCounter >= trainData.shape[0]:
                epochCounter -= trainData.shape[0]
                epoch.append(B*iteration/trainData.shape[0])
                error = sess.run(CE, feed_dict={X: trainData, y: trainTargetOneHot})
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
BEST IS LEARNING RATE 0.005 WITH LAMBDA 0.001 validation set accuracy 0.8152173913043478 
BEST TEST ACCURACY IS 0.8602150537634409 learning rate 0.005 batch size 300 lambda 0.001
"""
"""
learning rate 0.005 batch size 300 lambda 0 converged to loss 0.011897072 after 25.458617687225342
validation set accuracy 0.7934782608695652 correct 73
test set accuracy 0.7849462365591398 correct 73
training set accuracy 1.0 correct 747
learning rate 0.005 batch size 300 lambda 0.001 converged to loss 0.124096155 after 24.807463884353638
validation set accuracy 0.8152173913043478 correct 75
test set accuracy 0.8602150537634409 correct 80
training set accuracy 0.9946452476572959 correct 743
learning rate 0.005 batch size 300 lambda 0.1 converged to loss 1.0180767 after 25.407121896743774
validation set accuracy 0.5543478260869565 correct 51
test set accuracy 0.6666666666666666 correct 62
training set accuracy 0.7054886211512718 correct 527
learning rate 0.005 batch size 300 lambda 1 converged to loss 1.548487 after 25.847491025924683
validation set accuracy 0.5108695652173914 correct 47
test set accuracy 0.4946236559139785 correct 46
training set accuracy 0.4390896921017403 correct 328

learning rate 0.001 batch size 300 lambda 0 converged to loss 0.09990199 after 26.156989336013794
validation set accuracy 0.7934782608695652 correct 73
test set accuracy 0.7956989247311828 correct 74
training set accuracy 0.9866131191432396 correct 737
learning rate 0.001 batch size 300 lambda 0.001 converged to loss 0.14741674 after 26.082664012908936
validation set accuracy 0.7717391304347826 correct 71
test set accuracy 0.8172043010752689 correct 76
training set accuracy 0.9879518072289156 correct 738
learning rate 0.001 batch size 300 lambda 0.1 converged to loss 0.9445312 after 26.179368495941162
validation set accuracy 0.7391304347826086 correct 68
test set accuracy 0.7849462365591398 correct 73
training set accuracy 0.7751004016064257 correct 579
learning rate 0.001 batch size 300 lambda 1 converged to loss 1.4940742 after 26.359615802764893
validation set accuracy 0.4673913043478261 correct 43
test set accuracy 0.5806451612903226 correct 54
training set accuracy 0.5809906291834003 correct 434

learning rate 0.0001 batch size 300 lambda 0 converged to loss 1.1971946 after 26.843743801116943
validation set accuracy 0.5217391304347826 correct 48
test set accuracy 0.6559139784946236 correct 61
training set accuracy 0.6492637215528781 correct 485
learning rate 0.0001 batch size 300 lambda 0.001 converged to loss 1.3447523 after 25.072758197784424
validation set accuracy 0.5 correct 46
test set accuracy 0.5591397849462365 correct 52
training set accuracy 0.6050870147255689 correct 452
learning rate 0.0001 batch size 300 lambda 0.1 converged to loss 2.2411935 after 26.37841033935547
validation set accuracy 0.31521739130434784 correct 29
test set accuracy 0.27956989247311825 correct 26
training set accuracy 0.3172690763052209 correct 237
learning rate 0.0001 batch size 300 lambda 1 converged to loss 3.4235997 after 24.62093758583069
validation set accuracy 0.2391304347826087 correct 22
test set accuracy 0.3118279569892473 correct 29
training set accuracy 0.25167336010709507 correct 188

"""
