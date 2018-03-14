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
init = tf.global_variables_initializer()
sess.run(init)

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
MSE = tf.reduce_mean(tf.reduce_sum((predY - y)**2, 1))
# Classification threshold is 0.5. Cast bool Tensor output by tf.greater to get a
# tensor of 1.'s and 0.'s, i.e. the classifications yhat, then subtract y and
# count nonzeros to get # of failures.
ACC = 1-tf.count_nonzero(tf.cast(tf.greater(predY, 0.5), tf.float32) - y)/tf.shape(y, out_type=tf.int64)[0]

learningRates = [0.005, 0.001, 0.0001] # 1.1
if len(sys.argv) == 2 and (sys.argv[1] == "2" or sys.argv[1] == "3"):
    learningRates = [0.005] # 1.2, 1.3
for learningRate, colour in zip(learningRates, ['b.', 'r.', 'g.']):
    # Training mechanism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
    train = optimizer.minimize(loss=MSE)

    init = tf.global_variables_initializer()
    sess.run(init)
    initialW = sess.run(W)  
    initialb = sess.run(b)

    Bs = [500] # 1.1, 1.3
    if len(sys.argv) == 2 and sys.argv[1] == "2":
        Bs = [500, 1500, 3500] # 1.2
    lams = [0] # 1.1, 1.2
    if len(sys.argv) == 2 and sys.argv[1] == "3":
        lams = [0, 0.001, 0.1, 1] # 1.3
    for B in Bs:
        for lam in lams:
            start = time.time()
            epochCounter = 0
            epoch = []
            loss = []
            WD = 0 # 1.1, 1.2
            if len(sys.argv) == 2 and sys.argv[1] == "3":
                WD = lam/2*(tf.norm(W)**2) # 1.3
            for iteration in range(1, 20001):
                # Shuffle data
                minibatch = random.sample(list(zip(trainDataReshaped, trainTarget)), B)
                minibatchData, minibatchTarget = zip(*minibatch)
                _ = sess.run(train, feed_dict={X: minibatchData, y: minibatchTarget})
                error = sess.run(MSE, feed_dict={X: trainDataReshaped, y: trainTarget})
                if WD != 0:
                    error += sess.run(WD, feed_dict={X: trainDataReshaped, y: trainTarget})
                epochCounter += B
                if epochCounter >= trainData.shape[0]:
                    epochCounter -= trainData.shape[0]
                    epoch.append(B*iteration/trainData.shape[0])
                    loss.append(error)
            end = time.time()
            print("learning rate " + str(learningRate) + " batch size " + str(B) + " lambda " + str(lam) + " converged to loss " + str(loss[-1]) + " after " + str(end-start))
            if len(sys.argv) == 1:
                plt.plot(epoch, loss, colour, mew=0.0) # 1.1
            if len(sys.argv) == 2 and sys.argv[1] == "3":
                accuracy = sess.run(ACC, feed_dict={X: validDataReshaped, y: validTarget}) # 1.3
                print("validation set accuracy " + str(accuracy))
                accuracy = sess.run(ACC, feed_dict={X: testDataReshaped, y: testTarget})
                print("test set accuracy " + str(accuracy))
                accuracy = sess.run(ACC, feed_dict={X: trainDataReshaped, y: trainTarget})
                print("training set accuracy " + str(accuracy))

if len(sys.argv) == 1:
    plt.show() # 1.1

"""
1.1
learning rate 0.005 batch size 500 lambda 0 converged to loss 0.24658377 after 152.4940106868744
learning rate 0.001 batch size 500 lambda 0 converged to loss 1.0335683 after 158.31548500061035
learning rate 0.0001 batch size 500 lambda 0 converged to loss 7.590733 after 154.6288275718689
"""
"""
1.2
learning rate 0.005 batch size 500 lambda 0 converged to loss 0.2652751 after 152.71601796150208
learning rate 0.005 batch size 1500 lambda 0 converged to loss 0.12633248 after 192.21855854988098
learning rate 0.005 batch size 3500 lambda 0 converged to loss 0.0819071 after 320.7755215167999
"""
"""
1.3
learning rate 0.005 batch size 500 lambda 0 converged to loss 0.26546586 after 209.3253424167633
validation set accuracy 0.86
test set accuracy 0.806896551724138
training set accuracy 0.846
learning rate 0.005 batch size 500 lambda 0.001 converged to loss 0.18263519 after 205.7136116027832
validation set accuracy 0.9299999999999999
test set accuracy 0.8689655172413793
training set accuracy 0.9188571428571428
learning rate 0.005 batch size 500 lambda 0.1 converged to loss 4.4184537 after 206.76141834259033
validation set accuracy 0.94
test set accuracy 0.896551724137931
training set accuracy 0.9451428571428572
learning rate 0.005 batch size 500 lambda 1 converged to loss 34.576294 after 206.3271198272705
validation set accuracy 0.94
test set accuracy 0.903448275862069
training set accuracy 0.9562857142857143
"""
