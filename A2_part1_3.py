import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import numpy as np
import time

rng = np.random

batch_size = 500
lr = 0.005
lambdas = [0, 0.001, 0.1, 1]
num_iter = 20000

# Test accuracy: 0.35 with lambda: 0
# Test accuracy: 0.48 with lambda: 0.001
# Test accuracy: 0.662 with lambda: 0.1
# Test accuracy: 0.655 with lambda: 1



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
    print(trainData.shape, trainData.size)
    print(trainTarget.shape)
    plt.figure().suptitle("validation accuracy with regularization")
    for lam, colour in zip(lambdas, ['bo', 'ro', 'go', 'yo']):
        start = time.time()
        epoch_array = []
        loss_array = []
        W = tf.Variable(tf.random_normal([1, trainData.shape[1] * trainData.shape[2]]), name="weight")
        b = tf.Variable(tf.random_normal([1]), name="bias")
        X = tf.placeholder("float")
        y = tf.placeholder("float")
        score = tf.add(tf.reduce_sum(tf.multiply(W, X), 1), b)
        loss = (1 / (2 * batch_size)) * tf.reduce_sum(tf.square(tf.add(tf.reduce_sum(tf.multiply(W, X), 1), b) - y))
        regularize = tf.reduce_sum(tf.square(tf.transpose(W))) * lam / 2
        total_loss = loss + regularize
        if(lam == 0):
            total_loss = loss
        optim = tf.train.GradientDescentOptimizer(lr).minimize(total_loss)
        init = tf.global_variables_initializer()

        trainDataReshaped = trainData.reshape([-1, batch_size, trainData.shape[1] * trainData.shape[2]])
        trainTargetReshaped = trainTarget.reshape([-1, batch_size])
        validDataReshaped = validData.reshape([100, validData.shape[1] * validData.shape[2]])
        validTargetReshaped = validTarget.reshape([100])
        testDataReshaped = testData.reshape([testTarget.shape[0], testData.shape[1] * testData.shape[2]])
        testTargetReshaped = testTarget.reshape([testTarget.shape[0]])
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(int(num_iter / trainDataReshaped.shape[0])):
                new_epoch = True
                for miniBatchData, miniBatchTarget in zip(trainDataReshaped, trainTargetReshaped):
                    if(new_epoch):
                        new_epoch = False
                        if(epoch % 50 == 0):
                            guesses = sess.run(score, feed_dict={X: validDataReshaped, y: validTargetReshaped})
                            guesses = np.around(guesses)
                            guesses = np.clip(guesses, 0, 1)
                            accuracy = 1 - np.absolute(guesses - validTargetReshaped).sum() / 100
                            if(epoch % 200 == 0):
                                print("\n\nEpoch : " + str(epoch) +  "\n Loss : " + str(sess.run(loss, feed_dict={X: miniBatchData, y: miniBatchTarget})) + "\n Total Loss : " + str(sess.run(total_loss, feed_dict={X: miniBatchData, y: miniBatchTarget})))
                                print(accuracy)
                            epoch_array.append(epoch)
                            loss_array.append(accuracy)
                    sess.run(optim, feed_dict={X: miniBatchData, y: miniBatchTarget})
            guesses = sess.run(score, feed_dict={X: testDataReshaped, y: testTargetReshaped})
            guesses = np.around(guesses)
            guesses = np.clip(guesses, 0, 1)
            accuracy = 1 - np.absolute(guesses - testTargetReshaped).sum() / 145
            print("Test accuracy: " + str(accuracy) + " with lambda: " + str(lam))
        plt.plot(epoch_array, loss_array, colour, label="lambda: " + str(lam))
        end = time.time()
        print("Time for lambda : " + str(lam) + " is : " + str(end-start))
    plt.xlabel("epoch")
    plt.ylabel("accuracy %")
    plt.legend()
    plt.show()
                

        