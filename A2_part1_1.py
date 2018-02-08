import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import numpy as np

rng = np.random

batch_size = 500
learning_rate = [0.005, 0.001, 0.0001]
lam = 0
num_iter = 20000
num_runs_per_epoch = 7




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
    for lr, colour in zip(learning_rate, ['bo', 'ro', 'go']):
        epoch_array = []
        loss_array = []
        print("Learning Rate : " + str(lr))
        W = tf.Variable(tf.random_normal([1, trainData.shape[1] * trainData.shape[2]]), name="weight")
        b = tf.Variable(tf.random_normal([1]), name="bias")
        X = tf.placeholder("float")
        y = tf.placeholder("float")
        loss = (1 / (2 * batch_size)) * tf.reduce_sum(tf.square(tf.add(tf.reduce_sum(tf.multiply(W, X), 1), b) - y))
        regularize = tf.reduce_sum(tf.square(tf.transpose(W))) * lam / 2
        total_loss = loss + regularize
        if(lam == 0):
            total_loss = loss
        optim = tf.train.GradientDescentOptimizer(lr).minimize(total_loss)

        init = tf.global_variables_initializer()

        trainDataReshaped = trainData.reshape([-1, batch_size, trainData.shape[1] * trainData.shape[2]])
        trainTargetReshaped = trainTarget.reshape([-1, batch_size])
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(int(num_iter * batch_size / trainData.shape[0])):
                new_epoch = True
                for miniBatchData, miniBatchTarget in zip(trainDataReshaped, trainTargetReshaped):
                    if(new_epoch):
                        new_epoch = False
                        if(epoch % 500 == 0):
                            print("\n\nEpoch : " + str(epoch) +  "\n Loss : " + str(sess.run(loss, feed_dict={X: miniBatchData, y: miniBatchTarget})) + "\n Total Loss : " + str(sess.run(total_loss, feed_dict={X: miniBatchData, y: miniBatchTarget})))
                        #plt.plot(epoch, sess.run(loss, feed_dict={X: miniBatchData, y: miniBatchTarget}), colour, label="lr: " + str(lr))
                        epoch_array.append(epoch)
                        loss_array.append(sess.run(loss, feed_dict={X: miniBatchData, y: miniBatchTarget}))
                    sess.run(optim, feed_dict={X: miniBatchData, y: miniBatchTarget})
        plt.plot(epoch_array, loss_array, colour, label = "lr: " + str(lr))
    plt.legend()
    plt.show()
                

        