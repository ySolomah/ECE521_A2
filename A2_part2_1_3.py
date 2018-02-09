import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import numpy as np

rng = np.random

batch_size = 500
lr = 0.001
num_iter = 5000

scoreLosses = [0, 1]

plt.figure().suptitle("sigmoid cross entropy SGD vs Adam")
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
    for scoreLoss, colour in zip(scoreLosses, ['mo', 'bo']):
        if(scoreLoss == 0):
            optimName = "Cross Entropy"
        else:
            optimName = "Normal Eqn"
        epoch_array = []
        loss_array = []
        print("Learning Rate : " + str(lr))
        tf.set_random_seed(121)
        W = tf.Variable(tf.random_normal([1, trainData.shape[1] * trainData.shape[2]]), name="weight")
        b = tf.Variable(tf.random_normal([1]), name="bias")
        X = tf.placeholder("float")
        y = tf.placeholder("float")
        if(scoreLoss == 0):
            loss = (1 / batch_size) * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.add(tf.reduce_sum(tf.multiply(W, X), 1), b), labels=y))
            score = tf.sigmoid(tf.add(tf.reduce_sum(tf.multiply(W, X), 1), b))
        else:
            score = tf.add(tf.reduce_sum(tf.multiply(W, X), 1), b)
            loss = (1 / (2 * batch_size)) * tf.reduce_sum(tf.square(tf.add(tf.reduce_sum(tf.multiply(W, X), 1), b) - y))


        total_loss = loss
        optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)

        init = tf.global_variables_initializer()

        trainDataReshaped = trainData.reshape([-1, batch_size, trainData.shape[1] * trainData.shape[2]])
        trainTargetReshaped = trainTarget.reshape([-1, batch_size])
        testDataReshaped = testData.reshape([testTarget.shape[0], testData.shape[1] * testData.shape[2]])
        testTargetReshaped = testTarget.reshape([testTarget.shape[0]])

        with tf.Session() as sess:
            sess.run(init)
            #for epoch in range(int(num_iter / trainDataReshaped.shape[0])):
            for epoch in range(30):
                new_epoch = True
                for miniBatchData, miniBatchTarget in zip(trainDataReshaped, trainTargetReshaped):
                    if(new_epoch):
                        new_epoch = False
                        if(epoch % 100 == 0):
                            print("\n\nEpoch : " + str(epoch) +  "\n Loss : " + str(sess.run(loss, feed_dict={X: miniBatchData, y: miniBatchTarget})) + "\n Total Loss : " + str(sess.run(total_loss, feed_dict={X: miniBatchData, y: miniBatchTarget})))
                        epoch_array.append(epoch)
                        guesses = sess.run(score, feed_dict={X: testDataReshaped, y: testTargetReshaped})
                        guesses = np.around(guesses)
                        guesses = np.clip(guesses, 0, 1)
                        accuracy = 1 - (np.absolute(guesses - testTargetReshaped).sum() / testTargetReshaped.shape[0])
                        loss_array.append(accuracy)
                    sess.run(optim, feed_dict={X: miniBatchData, y: miniBatchTarget})

        plt.plot(epoch_array, loss_array, colour, label = "loss: " + optimName)
    plt.xlabel("epoch")
    plt.ylabel("accuracy %")
    plt.legend()
    plt.show()
                

        