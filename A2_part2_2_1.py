import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import numpy as np

rng = np.random

batch_size = 500
learning_rate = [0.01, 0.005, 0.002]
lam = 0.01
num_iter = 3000



plt.figure().suptitle("notMNIST softmax training accuracy")

with np.load("notMNIST.npz") as data:
    Data, Target = data ["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx]/255.
    Target = Target[randIndx]

    trainData, trainTarget = Data[:15000], Target[:15000]
    trainTarget = np.expand_dims(trainTarget, axis=1)
    z = np.zeros((trainTarget.shape[0], 9))
    trainTarget = np.concatenate((trainTarget, z), axis=1)

    validData, validTarget = Data[15000:16000], Target[15000:16000]
    validTarget = np.expand_dims(validTarget, axis=1)
    z = np.zeros((validTarget.shape[0], 9))
    validTarget = np.concatenate((validTarget, z), axis=1)

    testData, testTarget = Data[16000:], Target[16000:]
    testTargetCopy = np.copy(testTarget)
    testTarget = np.expand_dims(testTarget, axis=1)
    z = np.zeros((testTarget.shape[0], 9))
    testTarget = np.concatenate((testTarget, z), axis=1)
      
    for edit_array in [trainTarget, validTarget, testTarget]:
        for row in edit_array:
            idx = row[0].astype(int)
            row[0] = 0
            row[idx] = 1


    print(trainData.shape, trainData.size)
    print(trainTarget.shape)

    for lr, colour in zip(learning_rate, ['mo', 'bo', 'ro', 'go', 'yo']):
        epoch_array = []
        loss_array = []
        print("Learning Rate : " + str(lr))
        W = tf.Variable(tf.random_normal([1, 10, trainData.shape[1] * trainData.shape[2]]), name="weight")
        b = tf.Variable(tf.random_normal([1, 10]), name="bias")
        X = tf.placeholder("float")
        y = tf.placeholder("float")

        temp1 = tf.expand_dims(tf.transpose(X), axis=0)
        temp2 = tf.matmul(W, temp1)
        temp3 = tf.transpose(tf.reduce_sum(temp2, 0))
        temp4 = tf.add(temp3, b)
        temp4_5 = tf.nn.softmax_cross_entropy_with_logits(logits=temp4, labels=y)
        sigmoids = tf.nn.sigmoid_cross_entropy_with_logits(logits=temp4, labels=y)
        arg_max = tf.argmax(sigmoids, dimension=1)
        softmax_acc = tf.argmax(tf.nn.softmax(temp4), dimension=1)
        temp5 = tf.reduce_sum(temp4_5)
        loss = (1 / batch_size) * temp5
        regularize = tf.reduce_sum(tf.square(tf.transpose(W))) * lam / 2
        total_loss = loss + regularize

        optim = tf.train.AdamOptimizer(lr).minimize(total_loss)

        init = tf.global_variables_initializer()

        trainDataReshaped = trainData.reshape([-1, batch_size, trainData.shape[1] * trainData.shape[2]])
        trainTargetReshaped = trainTarget.reshape([-1, batch_size, trainTarget.shape[1]])
        testDataReshaped = testData.reshape([testData.shape[0], testData.shape[1] * testData.shape[2]])
        testTargetReshaped = testTarget.reshape([testTarget.shape[0], testTarget.shape[1]])

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(int(num_iter / trainDataReshaped.shape[0])):
                new_epoch = True
                for miniBatchData, miniBatchTarget in zip(trainDataReshaped, trainTargetReshaped):
                    if(new_epoch):
                        new_epoch = False
                        if(epoch % 100 == 0):
                            print("\n\nEpoch : " + str(epoch) +  "\n Loss : " + str(sess.run(loss, feed_dict={X: miniBatchData, y: miniBatchTarget})) + "\n Total Loss : " + str(sess.run(total_loss, feed_dict={X: miniBatchData, y: miniBatchTarget})))
                        epoch_array.append(epoch)
                        #guesses = sess.run(temp4_5, feed_dict={X: testDataReshaped, y: testTargetReshaped})
                        #accuracy = (guesses.sum() / guesses.shape[0])
                        #loss_array.append(accuracy)
                        #guesses = sess.run(arg_max, feed_dict={X: testDataReshaped, y: testTargetReshaped})
                        #accuracy = 1 - (((np.absolute(guesses - testTargetCopy)).clip(0, 1).sum())/guesses.shape[0])
                        guesses = sess.run(softmax_acc, feed_dict={X: testDataReshaped, y: testTargetReshaped})
                        accuracy = 1 - (((np.absolute(guesses - testTargetCopy)).clip(0, 1).sum())/guesses.shape[0])
                        loss_array.append(accuracy)

                    sess.run(optim, feed_dict={X: miniBatchData, y: miniBatchTarget})

        plt.plot(epoch_array, loss_array, colour, label = "lr: " + str(lr))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
                

        