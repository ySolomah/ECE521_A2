import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import numpy as np

rng = np.random

batch_size = 150
learning_rate = [0.002]
lambdas = [0.015]
num_iter = 5000

colours = ['b',  'r', 'g', 'y']
markers = ['o']

plt.figure().suptitle("Facescrub classification")

data = np.load("data.npy")/255
data = np.reshape(data, [-1, 32, 32])

target = np.load("target.npy")

np.random.seed(45689)
rnd_idx = np.arange(np.shape(data)[0])
np.random.shuffle(rnd_idx)

trBatch = int(0.803*len(rnd_idx))
validBatch = int(0.1*len(rnd_idx))

trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
        data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
        data[rnd_idx[trBatch + validBatch+1:-1],:]

trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], 0], \
        target[rnd_idx[trBatch+1:trBatch + validBatch], 0],\
        target[rnd_idx[trBatch + validBatch + 1:-1], 0]

print(trainData.shape)
print(trainTarget.shape)

trainTarget = np.expand_dims(trainTarget, axis=1)
z = np.zeros((trainTarget.shape[0], 5))
trainTarget = np.concatenate((trainTarget, z), axis=1)

validTarget = np.expand_dims(validTarget, axis=1)
z = np.zeros((validTarget.shape[0], 5))
validTarget = np.concatenate((validTarget, z), axis=1)

testTargetCopy = np.copy(testTarget)
testTarget = np.expand_dims(testTarget, axis=1)
z = np.zeros((testTarget.shape[0], 5))
testTarget = np.concatenate((testTarget, z), axis=1)
  
for edit_array in [trainTarget, validTarget, testTarget]:
    for row in edit_array:
        idx = row[0].astype(int)
        row[0] = 0
        row[idx] = 1


print(trainData.shape, trainData.size)
print(trainTarget.shape)
for i, lams in enumerate(zip(lambdas, colours)):
    for j, lrs in enumerate(zip(learning_rate, markers)):
        colour = lams[1]
        lam = lams[0]
        marker = lrs[1]
        lr = lrs[0]
        epoch_array = []
        loss_array = []
        print("Learning Rate : " + str(lr))
        W = tf.Variable(tf.random_normal([1, 6, trainData.shape[1] * trainData.shape[2]]), name="weight")
        b = tf.Variable(tf.random_normal([1, 6]), name="bias")
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

        plt.plot(epoch_array, loss_array, colour + marker, label = "lr: " + str(lr) + " with lam: " + str(lam))
plt.xlabel("epoch")
plt.ylabel("accuracy %")
plt.legend()
plt.show()
            

        