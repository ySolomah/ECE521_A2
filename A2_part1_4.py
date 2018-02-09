import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

# accuracy validation: 0.97
# accuracy test: 0.958620689655
# accuracy train: 0.993428571429
# accuracy mse: 0.00938463234807


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

 
    trainDataReshaped = trainData.reshape([3500, trainData.shape[1] * trainData.shape[2]])
    trainTargetReshaped = trainTarget.reshape([3500, 1])


    ones = np.full((trainDataReshaped.shape[0], 1), 1)
    trainDataReshaped = np.concatenate((trainDataReshaped, ones), axis=1)
    Weights = np.matmul(np.matmul(np.linalg.inv(np.matmul(trainDataReshaped.T, trainDataReshaped)), trainDataReshaped.T), trainTargetReshaped)


    validDataReshaped = validData.reshape([100, validData.shape[1] * validData.shape[2]])
    validTargetReshaped = validTarget.reshape([100, 1])
    ones = np.full((validDataReshaped.shape[0], 1), 1)
    validDataReshaped = np.concatenate((validDataReshaped, ones), axis=1)

    guesses = np.matmul(validDataReshaped, Weights)
    guesses = np.around(guesses)
    guesses = np.clip(guesses, 0, 1)
    accuracy = 1 - ((np.absolute(guesses - validTargetReshaped).sum()) / validTargetReshaped.shape[0])
    print("accuracy validation: " + str(accuracy))


    testDataReshaped = testData.reshape([testTarget.shape[0], testData.shape[1] * testData.shape[2]])
    testTargetReshaped = testTarget.reshape([testTarget.shape[0], 1])
    ones = np.full((testDataReshaped.shape[0], 1), 1)
    testDataReshaped = np.concatenate((testDataReshaped, ones), axis=1)
    guesses = np.matmul(testDataReshaped, Weights)
    guesses = np.around(guesses)
    guesses = np.clip(guesses, 0, 1)
    accuracy = 1 - ((np.absolute(guesses - testTargetReshaped).sum()) / testTargetReshaped.shape[0])
    print("accuracy test: " + str(accuracy))

    guesses = np.matmul(trainDataReshaped, Weights)
    mse = 1 / (2*(guesses.shape[0])) * numpy.sum(numpy.square(guesses - trainTargetReshaped))
    guesses = np.around(guesses)
    guesses = np.clip(guesses, 0, 1)
    accuracy = 1 - ((np.absolute(guesses - trainTargetReshaped).sum()) / trainTargetReshaped.shape[0])
    print("accuracy train: " + str(accuracy))  
    print("accuracy mse: " + str(mse)) 

        