
import pandas as pd
import numpy as np
import os

def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    return result

def getMnistData(digits, length=-1, sigmoid=True):
    mnist_dataset = pd.read_csv('./data/mnist_train.csv')
    mnist_validate = pd.read_csv('./data/mnist_test.csv')

    mnist_data = mnist_dataset.values[np.where(np.isin(mnist_dataset.values[:,0], digits))]
    mnist_validate = mnist_validate.values[np.where(np.isin(mnist_validate.values[:,0], digits))]

    if length > 0:
        mnist_data = mnist_data[:length]
        mnist_validate = mnist_validate[:length]

    if sigmoid:
        mnist_data[:, 1:] = round(sigmoid(mnist_data[:, 1:] - 128))
        mnist_validate[:, 1:] = round(sigmoid(mnist_validate[:, 1:] - 128))

    return mnist_data, mnist_validate

def getDroneData(length=-1, sigmoid=True, color=False):

    if color:
        data = pd.read_csv('/home/petedowney/github/MLQuanta/data/labels/dronesColor.csv')
    else:
        data = pd.read_csv('/home/petedowney/github/MLQuanta/data/labels/drones.csv')

    data = data.values

    if sigmoid:
        data[:, 1:] = round(sigmoid(data[:, 1:] - 128))

    if length > 0:
        data = data[:length]
    return data

def getBirdData(length = -1, sigmoid=True, color=False):
    # bird_data = pd.read_csv(r'C:\Users\rthab\miniconda3\envs\MLQuanta\MLQuanta\data\labels\birds.csv')
    if color:
        bird_data = pd.read_csv('/home/petedowney/github/MLQuanta/data/labels/birdsColor.csv')
    else:
        bird_data = pd.read_csv('/home/petedowney/github/MLQuanta/data/labels/birds.csv')
        
    bird_data = bird_data.values

    if sigmoid:
        bird_data[:, 1:] = round(sigmoid(bird_data[:, 1:] - 128))

    if length > 0:
        bird_data = bird_data[:length]
    return bird_data 
