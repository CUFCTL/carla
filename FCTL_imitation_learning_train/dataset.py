# from __future__ import print_function
import numpy as np
import h5py
import itertools

def genData(fileNames, batchSize=200):
    # fileNames = datasetFilesTrain
    # batchSize = 200
    batchX = np.zeros((batchSize, 88, 200, 3))
    batchY = np.zeros((batchSize, 4))
    idx = 0
    while True:  # to make sure we never reach the end
        counter = 0
        while counter <= batchSize - 1:
            idx = np.random.randint(len(fileNames) - 1)
            try:
                data = h5py.File(fileNames[idx], 'r')
            except:
                print(idx, fileNames[idx])

            count = data['rgb'].shape[0]
            dataIdx = np.random.randint(count - 1)
            batchX[counter] = data['rgb'][dataIdx]
            batchY[counter] = data['targets'][dataIdx]
            counter += 1
            data.close()
        yield (batchX, batchY)
