#from __future__ import print_function
import numpy as np
import h5py
import itertools

# Speed
def genData(fileNames, batchSize = 200):
    #fileNames = datasetFilesTrain
    #batchSize = 200 
    batchX = np.zeros((batchSize, 88, 200, 3))    
    batchY = np.zeros((batchSize, 28))
    idx = 0       
    while True: # to make sure we never reach the end
        counter = 0
        while counter<=batchSize-1:
            idx = np.random.randint(len(fileNames)-1) 
            try:
                data = h5py.File(fileNames[idx], 'r')
            except: 
                print(idx, fileNames[idx])
            
            dataIdx = np.random.randint(200-1) 
            batchX[counter] = data['rgb'][dataIdx]
            batchY[counter] = data['targets'][dataIdx]
            counter += 1
            data.close()
        yield (batchX, batchY)

# control 
def genBranch(fileNames, branchNum = 2, batchSize = 200):
    #fileNames = datasetFilesTrain
    #branchNum = 3 # Control signal, int ( 2 Follow lane, 3 Left, 4 Right, 5 Straight)
    #batchSize = 200 
    batchX = np.zeros((batchSize, 88, 200, 3))    
    batchY = np.zeros((batchSize, 28))
    idx = 0       
    while True: # to make sure we never reach the end
        counter = 0
        while counter<=batchSize-1:
            idx = np.random.randint(len(fileNames)-1) 
            try:
                data = h5py.File(fileNames[idx], 'r')
            except: 
                print(idx, fileNames[idx])
            
            dataIdx = np.random.randint(200-1)

            if data['targets'][dataIdx][24] == branchNum:
                batchX[counter] = data['rgb'][dataIdx]
                batchY[counter] = data['targets'][dataIdx]
                counter += 1
                data.close()
        yield (batchX, batchY)

