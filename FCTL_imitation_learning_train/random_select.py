from __future__ import print_function
import fnmatch
import os
import h5py
import numpy as np
import math

datasetDirVal = './dataset/SeqVal/link'
savename = 'output/Town02_CSL_5w_'
maxdataSize = 800
totalsize = 50000

filenums = math.ceil(totalsize / maxdataSize)

datasetFilesVal = []
for root, dirnames, filenames in os.walk(datasetDirVal):
    for filename in fnmatch.filter(filenames, '*.h5'):
        datasetFilesVal.append(os.path.join(root, filename))

count = 0
for h5file in datasetFilesVal:
    with h5py.File(h5file, 'r') as h5data:
        count += h5data['rgb'].shape[0]
print("Total image:", count)

per = 0

filecount = 0
while filecount < filenums:
    with h5py.File(savename + str(filecount) + ".h5") as h5save:
        h5save.create_dataset('rgb', shape=(0, 88, 200, 3), dtype='uint8', maxshape=(maxdataSize, 88, 200, 3))
        dataRecord = 1
        while dataRecord <= maxdataSize and filecount * maxdataSize + dataRecord <= totalsize:
            idx = np.random.randint(len(datasetFilesVal) - 1)
            with h5py.File(datasetFilesVal[idx], 'r') as h5read:
                count = h5read['rgb'].shape[0]
                dataIdx = np.random.randint(count - 1)
                image = h5read['rgb'][dataIdx]
                h5save['rgb'].resize((dataRecord, 88, 200, 3))
                h5save['rgb'][-1, :] = image
            dataRecord += 1
            newper = int((filecount * maxdataSize + dataRecord) / totalsize * 100)
            if newper != per:
                print("Percentage:", newper, "%")
            per = newper
    filecount += 1
