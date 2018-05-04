import h5py
import glob

town_name = 'Town01'
data_dir = 'data'

datasetDirTrain = './' + data_dir + '/' + town_name + '/SeqTrain/'
datasetDirVal = './' + data_dir + '/' + town_name + '/SeqVal/'

datasetFilesTrain = glob.glob(datasetDirTrain + '*.h5')
datasetFilesVal = glob.glob(datasetDirVal + '*.h5')

trainCount = 0
for trainH5Name in datasetFilesTrain:
    data = h5py.File(trainH5Name, 'r')
    trainCount += data['rgb'].shape[0]

valCount = 0
for valH5Name in datasetFilesVal:
    data = h5py.File(valH5Name, 'r')
    valCount += data['rgb'].shape[0]

print('trainCount:', trainCount)
print('valCount:', valCount)

