import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import glob
import h5py

from conet import Net
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)

matplotlib.use('Agg')


def plot_embedding(X, y, title=None):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)

    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(X_tsne.shape[0]):
        plt.text(X_tsne[i, 0], X_tsne[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

    plt.savefig(title)
    plt.close(fig)


# parameters
timeNumberFrames = 1  # 4 # number of frames in each samples

trainScratch = True

datasetDirVal = './dataset/SeqVal/'
datasetFilesVal = glob.glob(datasetDirVal + '*.h5')

memory_fraction = 0.25
image_cut = [115, 510]
dropoutVec = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 2
prefSize = _image_size = (88, 200, 3)
learningRate = 0.0002
beta1 = 0.7
beta2 = 0.85
branchConfig = [["Steer", "Gas", "Brake"], ["Speed"]]
params = [trainScratch, dropoutVec, image_cut, learningRate, beta1, beta2]

# GPU configuration
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.visible_device_list = '0'
config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
# use many gpus
config = tf.ConfigProto(allow_soft_placement=True)

tf.reset_default_graph()
sessGraph = tf.Graph()

# extract cnn features
sess = tf.Session(config=config)

# build model
print('Building Net ...')
netTensors = Net(branchConfig, params, timeNumberFrames, prefSize)

print('Initialize Variables in the Graph ...')
sess.run(tf.global_variables_initializer())  # initialize variables

saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2)
saver.restore(sess, 'test/model.ckpt')  # restore trained parameters

features = np.empty((0, 512), dtype=np.float32)

for h5file in datasetFilesVal:
    h5data = h5py.File(h5file, 'r')
    count = h5data['rgb'].shape[0]
    xs = h5data['rgb'][:count]
    xs = xs.astype(np.float32)
    xs = np.multiply(xs, 1.0 / 255.0)
    feedDict = {netTensors['inputs'][0]: xs, netTensors['dropoutVec']: [1] * len(dropoutVec)}
    feature = sess.run(netTensors['output']['features'], feedDict)
    features = np.concatenate((features, feature), axis=0)

lables = np.ones([features.shape[0]], dtype=np.float32) * 0.5

plot_embedding(features, lables, "Test")
