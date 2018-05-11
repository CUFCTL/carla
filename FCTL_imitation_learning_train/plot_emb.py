import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import h5py

from conet import Net
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)
import fnmatch
import os

matplotlib.use('Agg')


def build_batch(file_lists, number):
    img_array = np.empty((0, 88, 200, 3), dtype=np.uint8)
    for _ in range(number):
        id_file = np.random.randint(len(file_lists) - 1)
        with h5py.File(file_lists[id_file], 'r') as h5read:
            id_img = np.random.randint(h5read['rgb'].shape[0] - 1)
            img = np.expand_dims(h5read['rgb'][id_img], axis=0)
            img_array = np.concatenate((img_array, img), axis=0)

    return img_array


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


genh5 = True
save_image = True
timeNumberFrames = 1
trainScratch = True
town_name = 'Town01'
weatherID = '56'
batch_size = 800
data_size = 50000

datasetDir = [
    '../dataset/RawData/' + town_name + '_' + weatherID + '/' + town_name + '_CL/',
    '../dataset/RawData/' + town_name + '_' + weatherID + '/' + town_name + '_SL/'
]
model_path = '../dataset/Models/' + town_name + '_' + weatherID + '/train_CSL/'
gen_features_name = town_name + '_' + weatherID + '_CSL_img_features.h5'
tsne_title = town_name + '_' + weatherID + '_CSL_1k'

datasetFilesVal = []
for dir_path in datasetDir:
    for root, dirnames, filenames in os.walk(dir_path):
        for filename in fnmatch.filter(filenames, '*.h5'):
            datasetFilesVal.append(os.path.join(root, filename))

count = 0
for h5file in datasetFilesVal:
    with h5py.File(h5file, 'r') as h5data:
        count += h5data['rgb'].shape[0]
print("Total image:", count)

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
saver.restore(sess, model_path + 'test/model.ckpt')  # restore trained parameters

if genh5:
    with h5py.File(gen_features_name, 'w') as h5write:
        if save_image:
            h5write.create_dataset('rgb', shape=(0, 88, 200, 3), dtype="uint8", maxshape=(None, 88, 200, 3), compression="gzip")
        h5write.create_dataset('feature', shape=(0, 512), dtype="float32", maxshape=(None, 512), compression="gzip")
else:
    features = np.empty((0, 512), dtype=np.float32)

if data_size % batch_size == 0:
    iter_list = [batch_size] * int(data_size / batch_size)
else:
    iter_list = [batch_size] * int(data_size / batch_size) + [data_size % batch_size]

old_prog = 0
progress = 0
for size in iter_list:
    imgs = build_batch(datasetFilesVal, size)
    xs = np.multiply(imgs.astype(np.float32), 1.0 / 255.0)
    feedDict = {netTensors['inputs'][0]: xs, netTensors['dropoutVec']: [1] * len(dropoutVec)}
    feature = sess.run(netTensors['output']['features'], feedDict)
    if genh5:
        with h5py.File(gen_features_name, 'a') as h5write:
            if save_image:
                h5write['rgb'].resize(h5write['rgb'].shape[0] + imgs.shape[0], axis=0)
                h5write['rgb'][-imgs.shape[0]:] = imgs
            h5write['feature'].resize(h5write['feature'].shape[0] + feature.shape[0], axis=0)
            h5write['feature'][-imgs.shape[0]:] = feature
    else:
        features = np.concatenate((features, feature), axis=0)

    progress += size
    if progress != old_prog:
        print "Progress:", int(float(progress)/data_size * 100)
        old_prog = progress


if not genh5:
    # lables1 = np.ones([321], dtype=np.float32) * 1
    # lables2 = np.ones([100], dtype=np.float32) * 2
    # lables3 = np.ones([414], dtype=np.float32) * 3
    # lables  = np.concatenate((lables1, lables2, lables3), axis=0)

    lables = np.ones([features.shape[0]], dtype=np.int) * 1

    plot_embedding(features, lables, tsne_title)
