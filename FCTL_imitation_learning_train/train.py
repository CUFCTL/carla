import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
from IPython.display import clear_output
import time
import glob
import os

from conet import Net
from dataset import genData

# image augmentation
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

st = lambda aug: iaa.Sometimes(0.4, aug)
oc = lambda aug: iaa.Sometimes(0.3, aug)
rl = lambda aug: iaa.Sometimes(0.09, aug)
seq = iaa.Sequential([
    rl(iaa.GaussianBlur((0, 1.5))),  # blur images with a sigma between 0 and 1.5
    rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)),  # add gaussian noise to images
    oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)),  # randomly remove up to X% of the pixels
    oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2), per_channel=0.5)),
    # randomly remove up to X% of the pixels
    oc(iaa.Add((-40, 40), per_channel=0.5)),  # change brightness of images (by -X to Y of original value)
    st(iaa.Multiply((0.10, 2.5), per_channel=0.2)),  # change brightness of images (X-Y% of original value)
    rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),  # improve or worsen the contrast
    # rl(iaa.Grayscale((0.0, 1))), # put grayscale
], random_order=True)



# parameters
timeNumberFrames = 1  # 4 # number of frames in each samples
batchSize = 120  # size of batch
valBatchSize = 120  # size of batch for validation set
epochs = 100
trainScratch = True

# Configurations
num_images = 657800
itername = 80000
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


# data dir
datasetDirTrain = './dataset/SeqTrain/'
datasetDirVal = './dataset/SeqVal/'

datasetFilesTrain = glob.glob(datasetDirTrain + '*.h5')
datasetFilesVal = glob.glob(datasetDirVal + '*.h5')


batchListGenTrain = genData(fileNames=datasetFilesTrain, batchSize=batchSize)

batchListGenVal = genData(fileNames=datasetFilesVal, batchSize=batchSize)



with sessGraph.as_default():
    sess = tf.Session(graph=sessGraph, config=config)
    with sess.as_default():

        # build model
        print('Building Net ...')
        netTensors = Net(branchConfig, params, timeNumberFrames, prefSize)
      

        print('Initialize Variables in the Graph ...')
        sess.run(tf.global_variables_initializer())  # initialize variables

        # merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2)
        if not (trainScratch):
            saver.restore(sess, "test/model.ckpt")  # restore trained parameters

        # op to write logs to Tensorboard
        logsPath = './logs'
        modelPath = './test/'
        summary_writer = tf.summary.FileWriter(logsPath, graph=sessGraph)
        print('Start Training process ...')

        steps = 0

        for epoch in range(epochs):
            tStartEpoch = time.time()

            print("  Epoch:", epoch)

            for j in range(int(num_images / batchSize)):  # 5481

                steps += 1

                xs, ys = next(batchListGenTrain)

                # augment images
                xs = seq.augment_images(xs)
                xs = xs.astype(np.float32)
                xs = np.multiply(xs, 1.0/255.0)

                contSolver = netTensors['output']['optimizers']  # solverList[i]
                contLoss = netTensors['output']['losses']  # lossList[i]
                
                train_speed = ys[:, 3].reshape([120, 1])  # Speed
                train_speed = np.multiply(train_speed, 1.0/25.0)  # normalize Speed
               
                feedDict = {netTensors['inputs'][0]: xs, netTensors['inputs'][1]: train_speed,
                             netTensors['dropoutVec']: dropoutVec,
                            netTensors['targets'][0]: ys[:, 3].reshape([120, 1]),
                            netTensors['targets'][1]: ys[:, 0:3]}
                _, loss_value = sess.run([contSolver, contLoss], feed_dict=feedDict)

                # write logs at every iteration
                feedDict = {netTensors['inputs'][0]: xs, netTensors['inputs'][1]: train_speed,
                            netTensors['dropoutVec']: [1] * len(dropoutVec),
                            netTensors['targets'][0]: ys[:, 3].reshape([120, 1]),
                            netTensors['targets'][1]: ys[:, 0:3]}
                summary = merged_summary_op.eval(feed_dict=feedDict)
                summary_writer.add_summary(summary, epoch * num_images / batchSize + j)

                print(' Train::: Epoch: %d, Step: %d, TotalSteps: %d, Loss: %g' % (epoch, j, steps, loss_value))

                if steps % 10 == 0:
                    # clear_output(wait=True)netTensors
                    xs, ys = next(batchListGenVal)

                    xs = xs.astype(np.float32)
                    xs = np.multiply(xs, 1.0 / 255.0)

                    val_speed = ys[:, 3].reshape([120, 1])
                    val_speed = np.multiply(val_speed, 1.0 / 25.0)  # normalize Speed

                    contLoss = netTensors['output']['losses']
                    feedDict = {netTensors['inputs'][0]: xs, netTensors['inputs'][1]: val_speed,
                                netTensors['dropoutVec']: [1] * len(dropoutVec),
                                netTensors['targets'][0]: ys[:, 3].reshape([120, 1]),
                                netTensors['targets'][1]: ys[:, 0:3]}
                    loss_value = contLoss.eval(feed_dict=feedDict)
                    print("  Val::: Epoch: %d, Step: %d, TotalSteps: %d, Loss: %g" % (epoch, j, steps, loss_value))


                if steps % 10 == 0:
                    clear_output(wait=True)

                if steps % 50 == 0 and steps != 0:  # batchSize
                    print(j % 50, '  Save Checkpoint ...')
                    if not os.path.exists(modelPath):
                        os.makedirs(modelPath)
                    checkpoint_path = os.path.join(modelPath, "model.ckpt")
                    filename = saver.save(sess, checkpoint_path)
                    print("  Model saved in file: %s" % filename)


                if steps % itername == 0 and steps != 0:
                    # finish the training
                    break

            # finish all saved the models
            if steps % itername == 0 and steps != 0:
                # finish the training
                print('Finalize the training and Save Checkpoint ...')
                if not os.path.exists(modelPath):
                    os.makedirs(modelPath)
                checkpoint_path = os.path.join(modelPath, "model.ckpt")
                filename = saver.save(sess, checkpoint_path)
                print("  Model saved in file: %s" % filename)
                break

        tStopEpoch = time.time()
        print("  Epoch Time Cost:", round(tStopEpoch - tStartEpoch, 2), "s")
