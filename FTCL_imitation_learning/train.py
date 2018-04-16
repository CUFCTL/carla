import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
from IPython.display import clear_output
import time
import glob


from conet import Net
from dataset import genData,genBranch

# image augmentation
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

st = lambda aug: iaa.Sometimes(0.4, aug)
oc = lambda aug: iaa.Sometimes(0.3, aug)
rl = lambda aug: iaa.Sometimes(0.09, aug)
seq = iaa.Sequential([
        rl(iaa.GaussianBlur((0, 1.5))), # blur images with a sigma between 0 and 1.5
        rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)), # add gaussian noise to images
        oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)), # randomly remove up to X% of the pixels
        oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2),per_channel=0.5)), # randomly remove up to X% of the pixels
        oc(iaa.Add((-40, 40), per_channel=0.5)), # change brightness of images (by -X to Y of original value)
        st(iaa.Multiply((0.10, 2.5), per_channel=0.2)), # change brightness of images (X-Y% of original value)
        rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)), # improve or worsen the contrast
        #rl(iaa.Grayscale((0.0, 1))), # put grayscale
], random_order=True)



# parameters
timeNumberFrames = 1 #4 # number of frames in each samples
batchSize = 120 # size of batch
valBatchSize = 120 # size of batch for validation set
NseqVal = 5  # number of sequences to use for validation
# training parameters
epochs = 100
samplesPerEpoch = 500
L2NormConst = 0.001 
trainScratch = True

# Configurations
num_images = 657800 # 200 * 3289
memory_fraction=0.25
image_cut=[115, 510]
dropoutVec = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5
prefSize = _image_size = (88, 200, 3)
learningRate = 0.0002 # multiplied by 0.5 every 50000 mini batch
iterNum = 294000
beta1 = 0.7
beta2 = 0.85
controlInputs = [2,5,3,4] # Control signal, int ( 2 Follow lane, 3 Left, 4 Right, 5 Straight)
cBranchesOutList = ['Follow Lane','Go Left','Go Right','Go Straight','Speed Prediction Branch'] 

branchConfig = [["Steer", "Gas", "Brake"],["Speed"]]
params = [trainScratch, dropoutVec, image_cut, learningRate, beta1, beta2, num_images, iterNum, batchSize, valBatchSize, NseqVal, epochs, samplesPerEpoch, L2NormConst]

# GPU configuration
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.visible_device_list = '0'
config.gpu_options.per_process_gpu_memory_fraction = memory_fraction



# use many gpus
config = tf.ConfigProto(allow_soft_placement = True)

tf.reset_default_graph()
sessGraph = tf.Graph()


# Prepare data generators
batchListGenTrain = []

batchListGenVal = []

batchListName = []

# data dir
datasetDirTrain = './SeqTrain/'
datasetDirVal = './SeqVal/'

datasetFilesTrain = glob.glob(datasetDirTrain+'*.h5')
datasetFilesVal = glob.glob(datasetDirVal+'*.h5')



for i in range(len(branchConfig)):

    with tf.name_scope("Branch_" + str(i)):


        if branchConfig[i][0] == "Speed":
            miniBatchGen = genData(fileNames = datasetFilesTrain, batchSize = batchSize)
            batchListGenTrain.append(miniBatchGen)
            miniBatchGen = genData(fileNames = datasetFilesVal, batchSize = batchSize)
            batchListGenVal.append(miniBatchGen)
        else:
            # controlInputs = [2,5,3,4] # Control signal, int ( 2 Follow lane, 3 Left, 4 Right, 5 Straight)
            miniBatchGen = genBranch(fileNames = datasetFilesTrain, branchNum = 2, batchSize = batchSize)
            batchListGenTrain.append(miniBatchGen)
            miniBatchGen = genBranch(fileNames = datasetFilesVal, branchNum = 2, batchSize = batchSize)
            batchListGenVal.append(miniBatchGen)   


print(next(batchListGenTrain[0]))
#print(len(batchListGenTrain[1]))
#print(len(batchListGenVal[0]))
exit()



with sessGraph.as_default():
    sess = tf.Session(graph=sessGraph, config = config)
    with sess.as_default():
        
        # build model
        print('Building Net ...')
        netTensors = Net(branchConfig, params, timeNumberFrames, prefSize)
        #[ inputs['inputImages','inputData'], 
        #  targets['targetSpeed', 'targetController'],  
        #  'params', 
        #   dropoutVec, 
        #   output[optimizers, losses, branchesOutputs] 
        # ]
        
        #print(netTensors['output'])


        print('Initialize Variables in the Graph ...')
        sess.run(tf.global_variables_initializer()) # initialize variables
        
        # merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V2)
        if not(trainScratch):
            saver.restore(sess, "test/model.ckpt") # restore trained parameters

        # op to write logs to Tensorboard
        logsPath = './logs'
        modelPath = './test/'
        summary_writer = tf.summary.FileWriter(logsPath, graph=sessGraph)
        print('Start Training process ...')
        

        steps = 0

        for epoch in range(epochs):
            tStartEpoch = time.time()

            print("  Epoch:", epoch)

            for j in range(int(num_images/batchSize)): #5481

                steps += 1
                
                for i in range(0, len(branchConfig)):

                        xs , ys = next(batchListGenTrain[i])
                        
                        # augment images
                        xs = seq.augment_images(xs)
                        
                        contSolver = netTensors['output']['optimizers'][i]#solverList[i]
                        contLoss = netTensors['output']['losses'][i]#lossList[i]

                        inputData = []
                        inputData.append(sess.run(tf.one_hot(ys[:,24],4))) # Command Control
                        inputData.append(ys[:,10].reshape([120,1])) # Speed 
                        
                        # [ inputs['inputImages','inputData'], targets['targetSpeed', 'targetController'],  'params', dropoutVec', output[optimizers, losses, branchesOutputs] ]
                        feedDict = {netTensors['inputs'][0]: xs, netTensors['inputs'][1][0]: inputData[0], netTensors['inputs'][1][1]: inputData[1], netTensors['dropoutVec']: dropoutVec, netTensors['targets'][0]: ys[:,10].reshape([120,1]), netTensors['targets'][1]: ys[:,0:3]}                       
                        _,loss_value = sess.run([contSolver, contLoss], feed_dict = feedDict)
                        
                        # write logs at every iteration
                        feedDict = {netTensors['inputs'][0]: xs, netTensors['inputs'][1][0]: inputData[0], netTensors['inputs'][1][1]: inputData[1], netTensors['dropoutVec']: [1] * len(dropoutVec), netTensors['targets'][0]: ys[:,10].reshape([120,1]), netTensors['targets'][1]: ys[:,0:3]}                       
                        summary = merged_summary_op.eval(feed_dict=feedDict)
                        summary_writer.add_summary(summary, epoch * num_images/batchSize + j)

                        print("  Train::: Epoch: %d, Step: %d, TotalSteps: %d, Loss: %g" % (epoch, epoch * batchSize + j, steps, loss_value), cBranchesOutList[i])
                        
                        if steps % 10 == 0:
                            #clear_output(wait=True)netTensors
                            xs, ys = next(batchListGenVal[i])
                            contLoss = netTensors['output']['losses'][i]
                            feedDict = {netTensors['inputs'][0]: xs, netTensors['inputs'][1][0]: inputData[0], netTensors['inputs'][1][1]: inputData[1], netTensors['dropoutVec']: [1] * len(dropoutVec), netTensors['targets'][0]: ys[:,10].reshape([120,1]), netTensors['targets'][1]: ys[:,0:3]}                       
                            loss_value = contLoss.eval(feed_dict=feedDict)
                            print("  Val::: Epoch: %d, Step: %d, TotalSteps: %d, Loss: %g" % (epoch, epoch * batchSize + j, steps, loss_value), cBranchesOutList[i])
                            
                if steps % 10 == 0:
                    clear_output(wait = True)
                     

                if steps % 50 == 0 and steps!=0: # batchSize
                    print(j%50, '  Save Checkpoint ...')
                    if not os.path.exists(modelPath):
                        os.makedirs(modelPath)
                    checkpoint_path = os.path.join(modelPath, "model.ckpt")
                    filename = saver.save(sess, checkpoint_path)
                    print("  Model saved in file: %s" % filename)

                if steps % 50000 == 0 and steps!=0: # every 50000 step, multiply learning rate by half
                    print("Half the learning rate ....")
                    solverList = []
                    lossList = []
                    trainVars = tf.trainable_variables()
                    for i in range(0, len(branchConfig)):
                        with tf.name_scope("Branch_" + str(i)):
                            if branchConfig[i][0] == "Speed":
                                # we only use the image as input to speed prediction
                                #if not (j == 0):
                                # [ inputs['inputImages','inputData'], targets['targetSpeed', 'targetController'],  'params', dropoutVec', output[optimizers, losses, branchesOutputs] ]
                                              
                                params[3] = params[3] * 0.5 # update Learning Rate
                                contLoss = tf.reduce_mean(tf.square(tf.subtract(netTensors['output']['branchesOutputs'][-1], netTensors['targets'][0]))) #+ tf.add_n([tf.nn.l2_loss(v) for v in trainVars]) * L2NormConst
                                contSolver = tf.train.AdamOptimizer(learning_rate=params[3], beta1=params[4], beta2=params[5]).minimize(contLoss)
                                solverList.append(contSolver)
                                lossList.append(contLoss)
                                # create a summary to monitor cost tensor
                                tf.summary.scalar("Speed_Loss", contLoss)
                            else:
                                #if not (j == 0):
                                params[3] = params[3] * 0.5
                                contLoss = tf.reduce_mean(tf.square(tf.subtract(netTensors['output']['branchesOutputs'][i], netTensors['targets'][1]))) #+ tf.add_n([tf.nn.l2_loss(v) for v in trainVars]) * L2NormConst
                                contSolver = tf.train.AdamOptimizer(learning_rate=params[3], beta1=params[4], beta2=params[5]).minimize(contLoss)
                                solverList.append(contSolver)
                                lossList.append(contLoss)  
                                tf.summary.scalar("Control_Loss_Branch_"+str(i), contLoss)
                                
                    # update new Losses and Optimizers 
                    print('Initialize Variables in the Graph ...')
                    # merge all summaries into a single op
                    merged_summary_op = tf.summary.merge_all()
                    sess.run(tf.global_variables_initializer())
                    saver.restore(sess, "test/model.ckpt") # restore trained parameters

                if steps % 294000 == 0 and steps!=0:
                    # finish the training
                    break



            # finish all saved the models  
            if steps % 294000 == 0 and steps!=0:
                # finish the training
                print('Finalize the training and Save Checkpoint ...')
                if not os.path.exists(modelPath):
                    os.makedirs(modelPath)
                checkpoint_path = os.path.join(modelPath, "model.ckpt")
                filename = saver.save(sess, checkpoint_path)
                print("  Model saved in file: %s" % filename)
                break

            tStopEpoch = time.time()
            print "  Epoch Time Cost:", round(tStopEpoch - tStartEpoch,2), "s"









