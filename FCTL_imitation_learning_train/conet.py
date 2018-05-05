# from __future__ import print_function
import tensorflow as tf
import keras

from keras.layers import ConvLSTM2D, MaxPool3D, BatchNormalization, MaxPool2D
from tensorflow.contrib.layers import batch_norm
from network import load_imitation_learning_network


def controlNet(inputs, targets, shape, dropoutVec, branchConfig, params, scopeName,scopeName1,scopeName2):
    """
        Get one image/sequence of images to predict control operations for controling the vehicle
        inputs: N Batch of M images in order
        shape: [BatchSize, SeqSize, FrameHeight, FrameWeight, Channels]
        phase: placeholder for training
        scopeName: TensorFlow Scope Name to separate nets in the graph
    """
    with tf.variable_scope(scopeName) as scope:
        # with tf.name_scope("Network"):

        networkTensor = load_imitation_learning_network(inputs[0], inputs[1],
                                                        shape[1:3], dropoutVec,scopeName1,scopeName2)

        trainVars = tf.trainable_variables()

        for i in range(0, len(branchConfig)):
            with tf.name_scope("Branch_" + str(i)):
                if branchConfig[i][0] == "Speed":
                    # we only use the image as input to speed prediction
                    SpeedLoss = tf.reduce_mean(tf.square(tf.subtract(networkTensor[1], targets[0]))) 
                    # create a summary to monitor cost tensor
                    tf.summary.scalar("Speed_Loss", SpeedLoss)
                else:
                    steer_loss = tf.reduce_mean(tf.square(tf.subtract(networkTensor[0][:,0], targets[1][:,0])))
                    gas_loss = tf.reduce_mean(tf.square(tf.subtract(networkTensor[0][:,1], targets[1][:,1])))
                    brake_loss = tf.reduce_mean(tf.square(tf.subtract(networkTensor[0][:,2], targets[1][:,2])))

                    CtrLoss = 0.45 * steer_loss + 0.45 * gas_loss + 0.05 * brake_loss
                    tf.summary.scalar("Control_Loss_Branch_" + str(i), CtrLoss)
                    

        loss = 0.05 * SpeedLoss + 0.95 * CtrLoss
        tf.summary.scalar('total loss', loss)
        contSolver = tf.train.AdamOptimizer(learning_rate=params[3], beta1=params[4],
                                            beta2=params[5]).minimize(loss)

                    
        tensors = {
            'optimizers': contSolver,
            'losses': loss,
            'output': networkTensor
        }
    return tensors


def Net(branchConfig, params, timeNumberFrames, prefSize=(128, 160, 3)):
    shapeInput = [None, prefSize[0], prefSize[1], prefSize[2]]
    
    inputImages = tf.placeholder("float", shape=[None, prefSize[0], prefSize[1], prefSize[2]], name="input_image")
    
    inputData = tf.placeholder(tf.float32, shape=[None, 1], name="input_speed")

    inputs = [inputImages, inputData]
    
    dout = tf.placeholder("float", shape=[len(params[1])])

    targetSpeed = tf.placeholder(tf.float32, shape=[None, 1], name="target_speed")
    
    targetController = tf.placeholder(tf.float32, shape=[None, 3], name="target_control")

    targets = [targetSpeed, targetController]

    print('Building ControlNet ...')
    controlOpTensors = controlNet(inputs, targets, shapeInput, dout, branchConfig, params, scopeName='NET',scopeName1='First',scopeName2='Second')

    tensors = {
        'inputs': inputs,
        'targets': targets,
        'params': params,
        'dropoutVec': dout,
        'output': controlOpTensors
    }
    return tensors  