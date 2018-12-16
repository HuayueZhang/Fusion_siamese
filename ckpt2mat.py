import tensorflow as tf
import numpy as np
import scipy.io as sio

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # Create a session
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.log_device_placement = False
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        # load the meta graph and weights
        saver = tf.train.import_meta_graph('/home/zhy/fuse_cnn/model_with_eval/model.ckpt-20.meta')
        # name  = tf.train.latest_checkpoint('/home/zhy/fuse_cnn/model/')
        saver.restore(sess, '/home/zhy/fuse_cnn/model_with_eval/model.ckpt-20')

        # get weights
        graph = tf.get_default_graph()
        weights_1 = sess.run(graph.get_tensor_by_name('siamese/conv1/weights:0'))
        biases_1  = sess.run(graph.get_tensor_by_name('siamese/conv1/biases:0'))
        weights_2 = sess.run(graph.get_tensor_by_name('siamese/conv2/weights:0'))
        biases_2  = sess.run(graph.get_tensor_by_name('siamese/conv2/biases:0'))
        weights_3 = sess.run(graph.get_tensor_by_name('siamese/conv3/weights:0'))
        biases_3  = sess.run(graph.get_tensor_by_name('siamese/conv3/biases:0'))
        weights_output = sess.run(graph.get_tensor_by_name('combine/fc/weights:0'))
        biases_output  = sess.run(graph.get_tensor_by_name('combine/fc/biases:0'))

        # weights_1 = weights_1.reshape((9, 64))
        # weights_2 = weights_2.transpose((2, 0, 1, 3))
        # weights_2 = weights_2.reshape((64, 9, 128))
        # weights_3 = weights_3.transpose((2, 0, 1, 3))
        # weights_3 = weights_3.reshape((128, 9, 256))
        # weights_output = weights_output.reshape((8, 8, 512, 2))
        # weights_output = weights_output.transpose((2, 0, 1, 3))
        # weights_output = weights_output.reshape((512, 64, 2))


        sio.savemat('model.mat', {
            'weights_b1_1': weights_1,
            'biases_b1_1' : biases_1,
            'weights_b1_2': weights_2,
            'biases_b1_2' : biases_2,
            'weights_b1_3': weights_3,
            'biases_b1_3' : biases_3,
            'weights_output': weights_output,
            'biases_output ': biases_output,
        }, format='5')

if __name__ == "__main__":
    main()