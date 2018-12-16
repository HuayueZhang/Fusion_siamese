# -*- coding: utf-8 -*

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

flags = tf.app.flags
FLAGS = flags.FLAGS

def arg_scope(weight_decay=0.0005, data_format='NHWC'):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        padding='SAME',
                        data_format=data_format):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer()) as sc:
            return sc


def siamese_net(input, reuse=False):
    with tf.variable_scope('siamese', 'siamese_net', [input], reuse=reuse):
        net = slim.conv2d(input, 64, [3, 3], scope='conv1')
        net = slim.conv2d(net, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='max_pool')
        net = slim.conv2d(net, 256, [3, 3], scope='conv3')
    return net


def cls_net(in1, in2):
    with tf.variable_scope('combine', 'combine_net', [in1, in2]):
        net = tf.concat([in1, in2], axis=3)
        # net = slim.flatten(net, scope="flatten")  # 全连接之前要flatten
        # net = slim.fully_connected(net, 2, activation_fn=None, scope='fc')
        net = slim.conv2d(net, 2, [8, 8], padding='VALID', activation_fn=None, scope='fc')
        net = tf.squeeze(net)

    return net


def net_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
