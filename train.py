# -*- coding: utf-8 -*

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

import dataset
import model
import utils

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_integer('patch_size', 16, 'Patch size')
tf.app.flags.DEFINE_integer('train_iter', 5000, 'Total training iter')
tf.app.flags.DEFINE_integer('step', 500, 'Save after ... iteration')
tf.app.flags.DEFINE_integer('gpu_idx', 0, 'Gpu index')
tf.app.flags.DEFINE_float('weight_decay', 0.0005, 'Weight decay')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
tf.app.flags.DEFINE_float('momentum', 0.99, 'Momentum')
tf.app.flags.DEFINE_string('checkpoint_dir', './model_with_eval', 'Checkpoint directory')
FLAGS = tf.app.flags.FLAGS

ckpt_dir = FLAGS.checkpoint_dir
DATA_FORMAT = 'NHWC'
DATA_DIRECTORY = '/home/zhy/fuse_cnn/ILSVRC2012/train/blurd_x5/'
DATA_PATHS = utils.list_images(DATA_DIRECTORY)
n_batches = int(len(DATA_PATHS) // FLAGS.batch_size)

EVAL_DATA_DIRECTORY = '/home/zhy/fuse_cnn/ILSVRC2012/train/blurd_x5/'
EVAL_DATA_PATHS = utils.list_images(DATA_DIRECTORY)
eval_n_batches = int(len(DATA_PATHS) // FLAGS.batch_size)

gpus = '/gpu:%d'%(FLAGS.gpu_idx)
cpus = '/cpu:0'

def main(_):

    import setproctitle
    setproctitle.setproctitle('python_train')

    is_training = tf.placeholder(tf.bool, shape=())
    left_in = tf.placeholder(tf.float32, [None, FLAGS.patch_size, FLAGS.patch_size, 1], name='left')
    righ_in = tf.placeholder(tf.float32, [None, FLAGS.patch_size, FLAGS.patch_size, 1], name='right')
    with tf.name_scope('similarity'):
        label = tf.placeholder(tf.int32, [None,], name='label')

    arg_scope = model.arg_scope(weight_decay=FLAGS.weight_decay,
                                data_format=DATA_FORMAT)
    global_steps = tf.Variable(0, trainable=False)

    with tf.device(gpus):
        with tf.name_scope('model'):
            with slim.arg_scope(arg_scope):
                left_out = model.siamese_net(left_in, reuse=False)
                righ_out = model.siamese_net(righ_in, reuse=True)
                cls_out = model.cls_net(left_out, righ_out)

        loss = model.net_loss(cls_out, label)
        tf.summary.scalar('loss', loss)

        lr = FLAGS.learning_rate
        mm = FLAGS.momentum
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=mm, use_nesterov=True)
        train_op = optimizer .minimize(loss=loss, global_step=global_steps)
        # 损失函数优化器的minimize()中global_step=global_steps能够提供global_step自动+1的操作
        # global_steps是所有epoch的steps全都累加起来

    variables_to_train = tf.trainable_variables()
    for var in variables_to_train:
        tf.summary.histogram(var.op.name, var)

    saver = tf.train.Saver(max_to_keep = 500)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.log_device_placement = False
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Add summary writers
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('log/train', sess.graph)
    eval_writer = tf.summary.FileWriter('log/eval')

    # Init variables
    if not os.listdir(ckpt_dir):  # 文件夹为空
        init = tf.global_variables_initializer()
        # Run the init operation
        sess.run(init, {is_training: True})
    else:
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

    # This is just a dictionary including important parameters transferred to another function
    # These parameters are actually all parts, namely operations, of the graph, without being
    ops = {'is_training_pl': is_training,
           'left_in_pl': left_in,
           'righ_in_pl': righ_in,
           'label_pl': label,
           'loss': loss,
           'train_op': train_op,
           'merged': merged,
           'global_steps': global_steps}

    # train iter
    for i in range(FLAGS.train_iter):
        np.random.shuffle(DATA_PATHS)
        np.random.shuffle(EVAL_DATA_PATHS)

        for b in range(n_batches):
            step, train_loss = train_one_step(b, sess, ops, train_writer, is_training = True)
            eval_loss = eval_one_step(b, sess, ops, eval_writer, is_training = False)
            print "global step %d: train loss = %f; eval loss = %f" % (step, train_loss, eval_loss)

            saver.save(sess, "model_with_eval/model.ckpt", global_steps)


def train_one_step(b, sess, ops, writer, is_training):
    # An iteration/step
    # Get train data
    batch_paths = DATA_PATHS[b * FLAGS.batch_size: (b * FLAGS.batch_size + FLAGS.batch_size)]
    b_left_in, b_righ_in, b_label = dataset.batch(batch_paths)
    feed_dict = {ops['is_training_pl']: is_training,
                 ops['left_in_pl']: b_left_in,
                 ops['righ_in_pl']: b_righ_in,
                 ops['label_pl']: b_label}

    # run the training step
    summary_str, _, loss_rslt, global_steps = sess.run([ops['merged'], ops['train_op'], ops['loss'], ops['global_steps']],
                                         feed_dict = feed_dict)

    writer.add_summary(summary_str, global_steps)
    # 使得在tensorboard中，显示summary随global_steps的变化情况
    # 如果使用writer.add_summary(summary，global_step)时没有传global_step参数,会使scarlar_summary变成一条直线
    return global_steps, loss_rslt

def eval_one_step(b, sess, ops, writer, is_training):
    # An iteration/step
    # Get eval data
    batch_paths = EVAL_DATA_PATHS[b * FLAGS.batch_size: (b * FLAGS.batch_size + FLAGS.batch_size)]
    b_left_in, b_righ_in, b_label = dataset.batch(batch_paths)
    feed_dict = {ops['is_training_pl']: is_training,
                 ops['left_in_pl']: b_left_in,
                 ops['righ_in_pl']: b_righ_in,
                 ops['label_pl']: b_label}

    # Pass through the network without running the training step
    summary_str, loss_rslt, global_steps = sess.run([ops['merged'], ops['loss'], ops['global_steps']],
                                         feed_dict=feed_dict)
    writer.add_summary(summary_str, global_steps)
    return loss_rslt


if __name__ == '__main__':
    tf.app.run()