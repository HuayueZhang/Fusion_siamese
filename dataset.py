import tensorflow as tf
import utils
import numpy as np
from os.path import splitext, split

flags = tf.app.flags
FLAGS = flags.FLAGS

def batch(blurd_paths):
    blurd_images = utils.get_images(blurd_paths)

    clear_paths = []
    for blurd_path in blurd_paths:
        path1, ext = splitext(blurd_path)
        path2 = path1[:-1]
        path2 = path2 + str(0)
        path3, imgname = split(path2)
        path4, folder = split(path3)
        clear_paths.append(path4+'/clear_x1/'+imgname+ext)
    clear_images = utils.get_images(clear_paths)

    labels = np.random.randint(0, 2, FLAGS.batch_size)

    p1 = np.zeros_like(blurd_images)
    p2 = np.zeros_like(blurd_images)
    for i, label in enumerate(labels):
        p1[i, :, :] = clear_images[i, :, :] * label + blurd_images[i, :, :] * (1-label)
        p2[i, :, :] = clear_images[i, :, :] * (1-label) + blurd_images[i, :, :] * label

    p1 = p1[:, :, :, np.newaxis]
    p2 = p2[:, :, :, np.newaxis]
    return p1, p2, labels