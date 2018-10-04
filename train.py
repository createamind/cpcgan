from random import randint
import numpy as np
from scipy import stats
import tensorflow as tf

from utils.debug_tools import timeit
from utils import utils
from cpc import CPC
from data_utils import SortedNumberGenerator

def train_model(epochs, batch_size, lr=1e-3):
    
    terms = 4
    predict_terms = 4
    image_size = 64
    color = True
    sample_batch_size = batch_size // 4
    # Prepare data
    train_data = SortedNumberGenerator(batch_size=sample_batch_size, subset='train', terms=terms,
                                       positive_samples=sample_batch_size // 2, predict_terms=predict_terms,
                                       image_size=image_size, color=color, rescale=False)

    validation_data = SortedNumberGenerator(batch_size=batch_size, subset='valid', terms=terms,
                                            positive_samples=batch_size // 2, predict_terms=predict_terms,
                                            image_size=image_size, color=color, rescale=False)

    name = 'cpc'
    cpc_args = {
        'code_size': 128,
        'learning_rate': lr,
        'weight_decay': 1e-3,
        'batch_size': batch_size,
    }
    args = {name: cpc_args, 'model_name': 'cpc-supervised'}

    sess = tf.Session()
    cpc = CPC(name, args, reuse=False, sess=sess)

    for epoch in range(epochs):
        for (history_sample, future_sample), label_sample_batch in train_data:
            history, future, label = data_augmention(history_sample, future_sample, label_sample_batch)
            train_batch(cpc, history, future, label, 'Training')
        print()

        for (history, future), label in validation_data:
            train_batch(cpc, history, future, label, 'Testing')
        print()
        cpc.save()
        print('Model Saved')

def data_augmention(x_sample, y_sample, label_sample):
    cval = np.median([x_sample, y_sample])
    
    x_batch = [x_sample]
    y_batch = [y_sample]
    label_batch = [label_sample]

    # rotations
    pos_angle = np.random.randint(1, 15)
    x_batch.append(utils.rotate_image(x_sample, pos_angle, cval=cval))
    y_batch.append(utils.rotate_image(y_sample, pos_angle, cval=cval))
    label_batch.append(label_sample)

    neg_angle = np.random.randint(-15, -1)
    x_batch.append(utils.rotate_image(x_sample, neg_angle, cval=cval))
    y_batch.append(utils.rotate_image(y_sample, neg_angle, cval=cval))
    label_batch.append(label_sample)

    # shift
    shift = [0, 0, randint(-5, 5), randint(-5, 5), 0]
    x_batch.append(utils.shift_image(x_sample, shift, cval=cval))
    y_batch.append(utils.shift_image(y_sample, shift, cval=cval))
    label_batch.append(label_sample)

    x_batch = np.concatenate(x_batch)
    y_batch = np.concatenate(y_batch)
    label_batch = np.concatenate(label_batch)

    return x_batch, y_batch, label_batch

def train_batch(cpc, history, future, label, dataset):
    feed_dict = {
        cpc.x_history: history,
        cpc.x_future: future,
        cpc.y: label
    }

    if dataset == 'Training':
        training_step, logits, loss, _, summary = cpc.sess.run([cpc.global_step, cpc.logits, cpc.loss, cpc.opt_op, cpc.merged_op], feed_dict=feed_dict)
        cpc.writer.add_summary(summary, training_step)
    else:
        logits, loss = cpc.sess.run([cpc.logits, cpc.loss], feed_dict=feed_dict)

    prob = 1 / (1 + np.exp(-logits))
    accuracy = np.sum((prob > .5) == label) / np.size(label)
    print('\r{}:\tLoss {: .4f}\tAccuracy {:.2f}'.format(dataset, loss, accuracy), end="")

if __name__ == "__main__":

    train_model(
        epochs=10,
        batch_size=32,
    )