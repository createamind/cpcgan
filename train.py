from random import randint
import numpy as np
from scipy import stats
import tensorflow as tf

from utils.debug_tools import timeit
from utils import utils
from cpc import CPC
from data_utils import SortedNumberGenerator

""" Commented lines correspond to data augmentation, which takes so much time that I have to leave it behind """


def train_model(epochs, batch_size=32, learning_rate=1e-3, weight_decay=1e-4, code_size=128, model_name='supervised'):
    terms = 4
    predict_terms = 4
    image_size = 64
    color = True
    # sample_batch_size = batch_size // 4
    # Prepare data
    # train_data = SortedNumberGenerator(batch_size=sample_batch_size, subset='train', terms=terms,
    #    positive_samples=sample_batch_size // 2, predict_terms=predict_terms,
    #    image_size=image_size, color=color, rescale=False)

    train_data = SortedNumberGenerator(batch_size=batch_size, subset='train', terms=terms,
                                       positive_samples=batch_size // 2, predict_terms=predict_terms,
                                       image_size=image_size, color=color, rescale=False)

    validation_data = SortedNumberGenerator(batch_size=batch_size, subset='valid', terms=terms,
                                            positive_samples=batch_size // 2, predict_terms=predict_terms,
                                            image_size=image_size, color=color, rescale=False)

    name = 'cpc'
    cpc_args = {
        'code_size': code_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'model_name': model_name
    }

    sess = tf.Session()
    cpc = CPC(name, cpc_args, reuse=False, sess=sess, loss_type=model_name)

    cpc.restore()
    i = 0
    for _ in range(epochs):
        # for i, ((history_sample, future_sample), label_sample_batch) in enumerate(train_data):
            # history, future, label = data_augmention(history_sample, future_sample, label_sample_batch)
            # run_batch(cpc, history, future, label, 'Training', i)
        losses = []
        accuracies = [] if model_name == 'supervised' else None
        # Training
        cpc.log_tensorboard = True
        for (history, future), label in train_data:
            run_batch(cpc, history, future, label,
                      'Training', i, losses, accuracies)
            i += 1
            if i % 1e3 == 0:
                break
        print('\nTraining Epoch Is Done. \nStart Validation...')

        losses = []
        accuracies = [] if model_name == 'supervised' else None
        # Validation
        cpc.log_tensorboard = False
        for j, ((history, future), label) in enumerate(validation_data):
            run_batch(cpc, history, future, label,
                      'Testing', j, losses, accuracies)
            if j > 3e2:
                break
        print('\nValidation Epoch Is Done.\n Start Saving...')
        cpc.save()
        print('Model Saved')

# def data_augmention(x_sample, y_sample, label_sample):
#     cval = np.median([x_sample, y_sample])

#     x_batch = [x_sample]
#     y_batch = [y_sample]
#     label_batch = [label_sample]

#     # rotations
#     pos_angle = np.random.randint(1, 15)
#     x_batch.append(utils.rotate_image(x_sample, pos_angle, cval=cval))
#     y_batch.append(utils.rotate_image(y_sample, pos_angle, cval=cval))
#     label_batch.append(label_sample)

#     neg_angle = np.random.randint(-15, -1)
#     x_batch.append(utils.rotate_image(x_sample, neg_angle, cval=cval))
#     y_batch.append(utils.rotate_image(y_sample, neg_angle, cval=cval))
#     label_batch.append(label_sample)

#     # shift
#     shift = [0, 0, randint(-5, 5), randint(-5, 5), 0]
#     x_batch.append(utils.shift_image(x_sample, shift, cval=cval))
#     y_batch.append(utils.shift_image(y_sample, shift, cval=cval))
#     label_batch.append(label_sample)

#     x_batch = np.concatenate(x_batch)
#     y_batch = np.concatenate(y_batch)
#     label_batch = np.concatenate(label_batch)

#     return x_batch, y_batch, label_batch


def run_batch(cpc, history, future, label, dataset, i, losses, accuracies=None):
    feed_dict = {
        cpc.x_history: history,
        cpc.x_future: future,
        cpc.is_training: True if dataset == 'Training' else False,
        cpc.label: label
    }

    if accuracies is not None:  # supervised learning, in which we obtain logits for accuracy computation
        if dataset == 'Training':
            logits, loss, _, summary = cpc.sess.run(
                [cpc.logits, cpc.loss, cpc.opt_op, cpc.merged_op], feed_dict=feed_dict)
            cpc.writer.add_summary(summary, i)
        else:
            logits, loss = cpc.sess.run(
                [cpc.logits, cpc.loss], feed_dict=feed_dict)

        losses.append(loss)
        prob = 1 / (1 + np.exp(-logits))
        accuracy = np.sum((prob > .5) == label) / np.size(label)
        accuracies.append(accuracy)
        print('\r{} step {: 4d}:\tLoss {: .4f}\tAccuracy {:.2f}\t \
            Average Loss: {:.4f}\t Average Accuracy: {:.2f}'.format(dataset, int(i % 1e3), loss, accuracy, np.mean(losses), np.mean(accuracies)), end="")
    else:                       # unsupervised learning, in which we don't have logits
        if dataset == 'Training':
            loss, _, summary = cpc.sess.run(
                [cpc.loss, cpc.opt_op, cpc.merged_op], feed_dict=feed_dict)
            cpc.writer.add_summary(summary, i)
        else:
            loss = cpc.sess.run(cpc.loss, feed_dict=feed_dict)

        losses.append(loss)
        print('\r{} step {: 4d}:\tLoss {: .4f}\tAverage Loss {:.4f}'.format(
            dataset, int(i % 1e3), loss, np.mean(losses)), end="")


if __name__ == "__main__":

    train_model(
        epochs=10, model_name='dim'
    )
