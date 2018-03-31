# Created on May 31 2018
# Author: CRSilkworth
"""Training operation function definitions for Alex Net aside outside of the data pipeline and model definition."""
import tensorflow as tf
import numpy as np


def loss(logits, labels, num_classes=1000, weight_decay=0.0, to_run=None):
    """
    The alex net loss definition. Basically just cross entropy with a weight decay.

    Args:
        logits: The logits operation outputted by the model.
        labels: The image labels.
        num_classes: The number of possible labels.
        to_run: a dictionary which can be passed to pick up any ops inside the function that need to be fed to sess.run. Useful for debugging.
    Return:
        loss: a scalar tensor of he cross entropy + weight norm
    """
    with tf.name_scope('loss'):
        # Pull out all the trainiable weights excluding the biases.
        weights = [w for w in tf.trainable_variables() if not w.name.find('biases') > -1]

        # Find their l2 norms and add them up.
        l2s = tf.stack([tf.nn.l2_loss(w) for w in weights])
        weight_norm = weight_decay * tf.reduce_sum(l2s)

        # Find the cross entropy
        # oh_labels = tf.one_hot(labels, num_classes)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            # labels=oh_labels
            labels=labels
        )

        # Add the cross entropy and weight norms together.
        cross_entropy = tf.reduce_mean(cross_entropy)
        loss_val = cross_entropy + weight_norm

        to_run['weight_norm'] = weight_norm
        to_run['cross_entropy'] = cross_entropy

    return loss_val


def top_k_accuracy(logits, labels, k=1):
    """
    Calculate the percentage of time the label is in the top k predictions.

    Args:
        logits: The logits operation outputted by the model.
        labels: The image labels.
        k: an int. The number of top predictions to look for the correct label in.
    Returns:
        accuracy: A scalar tensor of the accuracy.
    """
    preds = tf.nn.softmax(logits)
    correct = tf.nn.in_top_k(predictions=preds, targets=labels, k=k)
    accuracy = tf.reduce_mean(tf.to_float(correct))
    return accuracy


def optimize(loss, learning_rate=0.01, momentum=0.9, global_step=None, to_run=None):
    """
    The optimization operation definition.

    Args:
        loss: a scalar tensor of he cross entropy + weight norm
        learning_rate: a float. The learning rate.
        momentum: a float. The learning momentum.
        global_step: The training step.
        to_run: a dictionary which can be passed to pick up any ops inside the function that need to be fed to sess.run. Useful for debugging.
    Returns:
        minimize: The minization operation.
    """
    with tf.name_scope('optimize'):
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=momentum
        )

        return optimizer.minimize(loss=loss, global_step=global_step)


def load_weights(session, weights_path):
    """
    Load the pretrained weights from a numpy file..

    Args:
        session: current tensorflow session
        weights_path: a string. The path to the weights file.
    """
    print "Info: Restoring weights from numpy file: {}".format(weights_path)

    # Load the weights into memory
    weights_dict = np.load(weights_path, encoding='bytes').item()

    # Loop over all layer ops
    for op_name in weights_dict:
        op_name_string = op_name if isinstance(op_name, str) else op_name.decode('utf8')

        print "  restore: {}".format(op_name_string)
        with tf.variable_scope(op_name_string, reuse=True):
            # Loop over list of weights/biases and assign them to their corresponding tf variable

            for data in weights_dict[op_name]:
                # Biases
                if len(data.shape) == 1:
                    var = tf.get_variable('biases')
                    session.run(var.assign(data))
                # Weights
                else:
                    var = tf.get_variable('weights')
                    session.run(var.assign(data))
