"""vonclites implementation"""
import os
import tensorflow as tf

from deployment import model_deploy
import metrics
import glob
import pprint
import os
import sys
import imp
import model_def as md
import image_net.pipeline as inputs

def _clone_fn(images,
              labels,
              index_iter,
              is_training):
    clone_index = next(index_iter)
    images = images[clone_index]
    labels = labels[clone_index]

    unscaled_logits, _ = md.squeeze_net_model(
        images,
        is_training=is_training
    )

    tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=unscaled_logits
    )
    predictions = tf.argmax(unscaled_logits, 1, name='predictions')
    return {
        'predictions': predictions,
        'images': images,
    }


def _configure_deployment(num_gpus):
    return model_deploy.DeploymentConfig(num_clones=num_gpus)


def _configure_session():
    gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=.8)
    return tf.ConfigProto(allow_soft_placement=True,
                          gpu_options=gpu_config)


def main(cfg):

    with tf.Graph().as_default():

        deploy_config = _configure_deployment(cfg.num_gpus)
        sess = tf.Session(config=_configure_session())

        with tf.device(deploy_config.variables_device()):
            global_step = tf.train.create_global_step()

        with tf.device(deploy_config.optimizer_device()):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=cfg.learning_rate
            )

        # Create the input pipeline from the tf_record files
        with tf.device(deploy_config.inputs_device()), tf.name_scope('inputs'):
            pipeline = inputs.Pipeline(cfg, sess)
            examples, labels = pipeline.data
            images = examples['image']

            image_splits = tf.split(
                value=images,
                num_or_size_splits=deploy_config.num_clones,
                name='split_images'
            )
            label_splits = tf.split(
                value=labels,
                num_or_size_splits=deploy_config.num_clones,
                name='split_labels'
            )

        # Build the model graph
        model_dp = model_deploy.deploy(
            config=deploy_config,
            model_fn=_clone_fn,
            optimizer=optimizer,
            kwargs={
                'images': image_splits,
                'labels': label_splits,
                'index_iter': iter(range(deploy_config.num_clones)),
                'is_training': pipeline.is_training
            }
        )

        # Define the various accuracy metrics and whatnot
        train_metrics = metrics.Metrics(
            labels=labels,
            clone_predictions=[clone.outputs['predictions']
                               for clone in model_dp.clones],
            device=deploy_config.variables_device(),
            name='training'
        )
        validation_metrics = metrics.Metrics(
            labels=labels,
            clone_predictions=[clone.outputs['predictions']
                               for clone in model_dp.clones],
            device=deploy_config.variables_device(),
            name='validation',
            padded_data=True
        )
        validation_init_op = tf.group(
            pipeline.validation_iterator.initializer,
            validation_metrics.reset_op
        )
        train_op = tf.group(
            model_dp.train_op,
            train_metrics.update_op
        )

        # Create the summaries
        with tf.device(deploy_config.variables_device()):
            train_writer = tf.summary.FileWriter(cfg.model_dir, sess.graph)
            eval_dir = os.path.join(cfg.model_dir, 'eval')
            eval_writer = tf.summary.FileWriter(eval_dir, sess.graph)
            tf.summary.scalar('accuracy', train_metrics.accuracy)
            tf.summary.scalar('loss', model_dp.total_loss)
            all_summaries = tf.summary.merge_all()

        # Define checkpoints to save model
        saver = tf.train.Saver(max_to_keep=cfg.keep_last_n_checkpoints)
        save_path = os.path.join(cfg.model_dir, 'model.ckpt')

        # Initialize the parameters, whether it be loading in weights or
        # starting from scratch.
        last_checkpoint = tf.train.latest_checkpoint(cfg.model_dir)
        if last_checkpoint:
            saver.restore(sess, last_checkpoint)
        else:
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)
        starting_step = sess.run(global_step)

        # Training loop
        for train_step in range(starting_step, cfg.max_train_steps):
            sess.run(train_op, feed_dict=pipeline.training_data)

            # When a summary is triggered.
            if train_step % cfg.summary_interval == 0:
                results = sess.run(
                    fetches={'accuracy': train_metrics.accuracy,
                             'summary': all_summaries},
                    feed_dict=pipeline.training_data
                )
                train_writer.add_summary(results['summary'], train_step)
                print('Train Step {:<5}:  {:>.4}'
                      .format(train_step, results['accuracy']))

            # When a model save checkpoint is reached
            if train_step % cfg.checkpoint_interval == 0:
                saver.save(sess, save_path, global_step)

            sess.run(train_metrics.reset_op)

            # When an model evaluation is reached.
            if train_step % cfg.validation_interval == 0:
                while True:
                    try:
                        sess.run(
                            fetches=validation_metrics.update_op,
                            feed_dict=pipeline.validation_data
                        )
                    except tf.errors.OutOfRangeError:
                        break
                results = sess.run({'accuracy': validation_metrics.accuracy})

                print('Evaluation Step {:<5}:  {:>.4}'
                      .format(train_step, results['accuracy']))

                summary = tf.Summary(value=[
                    tf.Summary.Value(tag='accuracy', simple_value=results['accuracy']),
                ])
                eval_writer.add_summary(summary, train_step)
                sess.run(validation_init_op)  # Reinitialize dataset and metrics


if __name__ == '__main__':
    # Read in config file
    assert len(sys.argv) == 2, "Must pass exactly one argument to the script, namely the cfg file, got " + str(len(sys.argv))
    abs_path = os.path.abspath(sys.argv[1])
    cfg = imp.load_source('*', abs_path)
    main(cfg)
