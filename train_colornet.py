import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from dataset.dataset_raw import RawDataSet
from nets import nets_factory
import tf_utils
import os

slim = tf.contrib.slim

DATA_FORMAT = 'NHWC'

dataset_params = {
"path" : "./training_data/",
"thread_num" : 1,
}
common_params = {
"image_width" : 512,
"image_height" : 512,
"batch_size" : 5,
}

# =========================================================================== #
# colornet flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'loss_alpha', 1., 'Alpha parameter in the loss function.')

# =========================================================================== #
# General Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.8, 'GPU memory fraction to use.')

# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')
tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')
tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.97, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 9.0,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', 'lisa', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 7, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', 'tf_records', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
    'model_name', 'colornet', 'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'batch_size', 10, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')
tf.app.flags.DEFINE_integer('max_number_of_steps', 500000,
                            'The maximum number of training steps.')

# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', None,
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS

# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():
        global_step = slim.create_global_step()
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(FLAGS.train_dir))
        sess = tf.InteractiveSession()
        dataset = RawDataSet(common_params, dataset_params)

        colornet_class = nets_factory.get_network(FLAGS.model_name)
        colornet_params = colornet_class.default_params
        color_net = colornet_class(colornet_params)
        color_net_shape = color_net.params.img_shape

        input_tensor = tf.placeholder(tf.float32, shape=(None, 512, 512, 3), name='input_image')
        gt_tensor = tf.placeholder(tf.float32, shape=(None, 512, 512, 3), name='groundtruth_image')

        arg_scope = color_net.arg_scope(weight_decay=FLAGS.weight_decay, data_format=DATA_FORMAT)
        with slim.arg_scope(arg_scope):
            predictions, end_points = color_net.net(input_tensor, is_training=True)

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        color_net.losses(gt_tensor, predictions)
        total_loss = tf.losses.get_total_loss()
        summaries.add(tf.summary.scalar('loss', total_loss))
        for variable in tf.trainable_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        learning_rate = tf_utils.configure_learning_rate(FLAGS, dataset.record_numbers, global_step)
        optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)

        summaries.add(tf.summary.scalar('learning_rate', learning_rate))
        summaries.add(tf.summary.image('input_img', input_tensor))
        summaries.add(tf.summary.image('gt_img', gt_tensor))
        summaries.add(tf.summary.image('pre_img',predictions))

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = slim.learning.create_train_op(
                total_loss,
                optimizer,
                summarize_gradients=False
            )
        summary_op = tf.summary.merge(list(summaries), name='summary_op')
        train_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
        config = tf.ConfigProto(log_device_placement=False,
                                gpu_options = gpu_options)
        saver = tf.train.Saver(max_to_keep=5,
                               keep_checkpoint_every_n_hours=1.0,
                               write_version=2,
                               pad_step_number=False)
        sess.run(tf.global_variables_initializer())

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        i = 0
        with slim.queues.QueueRunners(sess):

            while (i < FLAGS.max_number_of_steps):
                input_images, output_images = dataset.batch()
                _, summary_str = sess.run([train_op, summary_op], feed_dict={input_tensor: input_images, gt_tensor: output_images})
                if i % 50 == 0:
                    global_step_str = global_step.eval()
                    print('%diteraton' % (global_step_str))
                    train_writer.add_summary(summary_str, global_step_str)
                if i % 100 == 0:
                    global_step_str = global_step.eval()
                    saver.save(sess, FLAGS.train_dir, global_step=global_step_str)

                i += 1

if __name__ == '__main__':
    tf.app.run()
