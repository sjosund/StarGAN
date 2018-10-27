from collections import defaultdict
from functools import partial
import os

import configparser
from pathlib import Path
from shutil import copyfile
import time

from fire import Fire
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tqdm import tqdm

CHANNELS_FIRST = 'channels_first'
CHANNELS_LAST = 'channels_last'

INSTANCE_NORM_FORMAT = {
    CHANNELS_LAST: 'NHWC',
    CHANNELS_FIRST: 'NCHW',
}


class StarGAN:
    def __init__(self, local=False):
        current_time = int(time.time())

        config_path = 'config.ini' if not local else 'config_local.ini'
        paths_config_path = 'local.ini' if local else 'remote.ini'

        config = configparser.ConfigParser()
        config.read(config_path)
        paths_config = configparser.ConfigParser()
        paths_config.read(paths_config_path)

        self.data_path = paths_config['paths']['data']
        self.base_path = '/{}/celeba/'.format(self.data_path)
        self.img_base_path = '{}/img_align_celeba'.format(self.base_path)
        self.annotations_path = '{}/Anno/list_attr_celeba.txt'.format(
            self.base_path)
        self.models_dir = '{}{}'.format(
            paths_config['paths']['model'],
            current_time)
        self.simple_save_dir = '{}/simple_save'.format(self.models_dir)
        self.logs_dir = '{}{}'.format(
            paths_config['paths']['logs'],
            current_time)

        os.makedirs(self.models_dir)
        os.makedirs(self.logs_dir)

        config = configparser.ConfigParser()
        config.read(config_path)
        copyfile(config_path, '{}/config.ini'.format(self.logs_dir))

        self.used_attributes = config['data']['attributes'].split(',')
        self.IMG_SIZE = (
            int(config['data']['image_size']),
            int(config['data']['image_size'])
        )
        self.BATCH_SIZE = int(config['training']['batch_size'])
        self.n_epochs = int(config['training']['n_epochs'])
        self.batches_per_epoch = int(config['training']['batches_per_epoch']) if config['training'].get('batches_per_epoch') else None
        self.n_steps_generator = int(config['training']['n_steps_generator'])
        self.n_steps_discriminator = int(
            config['training']['n_steps_discriminator'])

        self.l_gp = float(config['loss']['l_gp'])
        self.l_cls = float(config['loss']['l_cls'])
        self.l_rec = float(config['loss']['l_rec'])
        self.l_adv = float(config['loss']['l_adv'])

        self.lr = float(config['optimizer']['lr'])

        self.n_examples = int(config['data']['n_examples'])
        self.initializer = partial(
            tf.truncated_normal_initializer,
            stddev=float(config['model']['stddev'])
        )
        self.data_format = paths_config['data']['format']
        self.debug = False

        self.sess = tf.Session()
        self.load_from_checkpoint = True

    def run(self):

        annotations = self.get_annotations()
        self.batches_per_epoch = int(len(annotations) / (self.BATCH_SIZE * (
                    self.n_steps_discriminator + self.n_steps_generator))) if self.batches_per_epoch is None else self.batches_per_epoch
        print('Batches per epoch', self.batches_per_epoch)
        next_image, next_source_label, next_target_label = self.dataset(
            annotations)

        # Setup Loss
        next_image_shape = next_image.shape.as_list()
        labels_as_image = tf_repeat_to_img(
            next_target_label,
            next_image_shape[1:-1] if self.data_format == CHANNELS_LAST else
                next_image_shape[2:],
            data_format=self.data_format
        )
        next_image_with_label = tf.concat(
            [next_image, labels_as_image],
            axis=-1 if self.data_format == CHANNELS_LAST else 1,
            name='serving_input'
        )

        G = tf.identity(self.generator(next_image_with_label),
                        name='serving_output')
        D_src_g, D_class_g = self.discriminator(G)
        D_src_x, D_class_x = self.discriminator(next_image)

        # with tf.variable_scope('gradient_penalty'):
        diff = G - next_image
        lambda_ = tf.random_uniform(shape=[self.BATCH_SIZE, 1, 1, 1])
        sampled_images = next_image + lambda_ * diff
        D_sampled_src, _ = self.discriminator(sampled_images)
        grads = tf.gradients(D_sampled_src, sampled_images)
        #     gradient_norm = tf.norm(grads, axis=0)  # Not numerically stable.
        # approximate the 2-norm with the Frobenius norm.
        epsilon = 1e-12
        gradient_norm = tf.sqrt(
            tf.reduce_sum(
                tf.square(grads),
                axis=[1, 2, 3]  # All axis except batch.
            ) + epsilon  # Add epsilon for numerical stability.
        )

        gradient_penalty = tf.reduce_mean(
            tf.square(
                gradient_norm - 1
            )
        )

        L_adv_D = tf.reduce_mean(D_src_x) - tf.reduce_mean(
            D_src_g) - self.l_gp * gradient_penalty
        L_adv_G = -tf.reduce_mean(D_src_g)

        G_shape = G.shape.as_list()
        source_labels_as_image = tf_repeat_to_img(
            next_source_label,
            G_shape[1:-1] if self.data_format == CHANNELS_LAST else G_shape[2:],
            data_format=self.data_format
        )
        # Reconstruction Loss
        G_with_source_label = tf.concat(
            [G, source_labels_as_image],
            axis=-1 if self.data_format == CHANNELS_LAST else 1
        )

        G_rec = self.generator(G_with_source_label)
        L_rec = tf.reduce_mean(tf.losses.absolute_difference(G_rec, next_image))

        # Classification loss
        L_G_cls = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.squeeze(next_target_label),
                logits=tf.squeeze(D_class_g)
            )
        )
        L_D_cls = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.squeeze(next_source_label),
                logits=tf.squeeze(D_class_x)
            )
        )

        tf.summary.image(
            'generated',
            tf.transpose(G, perm=(
            0, 2, 3, 1)) if self.data_format == CHANNELS_FIRST else G,
            max_outputs=3,
        )
        tf.summary.image(
            'reconstructed',
            tf.transpose(G_rec, perm=(
            0, 2, 3, 1)) if self.data_format == CHANNELS_FIRST else G_rec,
            max_outputs=3,
        )
        tf.summary.image(
            'original',
            tf.transpose(next_image, perm=(
            0, 2, 3, 1)) if self.data_format == CHANNELS_FIRST else next_image,
            max_outputs=3
        )

        L_total_D = -self.l_adv * L_adv_D + self.l_cls * L_D_cls
        L_total_G = self.l_adv * L_adv_G + self.l_cls * L_G_cls + self.l_rec * L_rec

        if self.debug:
            for loss_name, loss in [('L_adv_D', L_adv_D), ('L_G_cls', L_G_cls),
                                    ('L_D_cls', L_D_cls)]:
                for i in range(1, 7):
                    d = tf.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES,
                        'discriminator/d{}/kernel'.format(i)
                    )[0]
                    grad_norm = tf.norm(
                        tf.gradients(
                            ys=loss,
                            xs=d
                        )
                    )
                    tf.summary.scalar(
                        'grad_{}_{}'.format(loss_name, i),
                        grad_norm
                    )

        tf.summary.scalar('L_adv_D', L_adv_D)
        tf.summary.scalar('L_G_cls', L_G_cls)
        tf.summary.scalar('L_D_cls', L_D_cls)
        tf.summary.scalar('L_rec', L_rec)
        tf.summary.scalar('L_tot_G', L_total_G)
        tf.summary.scalar('L_tot_D', L_total_D)
        tf.summary.scalar('gradient_penalty', gradient_penalty)
        tf.summary.scalar('learning_rate', self.lr)

        if self.debug:
            tf.summary.histogram('D_class_g', D_class_g)
            tf.summary.histogram('D_x_class', D_class_x)

            tf.summary.histogram('G_rec', G_rec)
            tf.summary.histogram('G', G)

        merged_summary = tf.summary.merge_all()

        with tf.variable_scope('optimizers', reuse=tf.AUTO_REUSE):
            learning_rate = tf.placeholder(dtype=tf.float32, shape=None)
            opt_D = tf.train.AdamOptimizer(
                beta1=0.5,
                beta2=0.999,
                learning_rate=learning_rate,
                name='D'
            )
            opt_op_D = opt_D.minimize(
                L_total_D,
                var_list=tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    scope='discriminator'
                ),
                #         name='D_opt_op'
            )

            opt_G = tf.train.AdamOptimizer(
                beta1=0.5,
                beta2=0.999,
                learning_rate=learning_rate,
                name='G'
            )
            opt_op_G = opt_G.minimize(
                L_total_G,
                var_list=tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    scope='generator'
                ),
                #         name='G_opt_op'
            )

        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        # saver.restore(sess, '{}/model.ckpt-4')

        summary_writer = tf.summary.FileWriter(
            self.logs_dir,
            graph=self.sess.graph
        )

        lr_start = self.lr

        builder = tf.profiler.ProfileOptionBuilder
        opts = builder(builder.time_and_memory()).order_by('micros').build()

        # if self.load_from_checkpoint:
        #     current_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #     assign_ops = []
        #
        #     new_graph = tf.Graph()
        #     with tf.Session(graph=new_graph) as sess:
        #         tf.saved_model.loader.load(
        #             sess=sess,
        #             tags=[tag_constants.SERVING],
        #             export_dir='/Users/sjosund/Programming/models/stargan/stargan_original1537059936/simple_save/1')
        #         new_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #         for v in new_vars:
        #             v_ = [vv for vv in current_vars if v.name == vv.name][0]
        #             op = v_.assign(sess.run(v))
        #             assign_ops.append(op)
        #         self.sess.run(assign_ops)
                # tf.saved_model.simple_save(
                #     session=sess,
                #     export_dir='{}/{}'.format(self.simple_save_dir, 0),
                #     inputs={'serving_input': next_image_with_label},
                #     outputs={'serving_output': G}
                # )
                # return


        with tf.contrib.tfprof.ProfileContext('/tmp/train_dir',
                                              trace_steps=[],
                                              dump_steps=[]) as pctx:
            for epoch in tqdm(range(self.n_epochs)):
                losses = defaultdict(list)
                for j in tqdm(range(self.batches_per_epoch)):
                    losses_g = []
                    losses_d = []
                    for i in range(self.n_steps_generator):
                        # pctx.trace_next_step()
                        _, loss_g = self.sess.run(
                            [opt_op_G, L_total_G],
                            feed_dict={learning_rate: self.lr}
                        )
                        losses_g.append(loss_g)
                    for i in range(self.n_steps_discriminator):
                        # pctx.trace_next_step()
                        _, loss_d = self.sess.run(
                            [opt_op_D, L_total_D],
                            feed_dict={learning_rate: self.lr}
                        )
                        losses_d.append(loss_d)
                    losses['g'].append(np.mean(losses_g))
                    losses['d'].append(np.mean(losses_d))
                    summary = self.sess.run(merged_summary)
                # pctx.profiler.profile_operations(options=opts)

                # Learning rate decay
                if epoch >= 10:
                    self.lr -= lr_start / 10

                summary_writer.add_summary(summary=summary, global_step=epoch)
                saver.save(sess=self.sess,
                           save_path='{}/model.ckpt'.format(self.models_dir),
                           global_step=i)
                tf.saved_model.simple_save(
                    session=self.sess,
                    export_dir='{}/{}'.format(self.simple_save_dir, epoch),
                    inputs={'serving_input': next_image_with_label},
                    outputs={'serving_output': G}
                )
                print('Saved')

    def get_annotations(self):
        paths = list(Path(self.img_base_path).glob('*.jpg'))
        annotations = pd.read_csv(
            self.annotations_path,
            delim_whitespace=True,
            skiprows=1,
        )

        paths_ = list(map(
            lambda p: str(p).split('/')[-1],
            paths
        ))
        annotations = annotations.loc[paths_]
        annotations[annotations == -1] = 0
        annotations['path'] = annotations.index.map(
            lambda p: '{}/{}'.format(self.img_base_path, p))

        if self.n_examples != -1:
            annotations = annotations.iloc[:self.n_examples]
        return annotations


    def dataset(self, annotations):
        # TODO Use dict instead.
        # Data Loader
        labels = tf.constant(annotations[self.used_attributes].values,
                             dtype=tf.int8)
        paths = tf.constant(annotations['path'].values, dtype=tf.string)

        labels_ds = tf.data.Dataset.from_tensor_slices(labels)
        paths_ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = tf.data.Dataset.zip((labels_ds, paths_ds))
        ds = ds.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=32))
        ds = ds.apply(tf.contrib.data.map_and_batch(
            partial(parse_example, img_size=self.IMG_SIZE),
            batch_size=self.BATCH_SIZE,
            # num_parallel_batches=4
            num_parallel_calls=4
        ))
        ds = ds.prefetch(buffer_size=16)

        it = ds.make_one_shot_iterator()
        next_source_label, next_image = it.get_next()
        if self.data_format == CHANNELS_FIRST:
            next_image = tf.transpose(next_image, perm=[0, 3, 1, 2])
        next_source_label = tf.cast(next_source_label, tf.float32)

        bernoulli = tf.distributions.Bernoulli(probs=0.5, dtype=tf.float32)
        next_target_label = bernoulli.sample(
            sample_shape=tf.shape(next_source_label), )

        return next_image, next_source_label, next_target_label

    def generator(self, x):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            x = self.conv_block(x, 64, 7, 1, 3, name='c1')
            print('Generator')
            print(x.shape)
            x = self.conv_block(x, 128, 4, 2, 1, name='c2')
            print(x.shape)
            x = self.conv_block(x, 256, 4, 2, 1, name='c3')
            print(x.shape)

            x = self.residual_block(x, name='r1')
            print(x.shape)
            x = self.residual_block(x, name='r2')
            print(x.shape)
            x = self.residual_block(x, name='r3')
            print(x.shape)
            x = self.residual_block(x, name='r4')
            print(x.shape)
            x = self.residual_block(x, name='r5')
            print(x.shape)
            x = self.residual_block(x, name='r6')
            print(x.shape)

            x = self.deconv_block(x, 128, name='d1')
            print(x.shape)
            x = self.deconv_block(x, 64,
                                  name='d2')  # TODO Change back to the this
            print(x.shape)

            x = tf.layers.conv2d(
                inputs=x,
                filters=3,
                kernel_size=7,
                strides=1,
                activation=tf.nn.tanh,
                padding='same',
                name='output',
                kernel_initializer=self.initializer(),
                data_format=self.data_format,
            )
            print(x.shape)
            print('--------------------')

            return x

    def add_weight_summary(self, name):
        tf.summary.scalar(
            'kernel_{}'.format(name),
            tf.norm(
                tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    'discriminator/{}/kernel'.format(name)
                )[0]
            ),

        )
        tf.summary.scalar(
            'bias_{}'.format(name),
            tf.norm(
                tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    'discriminator/{}/bias'.format(name)
                )[0]
            )
        )

    def discriminator(self, x):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            print('Dicsriminator')
            print(x.shape)
            if self.debug:
                tf.summary.histogram('d_input', x)
            x = self.discriminator_block(x, 64, name='d1')
            print(x.shape)
            if self.debug:
                tf.summary.histogram('d1', x)
            x = self.discriminator_block(x, 128, name='d2')
            print(x.shape)
            if self.debug:
                tf.summary.histogram('d2', x)
            x = self.discriminator_block(x, 256, name='d3')
            print(x.shape)
            if self.debug:
                tf.summary.histogram('d3', x)
            x = self.discriminator_block(x, 512, name='d4')
            print(x.shape)
            if self.debug:
                tf.summary.histogram('d4', x)
            x = self.discriminator_block(x, 1024, name='d5')
            print(x.shape)
            if self.debug:
                tf.summary.histogram('d5', x)
            x = self.discriminator_block(x, 2048, name='d6')
            print(x.shape)
            if self.debug:
                tf.summary.histogram('d6', x)

                for i in range(1, 7):
                    self.add_weight_summary('d{}'.format(i))

            output_cls = tf.layers.conv2d(
                inputs=x,
                filters=len(self.used_attributes),
                kernel_size=2,
                strides=1,
                padding='valid',
                kernel_initializer=self.initializer(),
                data_format=self.data_format,
            )
            print(output_cls.shape)
            output_src = tf.layers.conv2d(
                inputs=x,
                filters=1,
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer=self.initializer(),
                data_format=self.data_format,
            )
            print(output_src.shape)

            return output_src, output_cls

    def conv_block(self, x, filters, kernel_size, stride, padding,
                   name, activation=tf.nn.relu, instance_norm=True):
        x = pad_image(x, padding=padding, data_format=self.data_format)
        x = tf.layers.conv2d(
            inputs=x,
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            name=name,
            kernel_initializer=self.initializer(),
            data_format=self.data_format,
            # scale=0.01, mode='fan_avg', distribution='normal'
            # )#tf.glorot_normal_initializer(),
        )
        if instance_norm:
            x = tf.contrib.layers.instance_norm(
                x, data_format=INSTANCE_NORM_FORMAT[self.data_format])
        x = activation(x)

        return x

    def discriminator_block(self, x, filters, name):
        x = self.conv_block(
            x,
            filters=filters,
            kernel_size=4,
            stride=2,
            padding=1,
            name=name,
            activation=partial(tf.nn.leaky_relu, alpha=0.2),
            instance_norm=False
        )

        return x

    def residual_block(self, x, name):
        return self.conv_block(x, 256, 3, 1, 1, name=name)

    def deconv_block(self, x, filters, name):
        x = tf.layers.conv2d_transpose(
            inputs=x,
            filters=filters,
            kernel_size=4,
            strides=2,
            padding='same',
            name=name,
            kernel_initializer=self.initializer(),
            data_format=self.data_format,
        )
        x = tf.contrib.layers.instance_norm(
            x, data_format=INSTANCE_NORM_FORMAT[self.data_format])
        x = tf.nn.relu(x)

        return x


def pad_image(x, padding, data_format):
    if data_format == CHANNELS_FIRST:
        paddings = tf.constant([
            [0, 0],
            [0, 0],
            [padding, padding],
            [padding, padding]
        ])
    else:
        paddings = tf.constant([
            [0, 0],
            [padding, padding],
            [padding, padding],
            [0, 0]])
    x = tf.pad(x, paddings=paddings, mode='CONSTANT')
    return x


def parse_example(labels, path, img_size):
    img = tf.read_file(path)
    img = tf.image.decode_jpeg(contents=img, channels=3)
    img = tf.image.crop_to_bounding_box(
        image=img,
        offset_height=20,
        offset_width=0,
        target_height=178,
        target_width=178
    )
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize_images(
        img,
        size=tf.constant(img_size, dtype=tf.int32)
    )
    img = (img - 128.) / 128.

    return labels, img


def tf_repeat_to_img(tensor, repeats, data_format):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension,
        length must be the same as the number of dimensions in input

    Returns:

    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.variable_scope("repeat"):
        # expanded_tensor = tf.expand_dims(tensor, -1)
        if data_format == CHANNELS_LAST:
            expanded_tensor = tf.expand_dims(tf.expand_dims(tensor, 1), 1)
            multiples = [1] + repeats + [1]
        else:
            expanded_tensor = tf.expand_dims(tf.expand_dims(tensor, -1), -1)
            multiples = [1, 1] + repeats

        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        # repeated_tensor = tf.reshape(tiled_tensor,
        #                              [tf.shape(tensor)[0]] + repeats + [1])
        # repeated_tensor = tf.cast(repeated_tensor, tf.float32)
        repeated_tensor = tf.cast(tiled_tensor, tf.float32)
    return repeated_tensor


if __name__ == '__main__':
    Fire(StarGAN)
