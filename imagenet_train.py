'''Ackownledgement:

Part of codes was based Amazon AWS
from https://github.com/aws-samples/deep-learning-models/tree/master/models/resnet/tensorflow

they used the Horovod to train the imagenet in 6 and half hours using single p3.16xlarge, by utilizing some convergence technique, we improve the result from
6 and half hours to 104 mins

Author: Min Zhang: min.zhang@ge.com

'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
import horovod.tensorflow as hvd
import os
import sys
import time
import argparse
import shutil
import logging
import re
from operator import itemgetter
import dynamicpipe  
import resnet50

def rank0log(logger, *args, **kwargs):
    if hvd.rank() == 0:
        logger.info('Log: '.join([str(x) for x in list(args)]))

class LogSessionRunHook(tf.train.SessionRunHook):
    def __init__(self, num_records, num_gpus, display_every=10, logger=None):
        self.num_records = num_records
        self.display_every = display_every
        self.logger = logger
        self.num_gpus = num_gpus

    def after_create_session(self, session, coord):
        rank0log(self.logger, 'Log:  Step Epoch Speed   Loss  FinLoss   LR   bs   imsize')
        self.elapsed_secs = 0.
        self.count = 0
        self.epoch = 0.
        self.total_sec = 0.
        self.start_t0 = time.time()
        
    def before_run(self, run_context):
        self.t0 = time.time()
        return tf.train.SessionRunArgs(
            fetches=[tf.train.get_global_step(),
                     'loss:0', 'total_loss:0', 'learning_rate:0', 'batch_size:0', 'image_size:0', 'trn-top5acc:0'])

    def after_run(self, run_context, run_values):
        self.elapsed_secs += time.time() - self.t0
        self.total_sec = time.time() - self.start_t0 
        self.count += 1
        
        global_step, loss, total_loss, lr, batch_size, image_size, top5acc = run_values.results
        self.epoch = self.epoch + batch_size * self.num_gpus / self.num_records
        if global_step == 1 or global_step % self.display_every == 0:
            dt = self.elapsed_secs / self.count
            img_per_sec = (batch_size * self.num_gpus) / dt
            self.logger.info('Log @train: steps@{:d} epoch@{:.1f} im/s@{:.2f} loss@{:.3f} total_loss@{:.3f} lr@{:.5f} bs@{:d} sz@{:d} top5@{:.4f} tm@{:.2f}'.format(global_step, self.epoch, img_per_sec, loss, total_loss, lr, batch_size, image_size, top5acc, self.total_sec))
            self.elapsed_secs = 0.
            self.count = 0

            
def cnn_model_function(features, labels, mode, params):
    labels = tf.reshape(labels, (-1,))  # Squash unnecessary unary dim
    
    model_dtype = tf.float16
    model_format = 'channels_first'
  
    inputs = features  # TODO: Should be using feature columns?
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    
    num_classes = params['n_classes']
    momentum = params['mom']
    weight_decay = params['wdecay']
    num_training_samples= params['num_training_samples']
    num_steps = params['num_steps']
    loss_scale = params['loss_scale']
    
    lr_strategy = params['lr_strategy']

    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.device('/cpu:0'):
            preload_op, (inputs, labels) = stage([inputs, labels])

    with tf.device('/gpu:0'):
        if mode == tf.estimator.ModeKeys.TRAIN:
            gpucopy_op, (inputs, labels) = stage([inputs, labels])
        inputs = tf.cast(inputs, model_dtype)
        imagenet_mean = np.array([121, 115, 100], dtype=np.float32)
        imagenet_std = np.array([70, 68, 71], dtype=np.float32)
        inputs = tf.subtract(inputs, imagenet_mean)
        inputs = tf.multiply(inputs, 1. / imagenet_std)
        if model_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])
        with fp32_trainable_vars(
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay)):
            top_layer = resnet50.inference_resnet_v1(
                inputs, data_format=model_format, training=is_training,
                conv_initializer=tf.variance_scaling_initializer(), adv_bn_init=True)
            logits = tf.layers.dense(top_layer, num_classes,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        predicted_classes = tf.argmax(logits, axis=1, output_type=tf.int32)
        logits = tf.cast(logits, tf.float32)
        if mode == tf.estimator.ModeKeys.PREDICT:
            probabilities = tf.softmax(logits)
            predictions = {
                'class_ids': predicted_classes[:, None],
                'probabilities': probabilities,
                'logits': logits
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
        
        
        
        train_top5acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32))
        
        #loss = tf.reduce_mean(-10 * tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32))
        
        loss = tf.identity(loss, name='loss')  # For access by logger (TODO: Better way to access it?)
        if mode == tf.estimator.ModeKeys.EVAL:
            with tf.device(None):
                # Allow fallback to CPU if no GPU support for these ops                
                accuracy = tf.metrics.accuracy(
                    labels=labels, predictions=predicted_classes)
                top5acc = tf.metrics.mean(
                    tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32))
                #newaccuracy = (hvd.allreduce(accuracy[0]), accuracy[1])
                #newtop5acc = (hvd.allreduce(top5acc[0]), top5acc[1])
                metrics = {'val-top1acc': accuracy, 'val-top5acc': top5acc}
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        assert (mode == tf.estimator.ModeKeys.TRAIN)
        

        
        global_step = tf.train.get_global_step()

        batch_size = tf.shape(inputs)[0]
        image_size = tf.shape(inputs)[2]
        
        
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([loss] + reg_losses, name='total_loss')
        
        with tf.device('/cpu:0'):  # Allow fallback to CPU if no GPU support for these ops
            learning_rate = dynamicpipe.learning_rate_schedule(lr_strategy, global_step)

            learning_rate = tf.identity(learning_rate, 'learning_rate')
            batch_size = tf.identity(batch_size, 'batch_size')
            image_size = tf.identity(image_size, 'image_size')
            
            
            train_top5acc = tf.identity(train_top5acc, 'trn-top5acc')
            tf.summary.scalar('trn-top5acc', train_top5acc)
            
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('batch_size', batch_size)
            tf.summary.scalar('image_size', image_size)
            
            
        opt = tf.train.MomentumOptimizer(
            learning_rate, momentum, use_nesterov=True)
        opt = hvd.DistributedOptimizer(opt)
        opt = MixedOptimizer(opt, scale=loss_scale)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) or []
        with tf.control_dependencies(update_ops):
            gate_gradients = (tf.train.Optimizer.GATE_NONE)
            train_op = opt.minimize(
                total_loss, global_step=tf.train.get_global_step(),
                gate_gradients=gate_gradients)
        train_op = tf.group(preload_op, gpucopy_op, train_op)  # , update_ops)

        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)

    
def add_cli_args():
    cmdline = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    cmdline.add_argument('--data_dir', default='/opt/ml/input/data/training',
                         help="""Path to dataset in TFRecord format
                         (aka Example protobufs). Files should be
                         named 'train-*' and 'validation-*'.""")
    
    cmdline.add_argument('--log_dir', default='/opt/ml/model/imagenet_resnet',
                         help="""Directory in which to write training
                         summaries and checkpoints. If the log directory already
                         contains some checkpoints, it tries to resume training
                         from the last saved checkpoint. Pass --clear_log if you
                         want to clear all checkpoints and start a fresh run""")
  
    cmdline.add_argument('--display_every', default=100, type=int,
                         help="""How often (in iterations) to print out
                         running information.""")
   
    cmdline.add_argument('--num_gpus', default=8, type=int,
                         help="""Specify total number of GPUS used to train a checkpointed model during eval.
                                Used only to calculate epoch number to print during evaluation""")
    cmdline.add_argument('--save_checkpoints_steps', type=int, default=1000)
    cmdline.add_argument('--save_summary_steps', type=int, default=1000)
   
    
    cmdline.add_argument('--mom', default=0.977, type=float,
                         help="""Momentum""")
    cmdline.add_argument('--wdecay', default=0.0005, type=float,
                         help="""Weight decay""")
    cmdline.add_argument('--loss_scale', default=256., type=float,
                         help="""loss scale""")
   
    return cmdline

def sort_and_load_ckpts(log_dir):
    ckpts = []
    for f in os.listdir(log_dir):
        m = re.match(r'model.ckpt-([0-9]+).index', f)
        if m is None:
            continue
        fullpath = os.path.join(log_dir, f)
        ckpts.append({'step': int(m.group(1)),
                      'path': os.path.splitext(fullpath)[0],
                      'mtime': os.stat(fullpath).st_mtime,
                      })
    ckpts.sort(key=itemgetter('step'))
    return ckpts

def main():
    gpu_thread_count = 2
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = str(gpu_thread_count)
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    hvd.init()
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.gpu_options.force_gpu_compatible = True  # Force pinned memory
    config.intra_op_parallelism_threads = 1  # Avoid pool of Eigen threads
    config.inter_op_parallelism_threads = 5
    #config.gpu_options.allow_growth = True
    log_name = 'hvd_train.txt'
    
    '''
      training stratey
    '''
    
    
    training_strategy = [
    {'epoch':[0,4], 'lr': [1.0,3.0],'lr_method':'linear','batch_size':768, 'image_size':(128, 128), 'data_dir':'160', 'prefix':'train'},
    {'epoch':[4,15], 'lr': [3.0,0.01],'lr_method':'linear','batch_size':768, 'image_size':(128, 128), 'data_dir':'160', 'prefix':'train'},
    {'epoch':[15,32], 'lr': [0.2,0.002],'lr_method':'exp','batch_size':256, 'image_size':(224, 224), 'data_dir':'320', 'prefix':'train'},
    {'epoch':[32,37], 'lr': [0.003,0.0005],'lr_method':'linear','batch_size':128, 'image_size':(288, 288), 'data_dir':'320', 'prefix':'train'}
    ]
    
    training_strategy = [
    {'epoch':[0,6], 'lr': [1.0,2.0],'lr_method':'linear','batch_size':740, 'image_size':(128, 128), 'data_dir':'160', 'prefix':'train'},
    {'epoch':[6,21], 'lr': [2.0,0.45],'lr_method':'linear','batch_size':740, 'image_size':(128, 128), 'data_dir':'160', 'prefix':'train'},
    {'epoch':[21,32], 'lr': [0.45,0.02],'lr_method':'exp','batch_size':256, 'image_size':(224, 224), 'data_dir':'320', 'prefix':'train'},
    {'epoch':[32,36], 'lr': [0.02,0.004],'lr_method':'exp','batch_size':196, 'image_size':(224, 224), 'data_dir':'320', 'prefix':'train'},
    {'epoch':[36,40], 'lr': [0.004,0.002],'lr_method':'exp','batch_size':128, 'image_size':(288, 288), 'data_dir':'320', 'prefix':'train'}
    ]
    
    num_training_samples= 1281167
    num_eval_samples = 50000
    
    
    cmdline = add_cli_args()
    FLAGS, unknown_args = cmdline.parse_known_args()
    
    
   
    do_checkpoint = hvd.rank() == 0
    
   
    barrier = hvd.allreduce(tf.constant(0, dtype=tf.float32))
    tf.Session(config=config).run(barrier)

    if hvd.local_rank() == 0 and not os.path.isdir(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    
    barrier = hvd.allreduce(tf.constant(0, dtype=tf.float32))
    tf.Session(config=config).run(barrier)

    
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)  # INFO, ERROR
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(FLAGS.log_dir, log_name))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(fh)
        
    if not FLAGS.save_checkpoints_steps:
        # default to save one checkpoint per epoch
        FLAGS.save_checkpoints_steps = 625
    if not FLAGS.save_summary_steps:
        # default to save one checkpoint per epoch
        FLAGS.save_summary_steps = 625
    
    data_strategy, lr_strategy = dynamicpipe.lr_strategy_parsing(training_strategy, num_training_samples, FLAGS.num_gpus)
    
    num_steps = lr_strategy[-1]['steps'][-1] + FLAGS.display_every
    
    rank0log(logger, 'Data strategy: ' + str(data_strategy))
    rank0log(logger, 'Learning rate strategy:' + str(lr_strategy))
    rank0log(logger, 'Total Max Training Steps: ' + str(num_steps))
    rank0log(logger, 'Checkpointing every ' + str(FLAGS.save_checkpoints_steps) + ' steps')
    rank0log(logger, 'Saving summary every ' + str(FLAGS.save_summary_steps) + ' steps')

   
    rank0log(logger, 'PY' + str(sys.version) + 'TF' + str(tf.__version__))
    rank0log(logger, "Horovod size: ", hvd.size())

    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_function,
        model_dir=FLAGS.log_dir,
        params={
            'n_classes': 1000,
            'mom': FLAGS.mom,
            'num_steps': num_steps,
            'wdecay': FLAGS.wdecay,
            'loss_scale': FLAGS.loss_scale,
            'num_training_samples': num_training_samples,
            'lr_strategy':lr_strategy
        },
        config=tf.estimator.RunConfig(
            session_config=config,
            save_summary_steps=FLAGS.save_summary_steps if do_checkpoint else None,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps if do_checkpoint else None,
            keep_checkpoint_max=None))

    num_preproc_threads = 6
    rank0log(logger, "Using preprocessing threads per GPU: ", num_preproc_threads)
    training_hooks = [hvd.BroadcastGlobalVariablesHook(0), PrefillStagingAreasHook()]
    if hvd.rank() == 0:
        training_hooks.append(
                LogSessionRunHook(num_training_samples,FLAGS.num_gpus,
                                  FLAGS.display_every, logger))
    start_time = time.time()
    classifier.train(
            input_fn=lambda: dynamicpipe.data_pipeline(num_training_samples, FLAGS.num_gpus, data_strategy, FLAGS.data_dir, mode = "TRAINING"),
            max_steps=num_steps,
            hooks=training_hooks)
    rank0log(logger, "Log: Finished in ", time.time() - start_time)
    
    rank0log(logger, "Log: Evaluating")
    rank0log(logger, "Log: Validation dataset size: 50000")
    eval_strategy = [{'epoch':1, 'batch_size':128, 'image_size':(288, 288), 'data_dir':'320', 'prefix':'validation'}]
    
    #evaluation on single GPU
    #if hvd.rank() == 0:
    rank0log(logger, ' step  top1    top5     loss   checkpoint_time(UTC)')
    ckpts = sort_and_load_ckpts(FLAGS.log_dir)
    for i, c in enumerate(ckpts):
        if hvd.rank() == i % FLAGS.num_gpus:
            eval_result = classifier.evaluate(
                input_fn=lambda: dynamicpipe.data_pipeline(num_eval_samples, 1, eval_strategy, FLAGS.data_dir, mode = "EVAL"),
                checkpoint_path=c['path'])
            c['epoch'] = i
            c['top1'] = eval_result['val-top1acc']
            c['top5'] = eval_result['val-top5acc']
            c['loss'] = eval_result['loss']
            logger.info('Log @eval: count@{:5d} step@{:5d} top1@{:5.3f} top5@{:6.2f} loss@{:6.2f} time@{time}'
                     .format(c['epoch'],
                             c['step'],
                             c['top1'] * 100,
                             c['top5'] * 100,
                             c['loss'],
                             time=time.strftime('%Y-%m-%d %H:%M:%S',
                                time.localtime(c['mtime']))))
            
        
      
    rank0log(logger, "Log Finished evaluation") 

if __name__ == '__main__':
    main()

