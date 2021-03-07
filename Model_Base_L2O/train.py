import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf

import data_preprocessing
import models
import utils

FLAGS = flags.FLAGS

# Model definition
flags.DEFINE_string('model_name', 'lista', '')
flags.DEFINE_integer('num_layers', 16, 'The number of layers.', lower_bound=1)
flags.DEFINE_float('model_lam', 0.4, 'The L1 regularization strength used for model initialization.')
flags.DEFINE_boolean('share_W', False, 'Share W over different layers.')
# Loss function
flags.DEFINE_enum('task', 'sc', ['sc', 'lasso', 'cs'],
        'The loss function used for training. Choose between '
        '`sc` (sparse coding), `lasso` (Lasso regression) and '
        '`cs` (compressive senssing).')
flags.DEFINE_float('lasso_lam', 0.005,
        'The L1 regularization strength in Lasso objective.')
# Alista settings
flags.DEFINE_string('alista_W_file', 'W.npy',
        'File that contains the analytically solved weight matrix.')
# Glista settings
flags.DEFINE_float('glista_alti', 5.0,
        'The alti parameter for the Glista models.')
flags.DEFINE_string('gain_func', 'relu',
        'The gain gate selection in Glista.')
# Tista settings
flags.DEFINE_float('tista_sigma2', None,
        'sigma2 for linear measurement noise for Tista.')
# Support selection settings
flags.DEFINE_float('ss_q_per_layer', '1.2',
        'Extra percentage of coordinates added to the support set in each layer.')
flags.DEFINE_float('ss_maxq', '13',
        'Maximum percentage of coordinates selected in the support set.')
# Exp settings
flags.DEFINE_integer('seed', 42, 'The RNG seed for the experiment.')
flags.DEFINE_float('base_lr', 0.0005, 'The base learning rate.')
flags.DEFINE_integer('epochs', 100000, 'The total number of epochs of the training process.')
flags.DEFINE_integer('num_train_images', 51200, 'The total number of training samples.')
flags.DEFINE_integer('num_val_images', 1024, 'The total number of validation samples.')
flags.DEFINE_integer('num_test_images', 1024, 'The total number of testing samples.')
flags.DEFINE_integer('train_batch_size', 128, 'The batch size for the training dataset.')
flags.DEFINE_integer('val_batch_size', 1024, 'The batch size for the validation dataset.')
flags.DEFINE_integer('test_batch_size', 1024, 'The batch size for the testing dataset.')
flags.DEFINE_multi_string('test_files', [], 'Files that are used for testing')
flags.DEFINE_boolean('test', False, 'Flag that indicates testing will be done')
# Exp saving and logging
flags.DEFINE_string('base_dir', None, 'Base experiment directory.')
flags.DEFINE_string('data_dir', None, 'Data directory for the experiment.')
flags.DEFINE_string('exp_name', 'lista_sc', 'The name of the experiment.')
flags.DEFINE_integer('replicate', 1, 'The replicate id of the experiment.')


def run(
    model_name,
    base_dir,
    data_dir,
    task='sc',
    train_batch_size=128,
    eval_batch_size=1024,
    epochs=125,
    mode='train'
):

  _NUM_TRAIN_IMAGES = FLAGS.num_train_images
  _NUM_EVAL_IMAGES  = FLAGS.num_val_images

  if task == 'sc' or task == 'lasso':
    training_steps_per_epoch = int(_NUM_TRAIN_IMAGES // train_batch_size)
    validation_steps_per_epoch = int(_NUM_EVAL_IMAGES // eval_batch_size)
  elif task == 'cs':
    training_steps_per_epoch = 3125
    validation_steps_per_epoch = 10

  _BASE_LR = FLAGS.base_lr

  # Deal with paths of data, checkpoints and logs
  base_dir = os.path.abspath(base_dir)
  model_dir = os.path.join(base_dir, 'models', FLAGS.exp_name,
          'replicate_' + str(FLAGS.replicate))
  log_dir = os.path.join(base_dir, 'logs', FLAGS.exp_name,
          'replicate_' + str(FLAGS.replicate))
  logging.info('Saving checkpoints at %s', model_dir)
  logging.info('Saving tensorboard summaries at %s', log_dir)
  logging.info('Use training batch size: %s.', train_batch_size)
  logging.info('Use eval batch size: %s.', eval_batch_size)
  logging.info('Training model using data_dir in directory: %s', data_dir)

  if task == 'sc' or task == 'lasso':
    A = np.load(
        os.path.join(data_dir, 'A.npy'),
        allow_pickle=True).astype(np.float32)
    M, N = A.shape
    F = None
    D = None
  elif task == 'cs':
    A = np.load(
        os.path.join(data_dir, 'A_128_512.npy'),
        allow_pickle=True).astype(np.float32)
    D = np.load(
        os.path.join(data_dir, 'D_256_512.npy'),
        allow_pickle=True).astype(np.float32)
    N = D.shape[0]
    F = D.shape[1]
  else:
    raise ValueError('invalid task type')

  if FLAGS.model_name.startswith('alista'):
    alista_W = np.load(
        os.path.join(data_dir, 'W.npy'),
        allow_pickle=True).astype(np.float32)

  np.random.seed(FLAGS.seed)

  if mode == 'train':
      train_dataset = data_preprocessing.input_fn(
          True,
          data_dir,
          train_batch_size,
          task,
          drop_remainder=False,
          A=A)
      val_dataset = data_preprocessing.input_fn(
          False,
          data_dir,
          eval_batch_size,
          task,
          drop_remainder=False,
          A=A)

  summary_writer = tf.summary.create_file_writer(log_dir)

  # Define a Lista model
  if FLAGS.model_name == 'lista':
    model = models.Lista(
        A, FLAGS.num_layers, FLAGS.model_lam, FLAGS.share_W, D, name='Lista'
    )
    output_interval = N
  elif FLAGS.model_name == 'lfista':
    model = models.Lfista(
        A, FLAGS.num_layers, FLAGS.model_lam, FLAGS.share_W, D, name='Lfista'
    )
    output_interval = N
  elif FLAGS.model_name == 'lamp':
    model = models.Lamp(
        A, FLAGS.num_layers, FLAGS.model_lam, FLAGS.share_W, D, name='Lamp'
    )
    output_interval = M + N
  elif FLAGS.model_name == 'step_lista':
    assert FLAGS.model_lam == FLAGS.lasso_lam
    model = models.StepLista(
        A, FLAGS.num_layers, FLAGS.lasso_lam, D, name='StepLista'
    )
    output_interval = N
  elif FLAGS.model_name == 'lista_cp':
    model = models.ListaCp(
        A, FLAGS.num_layers, FLAGS.model_lam, FLAGS.share_W, D, name='ListaCp'
    )
    output_interval = N
  elif FLAGS.model_name == 'lista_cpss':
    model = models.ListaCpss(
        A, FLAGS.num_layers, FLAGS.model_lam,
        FLAGS.ss_q_per_layer, FLAGS.ss_maxq,
        FLAGS.share_W, D, name='ListaCpss'
    )
    output_interval = N
  elif FLAGS.model_name == 'alista':
    model = models.Alista(
        A, alista_W, FLAGS.num_layers, FLAGS.model_lam,
        FLAGS.ss_q_per_layer, FLAGS.ss_maxq,
        D, name='Alista'
    )
    output_interval = N
  elif FLAGS.model_name == 'glista':
    model = models.Glista(
        A, FLAGS.num_layers, FLAGS.model_lam,
        FLAGS.ss_q_per_layer, FLAGS.ss_maxq,
        FLAGS.share_W, D, name='Glista',
        alti=FLAGS.glista_alti, gain_func=FLAGS.gain_func
    )
    output_interval = N * 2
  elif FLAGS.model_name == 'tista':
    assert FLAGS.tista_sigma2 is not None
    model = models.Tista(
        A, FLAGS.num_layers, FLAGS.model_lam, FLAGS.tista_sigma2,
        FLAGS.share_W, D, name='Tista'
    )
    output_interval = N
  else:
    raise NotImplementedError('Other types of models not are not implemented yet')

  var_list = {}
  checkpoint = tf.train.Checkpoint(model=model)
  prev_model, prev_layer = utils.check_and_load_partial(model_dir, FLAGS.num_layers)

  if task == 'lasso':
    _A_const = tf.constant(A, name='A_lasso_const')
    loss = utils.LassoLoss(_A_const, FLAGS.lasso_lam, N, F)
    metrics_compile = [utils.LassoObjective('lasso', _A_const, FLAGS.lasso_lam, M, N, -1)]
    monitor = 'val_lasso'
  else:
    loss = utils.MSE(N, F)
    metrics_compile = [utils.NMSE('nmse', N)]
    monitor = 'val_nmse'

  if mode == 'test':
    if prev_layer != FLAGS.num_layers:
      raise ValueError('Should have a fully trained model!')
    checkpoint.restore(prev_model).assert_existing_objects_matched()
    res_dict = {}
    eval_files = FLAGS.test_files
    if task == 'lasso':
      # do layer-wise testing
      test_metrics = [
        utils.LassoObjective('lasso_layer{}'.format(i), _A_const, FLAGS.lasso_lam, M, N, i)
        for i in range(FLAGS.num_layers)
      ]
    else:
      test_metrics = [
          utils.EvalNMSE('nmse_layer{}'.format(i), M, N, output_interval, i)
          for i in range(FLAGS.num_layers)
      ]
    for layer_id in range(FLAGS.num_layers):
      model.create_cell(layer_id)
    for i in range(len(eval_files)):
      val_ds = data_preprocessing.input_fn(
          False,
          data_dir,
          eval_batch_size,
          drop_remainder=False,
          A=A,
          filename=eval_files[i])
      logging.info('Compiling model.')
      model.compile(
          optimizer=tf.keras.optimizers.Adam(_BASE_LR),
          loss=loss,
          metrics=test_metrics)
      metrics = model.evaluate(
          x=val_ds,
          verbose=2)
      if task == 'lasso':
        output = model.predict(
            x=val_ds,
            verbose=2)
        final_xh = output[:, -N:]
        eval_file_basename = os.path.basename(eval_files[i]).strip('.npy')
        np.save(os.path.join(model_dir, eval_file_basename + '_final_output.npy'), final_xh)
      res_dict[eval_files[i]] = metrics[1:]
    for k,v in res_dict.items():
        logging.info('%s : %s', k, str(v))
    return

  for layer_id in range(FLAGS.num_layers):
    logging.info('Building Lista Keras model.')
    model.create_cell(layer_id)

    # Deal with the variables that have been trained in previous layers
    for name in var_list:
      var_list[name] += 1
    # Deal with the variables in the current layer
    for v in model.layers[layer_id].trainable_variables:
      if v.name not in var_list:
        var_list[v.name] = 0

    if layer_id == prev_layer - 1 and prev_model:
      checkpoint.restore(prev_model).assert_existing_objects_matched()
      logging.info('Checkpoint restored from %s.', prev_model)
      model.compile(
          optimizer=tf.keras.optimizers.Adam(_BASE_LR),
          loss=loss,
          metrics=metrics_compile)
      metrics = model.evaluate(
          x=val_dataset,
          verbose=2)
      val_metric = metrics[1]
      with summary_writer.as_default():
        for value, key in zip(metrics, model.metrics_names):
          tf.summary.scalar(key, value, layer_id + 1)
      continue
    elif layer_id < prev_layer:
      logging.info('Skip layer %d.', layer_id + 1)
      continue

    logging.info('Compiling model.')

    model.compile(
        optimizer=utils.Adam(var_list, True, learning_rate=_BASE_LR),
        loss=loss,
        metrics=metrics_compile)

    earlystopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        min_delta=0,
        patience=5,
        mode='min',
        restore_best_weights=False)
    cbs = [earlystopping_cb]

    logging.info('Fitting Lista Keras model.')
    model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=training_steps_per_epoch,
        callbacks=cbs,
        validation_data=val_dataset,
        validation_steps=validation_steps_per_epoch,
        verbose=2)
    logging.info('Finished fitting Lista Keras model.')
    model.summary()

    for i in range(2):
      logging.info('Compiling model.')
      model.compile(
          optimizer=utils.Adam(var_list, learning_rate=_BASE_LR * 0.2 * 0.1**i),
          loss=loss,
          metrics=metrics_compile)

      earlystopping_cb = tf.keras.callbacks.EarlyStopping(
          monitor=monitor,
          min_delta=0,
          patience=5,
          mode='min',
          restore_best_weights=False)
      cbs = [earlystopping_cb]

      logging.info('Fitting Lista Keras model.')
      history = model.fit(
          train_dataset,
          epochs=epochs,
          steps_per_epoch=training_steps_per_epoch,
          callbacks=cbs,
          validation_data=val_dataset,
          validation_steps=validation_steps_per_epoch,
          verbose=2)
      logging.info('Finished fitting Lista Keras model.')
    model.summary()
    val_metric = history.history[monitor][-1]
    with summary_writer.as_default():
      for key in history.history.keys():
        tf.summary.scalar(key, history.history[key][-1], layer_id + 1)
    try:
      checkpoint.save(utils.save_partial(model_dir, layer_id))
      logging.info('Checkpoint saved at %s', utils.save_partial(model_dir, layer_id))
    except tf.errors.NotFoundError:
      pass

  if task == 'cs':
    raise NotImplementedError('Compressive sensing testing part not implemented yet')
    data = np.load(
        os.path.join(data_dir, 'set11.npy'),
        allow_pickle=True)
    phi = np.load(
        os.path.join(data_dir, 'phi_128_256.npy'),
        allow_pickle=True).astype(np.float32)
    psnr = utils.PSNR()
    for im in data:
      im_ = im.astype(np.float32)
      cols = utils.im2cols(im_)
      patch_mean = np.mean(cols, axis=1, keepdims=True)
      fs = ((cols - patch_mean) / 255.0).astype(np.float32)
      ys = np.matmul(fs, phi.transpose())
      fs_rec = model.predict_on_batch(ys)[:, -N:]
      cols_rec = fs_rec * 255.0 + patch_mean
      im_rec = utils.col2im(cols_rec).astype(np.float32)
      psnr.update_state(im_, im_rec)
    logging.info('Test PSNR: %f', psnr.result().numpy())
    val_metric = float(psnr.result().numpy())
    with summary_writer.as_default():
      tf.summary.scalar('test_psnr', psnr.result().numpy(), 0)

  return val_metric


def main(_):
  base_dir = FLAGS.base_dir
  data_dir = FLAGS.data_dir
  mode = 'test' if FLAGS.test else 'train'
  run(
    FLAGS.model_name,
    base_dir,
    data_dir,
    task=FLAGS.task,
    train_batch_size=FLAGS.train_batch_size,
    eval_batch_size=FLAGS.val_batch_size,
    epochs=FLAGS.epochs,
    mode=mode
  )


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)

