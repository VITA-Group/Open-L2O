import os
import numpy as np
import tensorflow.compat.v2 as tf

# _SHUFFLE_BUFFER = 51200


def dataset_parser(value, A):
  """Parse an ImageNet record from a serialized string Tensor."""

  # return value[:A.shape[0]], value[A.shape[0]:]
  return value[:A.shape[0]], value


def process_record_dataset(dataset,
                           is_training,
                           batch_size,
                           shuffle_buffer,
                           drop_remainder=False,
                           A=None):
  """Given a Dataset with raw records, return an iterator over the records.

  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup time
      and use less memory.
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.

  Returns:
    Dataset of labels ready for iteration.
  """
  if is_training:
    # Shuffles records before repeating to respect epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    # Repeats the dataset for the number of epochs to train.
    dataset = dataset.repeat()

    # Use a private thread pool and limit intra-op parallelism. Enable
    # non-determinism only for training.
    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_threading.private_threadpool_size = 16
    options.experimental_deterministic = False
    dataset = dataset.with_options(options)

  dataset = dataset.apply(
      tf.data.experimental.map_and_batch(
          lambda x: dataset_parser(x, A),
          batch_size=batch_size,
          num_parallel_batches=2,
          drop_remainder=drop_remainder))

  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return dataset


def input_fn(is_training,
             data_dir,
             batch_size,
             task='sc',
             input_context=None,
             drop_remainder=False,
             A=None,
             filename=None):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    input_context: A `tf.distribute.InputContext` object passed in by
      `tf.distribute.Strategy`.
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.

  Returns:
    A dataset that can be used for iteration.
  """
  if filename is None:
    if is_training:
      # filename = task+'_train_data.npy'
      filename = 'train_data.npy'
    else:
      # filename = task+'_val_data.npy'
      filename = 'val_data.npy'

  data = np.load(os.path.join(data_dir, filename), allow_pickle=True)
  shuffle_buffer = 400000 if task == 'cs' else data.shape[0]
  # dataset = tf.data.TFRecordDataset(os.path.join(data_dir, filename))
  dataset = tf.data.Dataset.from_tensor_slices(data)

  if input_context:
    dataset = dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)

  dataset = dataset.cache()

  return process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      batch_size=batch_size,
      # shuffle_buffer=_SHUFFLE_BUFFER if task == 'sc' else 400000,
      shuffle_buffer=shuffle_buffer,
      drop_remainder=drop_remainder,
      A=A)

