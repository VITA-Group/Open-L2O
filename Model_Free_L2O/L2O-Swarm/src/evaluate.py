
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import monitored_session as ms

import meta
import util
import os
import pickle
import pdb
flags = tf.flags
logging = tf.logging


FLAGS = flags.FLAGS
flags.DEFINE_string("optimizer", "L2L", "Optimizer.")
flags.DEFINE_string("path", None, "Path to saved meta-optimizer network.")
flags.DEFINE_integer("num_epochs", 10, "Number of evaluation epochs.")
flags.DEFINE_integer("seed", None, "Seed for TensorFlow's RNG.")

flags.DEFINE_string("problem", "simple", "Type of problem.")
flags.DEFINE_integer("num_steps", 100,
                     "Number of optimization steps per epoch.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")

def main(_):
  # Configuration.
  num_unrolls = FLAGS.num_steps

  if FLAGS.seed:
    tf.set_random_seed(FLAGS.seed)

  # Problem.
  problem, net_config, net_assignments = util.get_config(FLAGS.problem,
                                                         FLAGS.path)

  # Optimizer setup.
  if FLAGS.optimizer == "Adam":
    cost_op = problem()
    problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    problem_reset = tf.variables_initializer(problem_vars)

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
    update = optimizer.minimize(cost_op)
    reset = [problem_reset, optimizer_reset]
  elif FLAGS.optimizer == "L2L":
    if FLAGS.path is None:
      logging.warning("Evaluating untrained L2L optimizer")
    optimizer = meta.MetaOptimizer(**net_config)
    meta_loss = optimizer.meta_loss(problem, 1, net_assignments=net_assignments, model_path = FLAGS.path)
    loss, update, reset, cost_op, x_final, constant = meta_loss
  else:
    raise ValueError("{} is not a valid optimizer".format(FLAGS.optimizer))
  with ms.MonitoredSession() as sess:
    # Prevent accidental changes to the graph.
    tf.get_default_graph().finalize()
    min_loss_record = []
    all_time_loss_record = []
    total_time = 0
    total_cost = 0
#    pdb.set_trace()
#    print(constant)
    x_record = [[sess.run(item) for item in x_final]]
    for _ in xrange(FLAGS.num_epochs):
      # Training.
      time, cost,  constants = util.eval_run_epoch(sess, cost_op, [update], reset,
                                  num_unrolls, x_final, constant)
      total_time += time
      total_cost += min(cost)
      all_time_loss_record.append(cost)
      min_loss_record.append(min(cost))
#      pdb.set_trace
#      print(x_finals)
      #x_record = x_record + x_finals
    with open('./{}/evaluate_record.pickle'.format(FLAGS.path),'wb') as l_record:
      record = {'all_time_loss_record':all_time_loss_record,'min_loss_record':min_loss_record,\
                'constants':[sess.run(item) for item in constants],\
                }
      pickle.dump(record, l_record)
    # Results.
    util.print_stats("Epoch {}".format(FLAGS.num_epochs), total_cost,
                     total_time, FLAGS.num_epochs)


if __name__ == "__main__":
  tf.app.run()
