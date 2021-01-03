

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
from six.moves import xrange
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn import monitored_session as ms

import meta
import util

flags = tf.flags
logging = tf.logging


FLAGS = flags.FLAGS
flags.DEFINE_string("save_path", None, "Path for saved meta-optimizer.")
flags.DEFINE_integer("num_epochs", 10000, "Number of training epochs.")
flags.DEFINE_integer("log_period", 100, "Log period.")
flags.DEFINE_integer("evaluation_period", 1000, "Evaluation period.")
flags.DEFINE_integer("evaluation_epochs", 20, "Number of evaluation epochs.")

flags.DEFINE_string("problem", "simple", "Type of problem.")
flags.DEFINE_integer("num_steps", 100,
                     "Number of optimization steps per epoch.")
flags.DEFINE_integer("unroll_length", 20, "Meta-optimizer unroll length.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_boolean("second_derivatives", False, "Use second derivatives.")


def main(_):
  # Configuration.
  num_unrolls = FLAGS.num_steps // FLAGS.unroll_length
  problem, net_config, net_assignments = util.get_config(FLAGS.problem)
  optimizer = meta.MetaOptimizer(**net_config)
  if FLAGS.save_path is not None:
    if not os.path.exists(FLAGS.save_path):
      os.mkdir(FLAGS.save_path)
      path = None
#      raise ValueError("Folder {} already exists".format(FLAGS.save_path))
    else:
      if os.path.exists('{}/loss-record.pickle'.format(FLAGS.save_path)):
        path = FLAGS.save_path
      else:
        path = None
  # Problem.
  

  # Optimizer setup.
  
  minimize = optimizer.meta_minimize(
              problem, FLAGS.unroll_length,
              learning_rate=FLAGS.learning_rate,
              net_assignments=net_assignments,
              model_path = path,
              second_derivatives=FLAGS.second_derivatives)

  step, update, reset, cost_op, x_final, test, fc_weights, fc_bias, fc_va = minimize
#  saver=tf.train.Saver()
  with ms.MonitoredSession() as sess:
    # Prevent accidental changes to the graph.
    tf.get_default_graph().finalize()
#    Step=[step for i in range(len(cost_op))]
    best_evaluation = float("inf")
    total_time = 0
    total_cost = 0
    loss_record = []
    constants = []
    for e in xrange(FLAGS.num_epochs):
      # Training.
      time, cost, constant, Weights = util.run_epoch(sess, cost_op, [update, step], reset,
                                  num_unrolls, test, [fc_weights, fc_bias, fc_va])
      cost= sum(cost)/len(cost_op)
      total_time += time
      total_cost += cost
      loss_record.append(cost)
      constants.append(constant)
      # Logging.
      if (e + 1) % FLAGS.log_period == 0:
        util.print_stats("Epoch {}".format(e + 1), total_cost, total_time,
                         FLAGS.log_period)
        total_time = 0
        total_cost = 0

      # Evaluation.
      if (e + 1) % FLAGS.evaluation_period == 0:
        eval_cost = 0
        eval_time = 0
        for _ in xrange(FLAGS.evaluation_epochs):
          time, cost, constant, weights = util.run_epoch(sess, cost_op, [update, step], reset,
                                  num_unrolls, test, [fc_weights, fc_bias, fc_va])
#          cost/=len(cost_op)
          eval_time += time
          eval_cost += sum(cost)/len(cost_op)

        util.print_stats("EVALUATION", eval_cost, eval_time,
                         FLAGS.evaluation_epochs)

        if FLAGS.save_path is not None and eval_cost < best_evaluation:
          print("Removing previously saved meta-optimizer")
          for f in os.listdir(FLAGS.save_path):
            os.remove(os.path.join(FLAGS.save_path, f))
          print("Saving meta-optimizer to {}".format(FLAGS.save_path))
#          saver.save(sess,'./quadratic/quadratic.ckpt',global_step = e)
          optimizer.save(sess, FLAGS.save_path)
          with open(FLAGS.save_path+'/loss_record.pickle','wb') as l_record:
            record = {'loss_record':loss_record, 'fc_weights':sess.run(weights[0]), \
                'fc_bias':sess.run(weights[1]), 'fc_va':sess.run(weights[2]), 'constant':sess.run(constant)}
            pickle.dump(record, l_record)
          best_evaluation = eval_cost
#    fc_weights = np.array(sess.run(fc_weights))
    

if __name__ == "__main__":
  tf.app.run()
