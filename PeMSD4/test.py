import time
import math
import utils
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime

# parameters
dataset = 'PeMSD4'
parser = argparse.ArgumentParser()
parser.add_argument('--T', type = int, default = 288, help = 'number of time slots per day')
parser.add_argument('--P', type = int, default = 12, help = 'length of historical time steps')
parser.add_argument('--Q', type = int, default = 12, help = 'length of prediction time steps')
parser.add_argument('--delta_pdf', type = float, default = 0.5, help = 'delta_pdf')
parser.add_argument('--train_ratio', type = int, default = 0.6, help = 'train/val/test')
parser.add_argument('--val_ratio', type = int, default = 0.2, help = 'train/val/test')
parser.add_argument('--test_ratio', type = int, default = 0.2, help = 'train/val/test')
parser.add_argument('--batch_size', type = int, default = 64, help = 'batch size')
parser.add_argument('--dist_file', type = str, default = './data/dist(' + dataset + ').csv', help = 'dist file')
parser.add_argument('--data_file', type = str, default = './data/' + dataset + '.npz', help = 'data file')
parser.add_argument('--model_file', type = str, default = './model/STJGCN_' + dataset, help = 'model file')
parser.add_argument('--log_file', type = str, default = './logs/log_' + dataset, help = 'log file')
args = parser.parse_args()

# parameters
start = time.time()
log = open(args.log_file, 'w')
utils.log_string(log, 'testing on the %s dataset' % (args.log_file[-6 :]))

# load data
utils.log_string(log, 'loading data...')
data = utils.load_data(args)
x_test, y_test, t_test = data[6 : 9]
utils.log_string(log, 'x_test:  %s\ty_test:  %s' % (x_test.shape, y_test.shape))
utils.log_string(log, 'data loaded!')

# test model
utils.log_string(log, '**** testing model ****')
utils.log_string(log, 'loading model from %s' % args.model_file)
graph = tf.Graph()
with graph.as_default():
    saver = tf.compat.v1.train.import_meta_graph(args.model_file + '.meta')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
with tf.compat.v1.Session(graph = graph, config = config) as sess:
    saver.restore(sess, args.model_file)
    parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        parameters += np.product([x.value for x in variable.get_shape()])
    utils.log_string(log, 'trainable parameters: {:,}'.format(parameters))
    pred = graph.get_collection(name = 'pred')[0]
    utils.log_string(log, 'model restored!')
    utils.log_string(log, 'evaluating...')
    pred_test = []
    num_test = x_test.shape[0]
    num_batch = math.ceil(num_test / args.batch_size)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            'x:0': x_test[start_idx : end_idx],
            't:0': t_test[start_idx : end_idx],
            'is_training:0': False}
        pred_batch = sess.run(pred, feed_dict = feed_dict)
        pred_test.append(pred_batch)
    pred_test = np.concatenate(pred_test, axis = 0)
# metric
utils.log_string(log, 'performance in each prediction step')
utils.log_string(log, '                  MAE\t\tRMSE\t\t MAPE')
MAE, RMSE, MAPE = [], [], []
for q in range(args.Q):
    mae, rmse, mape = utils.metric(pred_test[:, q], y_test[:, q])
    MAE.append(mae)
    RMSE.append(rmse)
    MAPE.append(mape)
    utils.log_string(log, 'step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
                     (q + 1, mae, rmse, mape * 100))
average_mae = np.mean(MAE)
average_rmse = np.mean(RMSE)
average_mape = np.mean(MAPE)
utils.log_string(log, 'average:         %.2f\t\t%.2f\t\t%.2f%%' %
                 (average_mae, average_rmse, average_mape * 100))
end = time.time()
utils.log_string(log, 'total time: %.2fs' % (end - start))
log.close()
sess.close()
