import time
import math
import argparse
import utils, model
import numpy as np
import tensorflow as tf
from datetime import datetime

# parameters
dataset = 'PeMSD3'
parser = argparse.ArgumentParser()
parser.add_argument('--T', type = int, default = 288, help = 'number of time slots per day')
parser.add_argument('--P', type = int, default = 12, help = 'length of historical time steps')
parser.add_argument('--Q', type = int, default = 12, help = 'length of prediction time steps')
parser.add_argument('--delta_pdf', type = float, default = 0.5, help = 'delta_pdf')
parser.add_argument('--delta_adt', type = float, default = 0.5, help = 'delta_adt')
parser.add_argument('--K', type = int, default = 2, help = 'kernel size')
parser.add_argument('--beta', type = float, default = 0.1, help = 'beta')
parser.add_argument('--d', type = int, default = 64, help = 'hidden dims')
parser.add_argument('--train_ratio', type = int, default = 0.6, help = 'train/val/test')
parser.add_argument('--val_ratio', type = int, default = 0.2, help = 'train/val/test')
parser.add_argument('--test_ratio', type = int, default = 0.2, help = 'train/val/test')
parser.add_argument('--batch_size', type = int, default = 64, help = 'batch size')
parser.add_argument('--epochs', type = int, default = 200, help = 'epoches to run')
parser.add_argument('--lr', type=float, default = 0.001, help = 'initial learning rate')
parser.add_argument('--dist_file', type = str, default = './data/dist(' + dataset + ').csv', help = 'dist file')
parser.add_argument('--data_file', type = str, default = './data/' + dataset + '.npz', help = 'data file')
parser.add_argument('--model_file', type = str, default = './model/STJGCN_' + dataset, help = 'model file')
parser.add_argument('--log_file', type = str, default = './logs/log_' + dataset, help = 'log file')
args = parser.parse_args()

start = time.time()
log = open(args.log_file, 'w')
utils.log_string(log, 'training on the %s dataset' % dataset)

# load data
utils.log_string(log, 'loading data...')
(x_train, y_train, t_train, x_val, y_val, t_val, x_test, y_test, t_test, mean, std, A_fw, A_bw) = utils.load_data(args)
utils.log_string(log, 'x_train: %s\ty_train: %s' % (x_train.shape, y_train.shape))
utils.log_string(log, 'x_val:   %s\ty_val:   %s' % (x_val.shape, y_val.shape))
utils.log_string(log, 'x_test:  %s\ty_test:  %s' % (x_test.shape, y_test.shape))
utils.log_string(log, 'data loaded!')

# train model
utils.log_string(log, 'compling model...')
num_train, _, N, C = x_train.shape
x, t, label, is_training = model.placeholder(args.P, args.Q, N, C)
pred = model.STJGCN(x, t, A_fw, A_bw, args.Q, args.T, args.d, args.K, args.delta_adt, is_training)
pred = pred * std + mean
loss = model.loss_func(pred, label, args.beta)
tf.compat.v1.add_to_collection('pred', pred)
tf.compat.v1.add_to_collection('loss', loss)
optimizer = tf.compat.v1.train.AdamOptimizer(args.lr)
global_step = tf.Variable(0, trainable = False)
train_op = optimizer.minimize(loss, global_step = global_step)
parameters = 0
for variable in tf.compat.v1.trainable_variables():
    parameters += np.product([x.value for x in variable.get_shape()])
utils.log_string(log, 'total trainable parameters: {:,}'.format(parameters))
utils.log_string(log, 'model compiled!')
saver = tf.compat.v1.train.Saver()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)
sess.run(tf.compat.v1.global_variables_initializer())
utils.log_string(log, '**** training model ****')
train_time, val_time = [], []
val_loss_min = np.inf
for epoch in range(args.epochs):
    # shuffle
    permutation = np.random.permutation(num_train)
    x_train, t_train, y_train = x_train[permutation], t_train[permutation], y_train[permutation]
    # train loss
    train_loss = []
    num_batch = math.ceil(num_train / args.batch_size)
    t1 = time.time()
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            x: x_train[start_idx : end_idx],
            t: t_train[start_idx : end_idx],
            label: y_train[start_idx : end_idx],
            is_training: True}
        _, loss_batch = sess.run([train_op, loss], feed_dict = feed_dict)
        train_loss.append(loss_batch)
    t2 = time.time()
    train_time.append(t2 - t1)
    train_loss = np.mean(train_loss)     
    # val loss
    num_val = x_val.shape[0]
    val_loss = []
    num_batch = math.ceil(num_val / args.batch_size)
    t1 = time.time()
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            x: x_val[start_idx : end_idx],
            t: t_val[start_idx : end_idx],
            label: y_val[start_idx : end_idx],
            is_training: False}
        loss_batch = sess.run(loss, feed_dict = feed_dict)
        val_loss.append(loss_batch)
    t2 = time.time()
    val_time.append(t2 - t1)
    val_loss = np.mean(val_loss)
    utils.log_string(
        log, '%s | epoch: %03d/%d, train_time: %.2fs, train_loss: %.2f, val_time: %.2fs, val_loss: %.2f' %
        (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1, args.epochs, train_time[epoch], train_loss, val_time[epoch], val_loss))
    if val_loss <= val_loss_min:
        utils.log_string(
            log, 'val loss decrease from %.2f to %.2f, saving model to %s' %
            (val_loss_min, val_loss, args.model_file))
        val_loss_min = val_loss
        saver.save(sess, args.model_file)
utils.log_string(
    log, 'training finished, average train time: %.2fs, average val time: %.2fs, min val loss: %.2f' %
    (np.mean(train_time), np.mean(val_time), val_loss_min))

# test model
utils.log_string(log, '**** testing model ****')
utils.log_string(log, 'loading model from %s' % args.model_file)
saver = tf.compat.v1.train.import_meta_graph(args.model_file + '.meta')
saver.restore(sess, args.model_file)
utils.log_string(log, 'model restored!')
utils.log_string(log, 'evaluating...')
pred_test = []
num_test = x_test.shape[0]
num_batch = math.ceil(num_test / args.batch_size)
t1 = time.time()
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
    feed_dict = {
        x: x_test[start_idx : end_idx],
        t: t_test[start_idx : end_idx],
        is_training: False}
    pred_batch = sess.run(pred, feed_dict = feed_dict)
    pred_test.append(pred_batch)
t2 = time.time()
utils.log_string(log, 'test time: %.2fs' % (t2 - t1))
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
utils.log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
log.close()
sess.close()
