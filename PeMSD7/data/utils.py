import math
import numpy as np
import scipy.sparse as sp

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def metric(pred, label):
    mask = label > 0.
    pred, label = pred[mask], label[mask]
    mae = np.mean(np.abs(pred - label))
    rmse = np.sqrt(np.mean((pred - label) ** 2))
    mape = np.mean(np.abs(pred - label) / label)
    return mae, rmse, mape

def normalized_laplacian(A):
    '''
    A = D^(-1/2)*A*D^(-1/2)
    '''
    D = np.sum(A, axis = 1)
    D = np.diag(D ** -0.5)
    A = np.matmul(np.matmul(D, A), D)
    return A
    
def seq2instance(seq, P, Q):
    num_instance = len(seq) - P - Q + 1
    x = np.zeros(shape = ((num_instance, P) + seq.shape[1 :]))
    y = np.zeros(shape = ((num_instance, Q) + seq.shape[1 :]))
    for i in range(num_instance):
        x[i] = seq[i : i + P]
        y[i] = seq[i + P : i + P + Q]
    return x, y

def load_data(args):
    # pre_defined adjacency matrix
    x = np.load(args.data_file)['data']
    num_interval, N, _ = x.shape
    dist = np.loadtxt(args.dist_file, delimiter = ',', skiprows = 1)
    dist_mx = np.zeros(shape = (N, N))
    dist_mx[:] = np.inf
    for row in dist:
        dist_mx[int(row[0]), int(row[1])] = row[2]
    std = np.std(dist_mx[~np.isinf(dist_mx)])
    A_fw, A_bw = [], []
    for i in range(args.P):
        A = np.exp(-((i + 1) * dist_mx) ** 2 / std ** 2)
        A += np.eye(N)
        A[A < args.delta_pdf] = 0
        A[A > 1] = 1
        A = A.astype(np.float32)
        A_fw.append(sp.coo_matrix(normalized_laplacian(A)))
        A_bw.append(sp.coo_matrix(normalized_laplacian(A.T)))
    # x
    num_train = round(args.train_ratio * num_interval)
    num_test = round(args.test_ratio * num_interval)
    num_val = num_interval - num_train - num_test
    mean = np.mean(x[: num_train], axis = (0, 1))
    std = np.std(x[: num_train], axis = (0, 1))
    # time feature
    num_day = math.ceil(num_interval / args.T)
    num_week = math.ceil(num_day / 7)
    dayofweek = np.repeat(np.arange(7), repeats = args.T, axis = 0)
    dayofweek = np.repeat(np.reshape(dayofweek, newshape = (1, -1)),
                          repeats = num_week, axis = 0)
    dayofweek = np.reshape(dayofweek, newshape = (-1, 1))
    dayofweek = dayofweek[: num_interval]
    timeofday = np.reshape(np.arange(args.T), newshape = (1, -1))
    timeofday = np.repeat(timeofday, repeats = num_day, axis = 0)
    timeofday = np.reshape(timeofday, newshape = (-1, 1))
    t = np.concatenate((dayofweek, timeofday), axis = -1)
    # train/val/test
    x_train, y_train = seq2instance(x[: num_train], args.P, args.Q)
    x_train = (x_train - mean) / std
    t_train, _ = seq2instance(t[: num_train], args.P, args.Q)
    x_val, y_val = seq2instance(x[num_train : num_train + num_val], args.P, args.Q)
    x_val = (x_val - mean) / std
    t_val, _ = seq2instance(t[num_train : num_train + num_val], args.P, args.Q)
    x_test, y_test = seq2instance(x[-num_test :], args.P, args.Q)
    x_test = (x_test - mean) / std
    t_test, _ = seq2instance(t[-num_test :], args.P, args.Q)
    return (x_train, y_train[..., 0], t_train, x_val, y_val[..., 0], t_val,
            x_test, y_test[..., 0], t_test, mean[0], std[0], A_fw, A_bw)
