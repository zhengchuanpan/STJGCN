import numpy as np
import tensorflow as tf
from tf_utils import conv2d

def placeholder(P, Q, N, C):
    x = tf.compat.v1.placeholder(
        shape = (None, P, N, C), dtype = tf.float32, name = 'x')
    t = tf.compat.v1.placeholder(
        shape = (None, P, 2), dtype = tf.int32, name = 't')
    label = tf.compat.v1.placeholder(
        shape = (None, Q, N), dtype = tf.float32, name = 'label')
    is_training = tf.compat.v1.placeholder(
        shape = (), dtype = tf.bool, name = 'is_training')
    return x, t, label, is_training

def FC(x, units, activations, use_bias, is_training):
    if isinstance(units, int):
        units = [units]
        activations = [activations]
    for unit, activation in zip(units, activations):
        x = conv2d(
            x, output_dims = unit, kernel_size = [1, 1], stride = [1, 1],
            padding = 'VALID', activation = activation,
            use_bias = use_bias, is_training = is_training)
    return x

def Embedding(t, N, T, d, is_training):
    '''
    t: [None, P, 2]
    '''
    SE = tf.Variable(
        tf.glorot_uniform_initializer()(shape = [1, 1, N, d]),
        dtype = tf.float32, trainable = True)
    SE = FC(
        SE, units = [d, d], activations = [tf.nn.relu, None],
        use_bias = True, is_training = is_training)
    dayofweek = tf.one_hot(t[..., 0], depth = 7)
    timeofday = tf.one_hot(t[..., 1], depth = T)
    TE = tf.concat((dayofweek, timeofday), axis = -1)
    TE = tf.expand_dims(TE, axis = 2)
    TE = FC(
        TE, units = [d, d], activations = [tf.nn.relu, None],
        use_bias = True, is_training = is_training)
    U = tf.add(SE, TE)
    U = FC(
        U, units = [d, d], activations = [tf.nn.relu, None],
        use_bias = True, is_training = is_training)
    return U
    
def GC2(x, A, use_bias, is_training):
    '''
    x: [None, P, N, d] 
    A: [N, N] 2d normalized laplacian matrix
    '''
    P = x.get_shape()[1].value
    N = x.get_shape()[2].value
    d = x.get_shape()[3].value
    A = tf.SparseTensor(indices = np.column_stack((A.row, A.col)),
                        values = A.data, dense_shape = A.shape)
    z = tf.reshape(tf.transpose(x, perm = (2, 0, 1, 3)), shape = (N, -1))
    z = tf.reshape(tf.sparse.sparse_dense_matmul(A, z), shape = (N, -1, P, d))
    z = tf.transpose(z, perm = (1, 2, 0, 3))
    z = FC(
        z, units = d, activations = tf.nn.relu,
        use_bias = use_bias, is_training = is_training)
    return z

def GC4(x, L, use_bias, is_training):
    '''
    x: [None, P, N, d] 
    L: [None, P, N, N] 4d normalized laplacian matrix
    '''
    d = x.get_shape()[3].value
    z = tf.matmul(L, x)
    z = FC(
        z, units = d, activations = tf.nn.relu,
        use_bias = use_bias, is_training = is_training)   
    return z

def STJGC_pdf(x, A_fw, A_bw, is_training):
    K = len(x)
    assert len(A_fw) == K and len(A_bw) == K
    z = []
    for k in range(K - 1):
        z_fw = GC2(x[k], A_fw[k], use_bias = False, is_training = is_training)
        z_bw = GC2(x[k], A_bw[k], use_bias = False, is_training = is_training)
        z.append(tf.add(z_fw, z_bw))
    z_fw = GC2(x[-1], A_fw[-1], use_bias = False, is_training = is_training)
    z_bw = GC2(x[-1], A_bw[-1], use_bias = True, is_training = is_training)
    z.append(tf.add(z_fw, z_bw))
    z = tf.math.add_n(z)
    return z

def STJGC_adt(x, U, B, delta_adt, is_training):
    K = len(x)
    assert len(U) == K
    z = []
    for k in range(K - 1):
        L1 = tf.matmul(
            tf.nn.conv2d(U[k], B, strides = [1, 1, 1, 1], padding = 'SAME'),
            tf.transpose(U[-1], perm = (0, 1, 3, 2)))
        L2 = tf.matmul(
            tf.nn.conv2d(U[-1], B, strides = [1, 1, 1, 1], padding = 'SAME'),
            tf.transpose(U[k], perm = (0, 1, 3, 2)))
        L1 = tf.compat.v2.where(
            condition = tf.less(L1, delta_adt), x = 0., y = L1)
        L2 = tf.compat.v2.where(
            condition = tf.less(L2, delta_adt), x = 0., y = L2)
        L1 = tf.nn.softmax(L1)
        L2 = tf.nn.softmax(L2)
        z1 = GC4(x[k], L1, use_bias = False, is_training = is_training)
        z2 = GC4(x[k], L2, use_bias = False, is_training = is_training)
        z.append(tf.add(z1, z2))
    L3 = tf.matmul(
        tf.nn.conv2d(U[-1], B, strides = [1, 1, 1, 1], padding = 'SAME'),
        tf.transpose(U[-1], perm = (0, 1, 3, 2)))
    L3 = tf.compat.v2.where(
        condition = tf.less(L3, delta_adt), x = 0., y = L3)
    L3 = tf.nn.softmax(L3)
    z3 = GC4(x[-1], L3, use_bias = True, is_training = is_training)
    z.append(z3)
    z = tf.math.add_n(z)
    return z

def STJGC(x, A_fw, A_bw, U, B, delta_adt, is_training):
    d = x[0].get_shape()[-1].value
    z_pdf = STJGC_pdf(x, A_fw, A_bw, is_training)
    z_adt = STJGC_adt(x, U, B, delta_adt, is_training)
    # gating fusion
    g1 = FC(
        z_pdf, units = d, activations = None,
        use_bias = False, is_training = is_training)
    g2 = FC(
        z_adt, units = d, activations = None,
        use_bias = True, is_training = is_training)    
    g = tf.nn.sigmoid(tf.add(g1, g2))
    z = tf.add(tf.multiply(g, z_pdf), tf.multiply(1 - g, z_adt))
    z = FC(
        z, units = [d, d], activations = [tf.nn.relu, None],
        use_bias = True, is_training = is_training)  
    return z, U[-1]

def Attention(x, is_training):
    '''
    x: [None, M, N, d]
    '''
    d = x.get_shape()[-1].value
    x = tf.transpose(x, perm = (0, 2, 1, 3))
    s = FC(
        x, units = d, activations = tf.nn.tanh,
        use_bias = True, is_training = is_training)
    s = FC(
        s, units = 1, activations = None,
        use_bias = False, is_training = is_training)
    s = tf.nn.softmax(tf.transpose(s, perm = (0, 1, 3, 2)))
    x = tf.matmul(s, x)
    x = FC(
        x, units = [d, d], activations = [tf.nn.relu, None],
        use_bias = True, is_training = is_training)     
    return x
    
def STJGCN(x, t, A_fw, A_bw, Q, T, d, K, delta_adt, is_training):
    '''
    x:     [None, P, N, C]
    t:     [None, P, 2]
    A:     [P, N, N]
    '''
    # spatio-temporal embedding
    P = x.get_shape()[1].value
    N = x.get_shape()[2].value
    U = Embedding(t, N, T, d, is_training)
    B = tf.Variable(
        tf.glorot_uniform_initializer()(shape = [1, 1, d, d]),
        dtype = tf.float32, trainable = True)
    # input
    x = FC(
        x, units = [d, d], activations = [tf.nn.relu, None],
        use_bias = True, is_training = is_training) 
    # dilated causal STJGC module
    assert K == 3
    # STJGC1  output: [2, 5, 8, 11]
    x_seq = [x[:, ::K], x[:, 1::K], x[:, 2::K]]  # [t - 2, t - 1, t]
    U_seq = [U[:, ::K], U[:, 1::K], U[:, 2::K]]
    A_fw_seq = [A_fw[2], A_fw[1], A_fw[0]]
    A_bw_seq = [A_bw[2], A_bw[1], A_bw[0]]
    z1, U = STJGC(x_seq, A_fw_seq, A_bw_seq, U_seq, B, delta_adt, is_training)
    z2 = tf.add(x[:, 2::K], z1) # residual connection
    # STJGC2  output: [8, 11]
    x_seq = [z2[:, :2], z2[:, 1:3], z2[:, 2:]]  
    U_seq = [U[:, :2], U[:, 1:3], U[:, 2:]]
    A_fw_seq = [A_fw[6], A_fw[3], A_fw[0]] 
    A_bw_seq = [A_bw[6], A_bw[3], A_bw[0]]
    z2, U = STJGC(x_seq, A_fw_seq, A_bw_seq, U_seq, B, delta_adt, is_training)
    z3 = tf.add(z1[:, 2:], z2)
    # STJGC3  output: [11]
    x_seq = [z3[:, :1], z3[:, 1:]]  
    U_seq = [U[:, :1], U[:, 1:]]
    A_fw_seq = [A_fw[3], A_fw[0]] 
    A_bw_seq = [A_bw[3], A_bw[0]]
    z3, _ = STJGC(x_seq, A_fw_seq, A_bw_seq, U_seq, B, delta_adt, is_training)
    # multi-range attention
    z = tf.concat((z1[:, -1:], z2[:, -1:], z3[:, -1:]), axis = 1)
    z = Attention(z, is_training) # [-1, N, 1, d]
    # prediction
    y = []
    for _ in range(Q):
        y_step = FC(
            z, units = [d, 1], activations = [tf.nn.relu, None],
            use_bias = True, is_training = is_training)
        y.append(y_step)
    y = tf.concat(y, axis = 2) # [-1, N, Q, 1]
    y = tf.transpose(y, perm = (0, 2, 1, 3))
    y = tf.squeeze(y)
    return y 

def loss_func(pred, label, beta, epsilon = 1.):
    # MAE + beta * MAPE * 100
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.compat.v2.where(condition = tf.math.is_nan(mask), x = 0., y = mask)
    mae = tf.abs(tf.subtract(pred, label))
    mape = 100 * mae / (label + epsilon)
    loss = mae + beta * mape
    loss *= mask
    loss = tf.compat.v2.where(condition = tf.math.is_nan(loss), x = 0., y = loss)
    loss = tf.reduce_mean(loss)
    return loss
