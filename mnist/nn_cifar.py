import tensorflow as tf
import numpy as np
import pickle
import sys

#
# weights objective
#

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file, encoding='bytes')

def load_train_data():
    cifar_train1 = load_data('D:\\Box\\cifar\\data_batch_1')
    cifar_train2 = load_data('D:\\Box\\cifar\\data_batch_2')
    cifar_train3 = load_data('D:\\Box\\cifar\\data_batch_3')
    cifar_train4 = load_data('D:\\Box\\cifar\\data_batch_4')
    cifar_train5 = load_data('D:\\Box\\cifar\\data_batch_5')
    data = np.concatenate((cifar_train1[b'data'], cifar_train2[b'data'], cifar_train3[b'data'], cifar_train4[b'data'], cifar_train5[b'data']))
    labels = np.concatenate((cifar_train1[b'labels'], cifar_train2[b'labels'], cifar_train3[b'labels'], cifar_train4[b'labels'], cifar_train5[b'labels']))
    return {b'data': data, b'labels': labels}

cifar_train = load_train_data()
cifar_test = load_data('D:\\Box\\cifar\\test_batch')

def get_train_batch(n):
    indices = np.random.choice(len(cifar_train[b'data']), size=n, replace=False)
    return np.ndarray.astype(cifar_train[b'data'][indices], float) / 255, np.array(cifar_train[b'labels'])[indices]

def get_test_batch(n):
    return np.ndarray.astype(cifar_test[b'data'][:n], float) / 255, cifar_test[b'labels'][:n]

summary_writer = tf.summary.FileWriter('./logs')

test_batch = get_test_batch(3000)

NX = 32

X = tf.placeholder(tf.float32, [None, 3072])
Xr = tf.reshape(X, [-1, 3, 1024])
Xr = tf.transpose(Xr, [0, 2, 1])
Xr = tf.reshape(Xr, [-1, 32, 32, 3])
Y_i = tf.placeholder(tf.uint8, [None])

D1 = 4
N1 = 16
S1 = 1
W1 = tf.Variable(tf.truncated_normal([D1, D1, 3, N1], stddev=0.1))
b1 = tf.Variable(tf.ones([N1])/10)

layer1_weight_summary = tf.summary.image('layer1_weight', tf.transpose(W1, [3, 0, 1, 2]), max_outputs=24)

D2 = 5
N2 = 32
S2 = 2
W2 = tf.Variable(tf.truncated_normal([D2, D2, N1, N2], stddev=0.1))
b2 = tf.Variable(tf.ones([N2])/10)

D3 = 6
N3 = 64
S3 = 2
W3 = tf.Variable(tf.truncated_normal([D3, D3, N2, N3], stddev=0.1))
b3 = tf.Variable(tf.ones([N3])/10)

NS = int(NX/S1/S2/S3)
F3 = int(NS**2 * N3)

N4 = 200
W4 = tf.Variable(tf.truncated_normal([F3, N4], stddev=0.1))
b4 = tf.Variable(tf.ones([N4])/10)

N5 = 10
W5 = tf.Variable(tf.truncated_normal([N4, N5], stddev=0.1))
b5 = tf.Variable(tf.ones([N5])/10)

# for i in range(N4):
#     W4r = tf.reshape(W4, [NS, NS, N2, N4])
#     W4i = tf.slice(W4r, [0, 0, 0, i], [-1, -1, -1, 1])
#     W4t = tf.transpose(W4i, [2, 0, 1, 3])
#     tf.summary.image('layer2_weight_' + str(i), tf.reshape(W4i, [N2, NS, NS, 1]))

# N5 = 10
# W5 = tf.Variable(tf.truncated_normal([N4, 10], stddev=0.1))
# b5 = tf.Variable(tf.ones([10])/10)

global_step = tf.Variable(0, trainable=False)

L1 = tf.nn.relu(tf.nn.conv2d(Xr, W1, strides=[1, S1, S1, 1], padding='SAME') + b1)
L2 = tf.nn.relu(tf.nn.conv2d(L1, W2, strides=[1, S2, S2, 1], padding='SAME') + b2)
L2p = tf.nn.max_pool(L2, [1, 4, 4, 1], [1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(tf.nn.conv2d(L2p, W3, strides=[1, S3, S3, 1], padding='SAME') + b3)
L3r = tf.reshape(L3, [-1, F3])
L4 = tf.nn.relu(tf.matmul(L3r, W4) + b4)
L5 = tf.nn.softmax(tf.matmul(L4, W5) + b5)
Y = L5

S_BS = 3
batch_L0_summary = tf.summary.image('batch_L0', tf.slice(Xr, [0, 0, 0, 0], [S_BS, -1, -1, -1]))
batch_L1_1_summary = tf.summary.image('batch_L1_1', tf.slice(L1, [0, 0, 0, 0], [S_BS, -1, -1, 3]))
batch_L1_2_summary = tf.summary.image('batch_L1_2', tf.slice(L1, [0, 0, 0, 3], [S_BS, -1, -1, 3]))
batch_L2_1_summary = tf.summary.image('batch_L2_1', tf.slice(L2, [0, 0, 0, 0], [S_BS, -1, -1, 3]))
batch_L2_2_summary = tf.summary.image('batch_L2_2', tf.slice(L2, [0, 0, 0, 3], [S_BS, -1, -1, 3]))
batch_L3_1_summary = tf.summary.image('batch_L3_1', tf.slice(L3, [0, 0, 0, 0], [S_BS, -1, -1, 3]))
batch_L3_2_summary = tf.summary.image('batch_L3_2', tf.slice(L3, [0, 0, 0, 3], [S_BS, -1, -1, 3]))
batch_L4_summary = tf.summary.image('batch_L4', tf.reshape(tf.slice(L4, [0, 0], [S_BS, -1]), [S_BS, 1, N4, 1]))
batch_L5_summary = tf.summary.image('batch_L5', tf.reshape(tf.slice(L5, [0, 0], [S_BS, -1]), [S_BS, 1, N5, 1]))

Y_ = tf.one_hot(Y_i, 10)
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_sum(tf.cast(is_correct, tf.float32))

learning_rate_start = 0.0003
learning_rate = tf.train.exponential_decay(learning_rate_start, global_step, 5000, 0.97, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_step = optimizer.minimize(cross_entropy, global_step=global_step)

batch_size = tf.cast(tf.shape(Y_i)[0], tf.float32)
accuracy_norm = accuracy / batch_size
cross_entropy_norm = cross_entropy / batch_size

summary_merged_train = tf.summary.merge([
    layer1_weight_summary,
    batch_L0_summary,
    batch_L1_1_summary,
    batch_L1_2_summary,
    batch_L2_1_summary,
    batch_L2_2_summary,
    batch_L3_1_summary,
    batch_L3_2_summary,
    batch_L4_summary,
    batch_L5_summary,
    tf.summary.scalar('accuracy', accuracy_norm),
    tf.summary.scalar('cross_entropy', cross_entropy_norm)
    ])

summary_merged_test = tf.summary.merge([
    tf.summary.scalar('accuracy_test', accuracy_norm),
    tf.summary.scalar('cross_entropy_test', cross_entropy_norm)
    ])

def run_weights(sess):
    for i in range(10000):
        train_x, train_y = get_train_batch(100)
        train_data = {X: train_x, Y_i: train_y}

        sess.run(train_step, feed_dict=train_data)

        train_a, train_c, train_s = sess.run([accuracy_norm, cross_entropy_norm, summary_merged_train], feed_dict=train_data)

        summary_writer.add_summary(train_s, i)

        test_data = {X: test_batch[0], Y_i: test_batch[1]}
        test_a, test_c, test_s = sess.run([accuracy_norm, cross_entropy_norm, summary_merged_test], feed_dict=test_data)

        summary_writer.add_summary(test_s, i)

        print('{0}\tTest A: {1}.\tCE: {2}\tTrain A: {3}.\tCE: {4}'.format(i, test_a, test_c, train_a, train_c))

#
# image objective
#

def run_image(sess):
    ()

#
# Main
#

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

summary_writer.add_graph(sess.graph)

objectivename = sys.argv[1] if len(sys.argv) > 1 else 'weights'
saverestoreaction = sys.argv[2] if len(sys.argv) > 2 else ''
saverestorename = sys.argv[3] if len(sys.argv) > 3 else 'model.ckpt'

if saverestoreaction == 'r' or saverestoreaction == 'rs':
    saver.restore(sess, saverestorename)

if objectivename == 'weights':
    run_weights(sess)
elif objectivename == 'image':
    run_image(sess)

if saverestoreaction == 's' or saverestoreaction == 'rs':
    save_path = saver.save(sess, saverestorename)
    print('Model saved: %s' % save_path)
