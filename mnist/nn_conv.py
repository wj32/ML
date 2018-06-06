import tensorflow as tf
mnist = tf.contrib.learn.datasets.load_dataset("mnist")

NX = 28

X = tf.placeholder(tf.float32, [None, NX**2])
Y_i = tf.placeholder(tf.uint8, [None])

D1 = 5
N1 = 1
S1 = 1
W1 = tf.Variable(tf.truncated_normal([D1, D1, 1, N1], stddev=0.1))
b1 = tf.Variable(tf.ones([N1])/10)

tf.summary.image("layer1_weight", tf.transpose(tf.reshape(W1, [1, D1, D1, N1]), [3, 1, 2, 0]))

F1 = int((NX/S1)**2 * N1)

N4 = 10
W4 = tf.Variable(tf.truncated_normal([F1, N4], stddev=0.1))
b4 = tf.Variable(tf.ones([N4])/10)

for i in range(N4):
    W4r = tf.reshape(W4, [NX, NX, N1, N4])
    W4i = tf.slice(W4r, [0, 0, 0, i], [-1, -1, -1, 1])
    W4t = tf.transpose(W4i, [2, 0, 1, 3])
    tf.summary.image("layer2_weight_" + str(i), tf.reshape(W4i, [N1, NX, NX, 1]))

# N5 = 10
# W5 = tf.Variable(tf.truncated_normal([N4, 10], stddev=0.1))
# b5 = tf.Variable(tf.ones([10])/10)

global_step = tf.Variable(0, trainable=False)

init = tf.global_variables_initializer()

Xr = tf.reshape(X, [-1, NX, NX, 1])
L1 = tf.nn.relu(tf.nn.conv2d(Xr, W1, strides=[1, S1, S1, 1], padding='SAME') + b1)
L1r = tf.reshape(L1, [-1, F1])
L4 = tf.nn.softmax(tf.matmul(L1r, W4) + b4)
Y = L4

S_BS = 3
tf.summary.image("batch_L0", tf.reshape(tf.slice(X, [0, 0], [S_BS, -1]), [S_BS, NX, NX, 1]))
tf.summary.image("batch_L1", tf.slice(L1, [0, 0, 0, 0], [S_BS, -1, -1, 1]))
tf.summary.image("batch_L4", tf.reshape(tf.slice(L4, [0, 0], [S_BS, -1]), [S_BS, 1, N4, 1]))

# L4 = tf.nn.relu(tf.matmul(L1r, W4) + b4)
# L5 = tf.nn.softmax(tf.matmul(L4, W5) + b5)
# Y = L5

Y_ = tf.one_hot(Y_i, 10)
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_sum(tf.cast(is_correct, tf.float32))

learning_rate_start = 0.0003
learning_rate = tf.train.exponential_decay(learning_rate_start, global_step, 10000, 0.97, staircase=True)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy, global_step=global_step)

batch_size = tf.cast(tf.shape(Y_i)[0], tf.float32)
accuracy_norm = accuracy / batch_size
cross_entropy_norm = cross_entropy / batch_size

tf.summary.scalar("accuracy", accuracy_norm)
tf.summary.scalar("cross_entropy", cross_entropy_norm)
summary_merged = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter("./logs")

sess = tf.Session()
sess.run(init)

summary_writer.add_graph(sess.graph)

for i in range(5000):
    batch_size = 100
    train_x, train_y = mnist.train.next_batch(batch_size)
    train_data = {X: train_x, Y_i: train_y}

    sess.run(train_step, feed_dict=train_data)

    train_a, train_c, s = sess.run([accuracy_norm, cross_entropy_norm, summary_merged], feed_dict=train_data)

    summary_writer.add_summary(s, i)

    test_data = {X: mnist.test.images, Y_i: mnist.test.labels}
    test_a, test_c = sess.run([accuracy_norm, cross_entropy_norm], feed_dict=test_data)

    print('{0}\tTest A: {1}.\tCE: {2}\tTrain A: {3}.\tCE: {4}'.format(i, test_a, test_c, train_a, train_c))
