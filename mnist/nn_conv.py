import tensorflow as tf
mnist = tf.contrib.learn.datasets.load_dataset("mnist")

NX = 28

X = tf.placeholder(tf.float32, [None, NX**2])
Y_i = tf.placeholder(tf.uint8, [None])

N1 = 4
S1 = 1
W1 = tf.Variable(tf.truncated_normal([5, 5, 1, N1], stddev=0.1))
b1 = tf.Variable(tf.ones([N1])/10)

N2 = 8
S2 = 2
W2 = tf.Variable(tf.truncated_normal([5, 5, N1, N2], stddev=0.1))
b2 = tf.Variable(tf.ones([N2])/10)

N3 = 12
S3 = 2
W3 = tf.Variable(tf.truncated_normal([4, 4, N2, N3], stddev=0.1))
b3 = tf.Variable(tf.ones([N3])/10)

F3 = int((NX/S1/S2/S3)**2 * N3)

N4 = 200
W4 = tf.Variable(tf.truncated_normal([F3, N4], stddev=0.1))
b4 = tf.Variable(tf.ones([N4])/10)

N5 = 10
W5 = tf.Variable(tf.truncated_normal([N4, 10], stddev=0.1))
b5 = tf.Variable(tf.zeros([10]))

global_step = tf.Variable(0, trainable=False)

init = tf.global_variables_initializer()

Xr = tf.reshape(X, [-1, NX, NX, 1])
L1 = tf.nn.relu(tf.nn.conv2d(Xr, W1, strides=[1, S1, S1, 1], padding='SAME') + b1)
L2 = tf.nn.relu(tf.nn.conv2d(L1, W2, strides=[1, S2, S2, 1], padding='SAME') + b2)
L3 = tf.nn.relu(tf.nn.conv2d(L2, W3, strides=[1, S3, S3, 1], padding='SAME') + b3)
L3r = tf.reshape(L3, [-1, F3])
L4 = tf.nn.relu(tf.matmul(L3r, W4) + b4)
L5 = tf.nn.softmax(tf.matmul(L4, W5) + b5)
Y = L5

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

sess = tf.Session()
sess.run(init)

for i in range(5000):
    batch_size = 100
    train_x, train_y = mnist.train.next_batch(batch_size)
    train_data = {X: train_x, Y_i: train_y}

    sess.run(train_step, feed_dict=train_data)

    train_a, train_c = sess.run([accuracy_norm, cross_entropy_norm], feed_dict=train_data)

    test_data = {X: mnist.test.images, Y_i: mnist.test.labels}
    test_a, test_c = sess.run([accuracy_norm, cross_entropy_norm], feed_dict=test_data)

    print('{0}\tTest A: {1}.\tCE: {2}\tTrain A: {3}.\tCE: {4}'.format(i, test_a, test_c, train_a, train_c))
