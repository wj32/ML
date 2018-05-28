import tensorflow as tf
mnist = tf.contrib.learn.datasets.load_dataset("mnist")

X = tf.placeholder(tf.float32, [None, 784])
Y_i = tf.placeholder(tf.uint8, [None])

N1 = 200
W1 = tf.Variable(tf.truncated_normal([784, N1], stddev=0.1))
b1 = tf.Variable(tf.zeros([N1]))

N2 = 100
W2 = tf.Variable(tf.truncated_normal([N1, N2], stddev=0.1))
b2 = tf.Variable(tf.zeros([N2]))

N3 = 40
W3 = tf.Variable(tf.truncated_normal([N2, N3], stddev=0.1))
b3 = tf.Variable(tf.zeros([N3]))

N4 = 10
W4 = tf.Variable(tf.truncated_normal([N3, N4], stddev=0.1))
b4 = tf.Variable(tf.zeros([N4]))

global_step = tf.Variable(0, trainable=False)

init = tf.global_variables_initializer()

L1 = tf.sigmoid(tf.matmul(X, W1) + b1)
L2 = tf.sigmoid(tf.matmul(L1, W2) + b2)
L3 = tf.sigmoid(tf.matmul(L2, W3) + b3)
L4 = tf.nn.softmax(tf.matmul(L3, W4) + b4)
Y = L4

Y_ = tf.one_hot(Y_i, 10)
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_sum(tf.cast(is_correct, tf.float32))

learning_rate_start = 0.005
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
