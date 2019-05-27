import tensorflow as tf
import matplotlib.pyplot as plt

# y = W * x + b
x_train = [1, 2, 3]
y_train = [1, 2, 3]
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# H:Hypothesis
H = W * x_train + b

# cost(loss) function
# (H-y)*(H-y)->minimumize
cost =  tf.reduce_mean(tf.square(H - y_train))

#Optimizer
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#train_op = train_op.minimize(cost)

# Lainch the graph on a session.
W_val_arr = []
b_val_arr = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #fit
    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train_op, cost, W, b])
        W_val_arr.append(W_val)
        b_val_arr.append(b_val)

        if step % 100 == 0 :
            print(step, cost_val, W_val, b_val)
# Learns best fit W:[ 1.],  b:[ 0.]

plt.plot(W_val_arr, "r+", linewidth=0.01, label="W_val")
plt.plot(b_val_arr, "b+", linewidth=0.05, label="b_val")
plt.ylim([-0.5, 1.5])
plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('val')
plt.show()
