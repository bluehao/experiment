import tensorflow as tf
import mnist_forward as mf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import os


#定义神经网络需要的参数
REGULARIZTION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
BATCH_SIZE = 100
TRAINING_STEP = 30000

#模型保存路径
MOVING_SAVE_PATH = "save_model/"
MOVING_SAVE_NAME = "model.ckpt"


#定义训练过程的函数
def train(mnist):
    #定义输入输出
    x = tf.placeholder(tf.float32, [None, mf.INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, mf.OUT_NODE], name="y-output")
    #定义l2正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZTION_RATE)
    #利用mnist_forward中的forward_prop函数计算前向传播的结果
    y = mf.forward_prop(x, regularizer)
    #初始化训练轮数
    global_step = tf.Variable(0, trainable=False)
    #定义变量的滑动平均操作
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())
    #定义交叉熵并计算其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #定义损失函数
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    #定义带有衰减率的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    #优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name="train")

    #初始化Tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #初始化所有变量
        tf.global_variables_initializer().run()
        #训练模型
        for i in range(TRAINING_STEP):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _op, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            #每1000轮保存一次模型
            if i % 1000 == 0:
                #输出当前训练batch上的损失函数大小，来了解模型的训练情况
                print("After %d traning steps, loss is %g." % (step, loss_value))
                # #保存当前模型，加入global变量，会在保存的文件名后面加上训练的轮数
                if not os.path.exists(MOVING_SAVE_PATH):
                    os.makedirs(MOVING_SAVE_PATH)
                saver.save(sess, os.path.join(MOVING_SAVE_PATH, MOVING_SAVE_NAME), global_step)


def main(argv=None):
    mnist = read_data_sets("data/", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()


