import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# #载入mnist数据集
# mnist = input_data.read_data_sets("data", one_hot=True)

# #打印Training data size: 55000
# print("Training data size: ", mnist.train.num_examples)
#
# #打印Validating data size；5000
# print("Validating data size: ", mnist.validation.num_examples)
#
# #打印Testing data size: 10000
# print("Testing data size: ", mnist.test.num_examples)
#
# #打印Example training data
# print("Example training data: ", mnist.train.images[0])
#
# #打印Example training data label
# print("Example training data lable: ", mnist.train.labels[0])

# mnist数据相关的常数
INPUT_NODE = 784      # 输入层的节点数
OUTPUT_NODE = 10      # 输出层的节点数

# 配置神经网络的参数
LAYER1_NODE = 500     # 隐藏层节点数
BATCH_SIZE = 100      # 一个训练batch的训练数据个数。数字越小时，训练过程越接近随机梯度下降；数字越大时，训练越接近梯度下降
LEARNING_RATE_BASE = 0.8        # 基础的学习率
LEARNING_RATE_DECAY = 0.99      # 学习率的衰减率
REGULARIZATION_RATE = 0.0001    # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000          # 训练轮数
MOVING_AVERAGE_DECAY = 0.99     # 滑动平均衰减率

# 定义一个函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果。
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 当没有提供滑动平均类时，直接使用参数当前的取值。
    if avg_class == None:
        # 计算隐藏层的前向传播结果，这里使用了Relu激活函数。
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        # 计算输出层的前向传播结果，因为在计算损失函数时会计算softmax函数，所以这里不需要加入激活函数。
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 首先使用avg_class.average函数来计算变量的滑动平均值，再计算相应的神经网络前向传播结果。
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

# 训练模型的过程。
def train(mnist):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=[None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODE], name='y-input')

    # 生成隐藏层的参数。
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数。
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算在当前参数下神经网络前向传播的结果，这里滑动平均类为None
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义存储训练轮数的变量。
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    # 给定训练轮数的变量可以加快训练早起变量的更新速度
    # variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数。
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失
    regularization = regularizer(weights1) + regularizer(weights2)
    #总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY )

    # 使用tf.train.gradientDescentOptimizer优化算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 在训练神经网络模型时，每过一边数据既需要通过反向传播来更新神经网络的参数
    train_op = tf.group(train_step)

    # 检验神经网络前向传播结果是否正确
    # 这个运算将布尔型的数值转化为实数型，然后计算平均值，这个平均值就是模型在这一组数据上的正确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        # 准备验证数据
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # 准备测试数据，这部分数据训练时是不可见的，只作为模型优劣的最后评价标准
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 迭代的训练神经网络
        for i in range(TRAINING_STEPS):
            # 每1000轮输出一次在验证数据集上的测试结果
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy is %g" % (i, validate_acc))

            #产生这一轮使用的一个batch的训练数据，并运行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        #在训练结束后，在测试数据上检测神经网络模型的最终正确率。
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy is %g" % (TRAINING_STEPS, test_acc))

# 主程序入口
def main(argv = None):
    mnist = input_data.read_data_sets("data/", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()

