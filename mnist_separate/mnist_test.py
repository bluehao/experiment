import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import  mnist_train


#时间间隔
TIME_SECS = 10

def mnsit_test(mnist):
    #定义输入输出
    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUT_NODE], name='y-output')
    validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

    #调用前向传播的函数求结果，将正则化置为空，因为测试的时候并不关注正则化损失的值
    y = mnist_forward.forward_prop(x, None)
    #使用前向传播的结果计算正确率
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    #通过变量重命名的方式来加载模型
    variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
    variable_averages_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variable_averages_restore)

    while True:
        with tf.Session() as sess:
            # tf.global_variables_initializer().run()
            ckpt = tf.train.get_checkpoint_state(mnist_train.MOVING_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                #加载模型
                saver.restore(sess, ckpt.model_checkpoint_path)
                #通过文件名得到模型保存时的迭代轮数
                global_step = ckpt.model_checkpoint_path.split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print("After %s traning step(s), validation accuracy=%g" % (global_step, accuracy_score))
            else:
                print("No checkpoint file found!")
        #每10秒加载一次最新的模型，并计算其正确率
        time.sleep(TIME_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("data/", one_hot=True)
    mnsit_test(mnist)

if __name__ == '__main__':
    tf.app.run()
