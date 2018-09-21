import numpy as np
import tensorflow as tf
import tensorflow.contrib.learn as TFLearn
import tensorflow.contrib.layers as layers
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
from tensorflow.python.ops.rnn import dynamic_rnn

#加载matplotlib工具包，使用该工具包可以对预测的sin函数曲线进行绘图
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

#通过TFLearn来训练模型
learn = TFLearn

HIDDEN_SIZE = 30                #LSTM中隐藏节点的个数
NUM_LAYERS = 2                  #LSTM的层数
TIMESTEPS = 10                  #循环神经网络的截断长度
TRAINING_STEPS = 10000          #训练轮数
BATCH_SIZE = 32                 #batch的大小
TRAINING_EXAMPLES = 10000       #训练数据个数
TEST_EXAMPLES = 1000            #测试数据的个数
SAMPLE_GAP = 0.01               #采样间隔


#生成训练和测试数据
def generate_data(seq):
    x = []
    y = []
    #序列的第i项和TIMESTEPS-1项合在一起作为输入，第i+TIMESTEPS项作为输出
    #即用sin函数前面的TIMESTEPS个点的信息，预测第i+TIMESTEPS个点的函数值
    for i in range(len(seq) - TIMESTEPS - 1):
        x.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    x=np.array(x, dtype=np.float32)
    y=np.array(y, dtype=np.float32)
    return x, y


#LSTM结构单元
def LstmCell():
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
    return lstm_cell


#LSTM模型
def lstm_model(x, y):
    #定义lstm
    cell = tf.nn.rnn_cell.MultiRNNCell([LstmCell() for _ in range(NUM_LAYERS)])

    #使用tensorflow接口将多层lstm连接成RNN并计算前向传播结果
    outputs, _ = dynamic_rnn(cell, x, dtype=tf.float32)
    #输出最后一时刻的结果，即预测结果
    output = tf.reshape(outputs, [-1, HIDDEN_SIZE])
    #对lstm网络加一层全连接网络,并将数据压缩成一维数组
    predictions = tf.contrib.layers.fully_connected(output, 1, None)
    # 将predictions和labels调整为统一的shape
    y = tf.reshape(y, [-1])
    predictions = tf.reshape(predictions, [-1])
    # 计算损失值
    loss = tf.losses.mean_squared_error(predictions, y)

    #创建模型优化器，并得到优化步骤
    train_op = layers.optimize_loss(loss,
                                    tf.train.get_global_step(),
                                    optimizer="Adagrad",
                                    learning_rate=0.1)
    return predictions, loss, train_op

#建立深层循环网络模型
regressor = SKCompat(learn.Estimator(model_fn=lstm_model))

#用正弦函数生成训练数据和测试数据
test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TEST_EXAMPLES) * SAMPLE_GAP
train_x, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES, dtype=np.float32)))
test_x, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TEST_EXAMPLES, dtype=np.float32)))

#调用fit函数训练模型
regressor.fit(train_x, train_y, batch_size=BATCH_SIZE, steps= TRAINING_STEPS)

#使用训练好的模型对测试数据进行测试
predicted = [[pred] for pred in regressor.predict(test_x)]

#计算rmse作为计算指标
rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
print("Mean Error is : %f" % rmse[0])

#保存输出结果图
fig = plt.figure()
plot_predicted = plt.plot(predicted, label='predicted')
plot_test = plt.plot(test_y, label='real_sin')
plt.legend([plot_predicted, plot_test], ['predicted', 'real_sin'])
fig.savefig('sin.png')
