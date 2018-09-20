import numpy as np
import tensorflow as tf
from tensorflow.models.rnn.ptb import reader


DATA_PATH = 'data/'          #PTB数据存放的路径
HIDDEN_SIZE = 200            #隐藏层的大小
NUM_LAYERS = 2               #深层神经网络中LSTM结构的层数
VOCABULARY_SIZE = 10000      #词汇表的大小
LEARNING_RATE = 1.0          #学习速率
TRAINING_BATCH_SIZE = 20     #训练数据batch的大小
TRAINING_NUM_STEP = 35       #训练数据截断的长度

#在测试的时候不需要截断数据，所以可将数据看作是一个超长的序列
EVAL_BATCH_SIZE = 1          #测试数据batch的大小
EVAL_NUM_STEP = 1            #测试数据的截断长度
NUM_EPOCH = 2                #使用训练数据的轮数
KEEP_PROP = 0.5              #节点不被dropout的概率
MAX_GRAD_NORM = 5            #用于控制梯度膨胀的参数


#通过一个PTBModel类来描述模型，这样方便管理和维护循环神经网络
class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        #记录使用batch的大小和截断长度
        self.batch_size = batch_size
        self.num_steps = num_steps

        #定义输入层，输入层的维度为batch_size * num_steps
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])

        #定义预期输出
        self.target = tf.placeholder(tf.int32, [batch_size, num_steps])

        #定义使用LSTM结构为循环体结构且使用dropout的深层循环神经网络
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
        if is_training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROP)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS, state_is_tuple=True)

        #初始化最初的状态，为全零向量
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        #将单词ID转化为单词向量，词汇表大小为VOCABULARY_SZIE, 每个单词的向量维度为HIDDEN_SIZE（个人觉得单词向量维度可以是任意的）,
        #所以embedding大小为VOCABULARY_SIZE * HIDDEN_SIZE
        embedding = tf.get_variable("embedding", [VOCABULARY_SIZE, HIDDEN_SIZE])

        #将原本batch_sizee * num_steps个单词ID转化为单词向量，转化后的输入层维度为batch_sizee * num_steps * HIDDEN_SIZE
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        #只在训练时使用dropout
        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROP)

        #定义输出列表，将不同时刻的输出收集起来，再通过一个全连接层得到最终输出
        outputs = []
        #state存储不同batch中的状态
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                #从输入数据中获取当前时刻的输入并传入到lstm结构中
                cell_output, state = cell(inputs[:, time_step, :], state)
                #将当前输出加入outputs队列
                outputs.append(cell_output)
        #把输出队列展开成[batch, hidden_size * num_steps]的形状，然后再reshape成[batch * hidden_size, num_steps]
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])

        #将得到的输出经过一个全连接层，得到最终的预测结果，最终的预测结果在长度上都是为VOCABULARY_SIZE的数组，经过softmax层后
        #表示下一个位置是不同单词的概率
        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCABULARY_SIZE])
        biases = tf.get_variable("biases", [VOCABULARY_SIZE])
        logits = tf.matmul(output, weight) + biases

        #定义交叉熵损失函数,tf.contrib.legacy_seq2seq.sequence_loss_by_example计算一个序列交叉熵和
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],
                                                                  [tf.reshape(self.target, [-1])],
                                                                  [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        #计算得到每个batch的损失和，求平均值
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        #只在训练模型时进行反向传播
        if not is_training:
            return
        trainable_variables = tf.trainable_variables()
        #tf.clip_by_global_norm控制梯度的大小，避免梯度爆炸
        grades, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)

        #定义优化方法
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        #定义训练步骤
        self.train_op = optimizer.apply_gradients(zip(grades, trainable_variables))

#使用给定的模型model在数据data上运行train_op，并返回在全部数据上的perplexity值
def run_epoch(session, model, data, train_op, epoch_size, output_log):
    #计算perplexity的辅助值
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    for step in range(epoch_size):
        x, y = session.run(data)

        #在当前batch上运行train_op，并计算损失值，
        cost, state, _ = session.run([model.cost, model.final_state, train_op],
                                     {model.input_data: x, model.target: y, model.initial_state: state})
        #将不同时刻不同batch的概率加起来，将这个和作指数运算就可以得到perplexity值
        total_costs += cost
        iters += model.num_steps
        #只有在训练时输出日志
        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %.3f" % (step, np.exp(total_costs / iters)))

    #返回给定模型在给定数据上的perplexity
    return np.exp(total_costs / iters)


def main(argv=None):
    # 获取数据源
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
    #定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # 计算一个epoch需要训练的次数
    train_data_len = len(train_data)
    train_batch_len = train_data_len // TRAINING_BATCH_SIZE
    train_epoch_size = (train_batch_len - 1) // TRAINING_NUM_STEP

    valid_data_len = len(valid_data)
    valid_batch_len = valid_data_len // EVAL_BATCH_SIZE
    valid_epoch_size = (valid_batch_len - 1) // EVAL_NUM_STEP

    test_data_len = len(test_data)
    test_batch_len = test_data_len // EVAL_BATCH_SIZE
    test_epoch_size = (test_batch_len - 1) // EVAL_NUM_STEP

    #定义训练使用的循环神经网络
    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAINING_BATCH_SIZE, TRAINING_NUM_STEP)

    #定义评测用的模型
    with tf.variable_scope("language_model", reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        train_d = reader.ptb_producer(train_data, TRAINING_BATCH_SIZE, TRAINING_NUM_STEP)
        valid_d = reader.ptb_producer(valid_data, EVAL_BATCH_SIZE, EVAL_NUM_STEP)
        test_d = reader.ptb_producer(test_data, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

        # 开启多线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        #使用训练数据训练模型
        for i in range(NUM_EPOCH):
            print("in iteration: %d" % (i + 1))
            #在所有训练数据上训练循环神经网络
            run_epoch(session, train_model, train_d, train_model.train_op, train_epoch_size, True)

            #使用验证数据评测模型效果
            valid_perplexity = run_epoch(session, eval_model, valid_d, tf.no_op(), valid_epoch_size, False)
            print("Epoch: %d Validation Perplexity: %.3f" % (i + 1, valid_perplexity))

        #使用测试数据测试模型
        test_perplexity = run_epoch(session, eval_model, test_d, tf.no_op(), test_epoch_size, False)
        print("Test Perplexity: %.3f" % test_perplexity)

        # 关闭多线程
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
