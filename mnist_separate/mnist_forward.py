import tensorflow as tf

# 定义神经网络各层的节点数
INPUT_NODE = 784
HIDEN_NODE = 500
OUT_NODE = 10

#初始化权重矩阵的值，并将权重值的正则化项加入到自定义的集合中
def get_weigths(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    #将weights正则化后的值加入到自定的losses集合中
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

#定义前向传播的函数
def forward_prop(input_tensor,regularizer):
    #声明神经网络第一层前向传播的过程
    with tf.variable_scope('layer1'):
        weights = get_weigths([INPUT_NODE, HIDEN_NODE], regularizer)
        bias = tf.get_variable("bias", [HIDEN_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + bias)

    #声明第二层神经网络第二层前向传播的过程
    with tf.variable_scope('layer2'):
        weights = get_weigths([HIDEN_NODE, OUT_NODE], regularizer)
        bias = tf.get_variable("bias", [OUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + bias

    return layer2


