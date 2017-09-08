import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# placeholder占位符

# x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。
# 我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。
# 我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。
# （这里的None表示此张量的第一个维度可以是任何长度的。）
x = tf.placeholder("float", [None, 784])
# W代表权重
# W的维度是[784，10]，因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量，每一位对应不同数字类。
W = tf.Variable(tf.zeros([784, 10]))
# b代表偏置量
# b的形状是[10]，所以我们可以直接把它加到输出上面。
b = tf.Variable(tf.zeros([10]))

# matmul表示x*W，再加上b，输入到softmax函数里
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 至此，我们先用了几行简短的代码来设置变量，然后只用了一行代码来定义我们的模型。

# 计算交叉熵
y_ = tf.placeholder("float", [None, 10])
# 首先用tf.log计算y的每个元素的对数。
# 接下来吧y_的每一个元素和tf.log(y)的对应元素相乘。
# 最后用tf.reduce_sum计算张量的所有元素之和。
# 注意，这里的交叉熵不仅仅用来衡量单一的一对预测和真实值，而是所有100幅图片的交叉熵的总和。对于100个数据点的预测表现比单一数据点的表现能更好地描述我们的模型的性能
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 训练步骤
# 使用剃度下降，用0.01学习速率来最小化交叉熵。
# 这里实际上所做的是，它会在后台给描述你的计算的那张图里面增加一系列新的计算操作单元用于实现反向传播算法和梯度下降算法。
# 然后，它返回给你的只是一个单一的操作，当运行这个操作时，它用梯度下降算法训练你的模型，微调你的变量，不断减少成本。
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 初始化我们创建的变量
init = tf.initialize_all_variables()
# 在一个 Session 里面启动我们的模型，并且初始化变量
sess = tf.Session()
sess.run(init)
# 随机剃度下降
for i in range(1000):
    # 循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点
    # 然后我们用这些数据点作为参数替换之前的占位符来运行 train_step
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# tf.argmax能给出某个tensor（张量）对象在某一维上的数据最大值所在的索引值
# tf.argmax(y, 1)返回的是模型对于任一输入x预测到的标签值
# tf.argmax(y_, 1)代表正确的标签
# 用tf.equal预测是否真是标签匹配（索引位置一样表示匹配）
# 结果返回的是一组布尔值

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 为了确定正确预测的比例，我们可以吧布尔值转换成浮点数，然后取平均。
# 例如 true，false，true，true，变成1, 0, 1, 1，平均后得到0.75
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# 准确率大概在91%
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

