# pointerNetwork
a custom layer in keras to implement a pointer network decoder

## 递归神经网络求解优化问题——日程安排和资源分配问题

神经网络由多层网络组成，每一层又由神经元组成：

![Alt](doc/NN.JPG)

每个神经元接受输入，通过激活函数做非线性变换，并输出到下一层神经网络。以常见神经网络为例：

输入$x_1, x_2, \dots x_j $到神经元k，神经元输出为 $y_k = \sigma \big( \Sigma_{j=0}^m w_{kj} x_j + b_{kj} \big) $，其中w为权重，b为bias，x为来自上一层的输入值，y为输出到下一层的输出值，$\sigma$为激活函数，控制当输入值大于一定阈值时，神经元处于打开状态并输出；否则为关闭。一般采用的激活函数有双曲正切[hyperbolic tangent](https://en.wikipedia.org/wiki/Hyperbolic_function)，[sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) 函数等等。

关于反向传播[Backpropagation](https://en.wikipedia.org/wiki/Backpropagation):

神经网络的学习机制是：依靠每次输出结果和目标值比对生成损失函数loss function，利用损失函数的偏导数partial derivative求出下一次训练应该使用的参数的变化方向，最后带入梯度下降算法[gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)得到下一次训练应该使用的参数向量（对于不同的求解方法比如牛顿法和各种梯度下降方法，这里就不一一赘述了）。

在一个神经网络中，需要经过训练学习得到的参数集为所有神经元的weight 和bias值，每一层的损失函数都与下一层的实际预测结果和目标结果相关。从最后一层开始，逐层回溯并为每一层的梯度下降提供偏导数的过程叫做back propagation。

**反向传播**：

为方便起见，假设最终的损失函数为目标和预测结果之间的欧几里得距离平方：

$E = \frac{1}{2n} \sum \parallel \text{ } y - \hat{y} \text{ } \parallel ^2$

其中$\hat{y}$为实际预测值，y为目标值，$\parallel \text{ } \bullet \text{ } \parallel$ 代表两个向量间的距离。

所以有 $\cfrac{\partial E}{\partial \hat{y}} = y - \hat{y}$

仍然为方便起见，假设每个神经元的激活函数为逻辑函数logistic function：

$o_j = \sigma(z_j) = \cfrac{1}{1 + e^{-z_j}}$

其中$o_j$为第 j 层的输出，$z_j$为第 j 层被激活以前，对上一层 i 层输出 $o_i$ 的加权之和，$z_j = \sum w_{ij} o_i + b_{ij}$，$w_{ij}$为从 i 层到 j 层连接的weight向量，$b_{ij}$为从 i 层到 j 层连接的bias向量。

所以有 $\cfrac{\partial z_{ij}}{\partial w_{ij}} =  \cfrac{\partial \sum (w_{ij} o_i + b_{ij})}{\partial w_{ij}} = o_i$，

$\cfrac{\partial o_j}{\partial z_j} = \cfrac{\partial \sigma(z_j}{\partial z_j} = \cfrac{e^z}{(1+e^z)^2}$

根据偏导数的链式法则，把所有偏导数连接起来可以得到：$\cfrac{\partial E}{\partial w_{ij}} = \cfrac{\partial E}{\partial o_j} \cfrac{\partial o_j}{\partial z_j} \cfrac{\partial z_j}{\partial w_{ij}}$ 

其中，当 j 为最后一层时，$o_j = \hat{y}$ 并且 $\cfrac{\partial E}{\partial o_j} = \cfrac{\partial E}{\partial \hat{y}} $

这个偏导数链的每个部分都可以根据上面的公式得到。

### 递归神经网络：

神经网络可以分为[前馈神经网络](https://en.wikipedia.org/wiki/Feedforward_neural_network)和[递归神经网络](https://en.wikipedia.org/wiki/Recurrent_neural_network)。

前馈神经网络和递归神经网络的最大区别是，神经元的连接是否形成了闭环。前馈神经网络的信息流向只有从输入层到输出层一个方向。相反，递归神经网络依靠神经元之间的有向环实现记忆功能。

优化问题和时间序列问题一般来说采用递归神经网络解决。

最常用的递归神经网络结构是1997年由德国人Hochreiter和Schmidhuber提出的Long-Short-Term-Memory ([LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory))。这种结构在每次递归中直接应用记忆数据，使得记忆数据在多重神经元层的反向传播中不至于失真（参考[多伦多大学讲义](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L15%20Exploding%20and%20Vanishing%20Gradients.pdf)）。在LSTM的内部结构中通过input gate/output gate/forget gate, etc. 实现输入数据和记忆数据的相加而非相乘。

LSTM神经元的拓扑结构如下图：

![alt](doc/greff_lstm_diagram.png)
[图片来源](https://arxiv.org/pdf/1503.04069.pdf)

抽象为数学建模：

在 t 时间的输入：$x^t$

N: LSTM神经元个数

M: 输入维度

脚注i, o, f, z：input gate, output gate, forget gate, block

需要训练的参数为：

输入weight: $W_z, W_i, W_f, W_o \in \mathcal{R}^{N \times M}$

递归weight: $R_z, R_i, R_f, R_o \in \mathcal{R}^{N \times N}$

bias: $b_z, b_i, b_f, b_o \in \mathcal{R}^N$

公式为：

输入 $z^t = \sigma \big( W_z x^t + R_z y^{t-1} + b_z \big)$

input gate $i^t = \sigma \big( W_i x^t + R_i y^{t-1} + b_i \big)$

forget gate $f^t = \sigma \big( W_f x^t + R_f y^{t-1} + b_f \big)$

cell $c^t = z^t \odot i^t + c^{t-1} \odot f^t$

output gate $o^t = \sigma \big( W_o x^t + R_o y^{t-1} + b_o \big)$

输出 $y^t = h \big( c^t \big) \odot o^t$

### Attention network：

由LSTM神经元组合起来，或者由LSTM神经元变化得来的神经网络层在最近几年得到了快速发展。一种起初用于对齐图像和描述文字的网络[attention network, Bahdanau et. al.](https://arxiv.org/pdf/1409.0473.pdf)因为它的对齐原理，也可以被用于解决优化问题。

Bahdanau et.al. 的论文中提出的attention network使用了普通的LSTM神经元作为编码器，以编码器的输出作为attention model解码器的输入，并调节解码器输出的权重，达到与编码器对齐的目的。

论文中给出的解码器也是一个递归神经网络，拓扑结构比较复杂：

![alt](doc/attentionNetTopology.png)

抽象为数学建模：

编码器输入$X = (x_1, \dots, x_{T_x}), x_i \in \mathcal{R}^{K_x}$

解码器输出$Y = (y_1, \dots, y_{T_y}), y_i \in \mathcal{R}^{K_y}$

$K_x, K_y$ 是所有输入和输出元素的集合， $T_x, T_y$是输入和输出向量的长度。在attention network中，输入和输出的元素集，以及输入和输出的向量长度都可以不相等。而在排序问题中，一定有$K_x = K_y, T_x = T_y$，并且输入长度可变。

解码器中，

hidden state $s_i = (1 - z_i) \circ s_{i-1} + z_i \circ \tilde{s}_i$，其中

proposal state $\tilde{s}_i = tanh \big( W_p y_{i-1} + U_p [r_i \circ s_{i-1}] + C_p c_i \big)$

update state $z_i = \sigma \big( W_z y_{i-1} + U_z s_{i-1} + C_z c_i \big)$

reset state $z_i = \sigma \big( W_r y_{i-1} + U_r s_{i-1} + C_r c_i \big)$

context vector $c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j$

probability vector $\alpha_{ij} = \cfrac{exp(e_{ij})}{\sum_{k=1}^{T_x} exp(e_{ik})}$

alignment vector $e_{ij} = v_a^T tanh \big( W_a s_{i-1} + U_a h_j \big)$

其中 $h_j$ 是编码器的 hidden state 输出。

在这个结构中，可训练的模型参数为$W, U, C$ 和 $v_a^T$。

### Pointer network：

类似于attention network或者encoder-decoder结构的RNN广泛应用于机器翻译和生成图片、视频的文字描述，所以输出序列的元素集合通常是一个不同于输入序列集合，并且需要提前定义的字典。这种定义方式并不能满足在优化问题（比如排序、装箱、路径问题）的应用——优化问题的输出元素集合是输入集合的排列组合，并且根据输入序列长度变化。

[Pointer network](https://arxiv.org/pdf/1506.03134.pdf)调整了attention network的连接结构，使得输出序列是指向输入序列的一组指针：

![alt](doc/pointerNetCapture.JPG)

[图片来源](https://arxiv.org/pdf/1506.03134.pdf)

相对于attention network, pointer network取消了context vetor的连接以及 $y_t$，直接采用probability vector $\alpha_{t}$ 作为编码器的输入序列以及整个网络的输出序列：

![alt](doc/pointerNetTopology.png)

### Pointer Network的代码实现：

编码器采用了keras库提供的LSTM神经元。作为解码器，使用keras+Theano框架自行编写了订制化的神经网络层。

关于[keras](https://keras.io/#keras-the-python-deep-learning-library)：近年来跟神经网络和深度学习一起兴起的一个开源高层神经网络功能API，在后端可以使用tensorflow, theano, CNTK三种框架，项目发起人[François Chollet](https://github.com/fchollet)就职于google deepmind并在几年前在业余时间创建了开源项目keras。

相比直接使用tensorflow或其他框架搭建神经网络，keras更注重功能模块化，封装了包括常用神经网络层，优化器，目标函数，激活函数，初始分布等等功能，使使用者能够更方便地搭建各种神经网络结构而不需要担心后端的实现细节。另外keras还支持订制化的神经网络层定义，可以方便地与其他模块集成，并不影响前向和反向传播通路。

pointer network的订制化神经网络层按照keras网络层定义的惯例包含如下部分：

Build 部分在模型compile以前被执行，用来搭建网络层结构，包括维度，记忆状态，可训练参数集的定义和初始化，etc.

Call 部分在模型compile时被执行，一般负责主要的计算任务。在RNN中，另外有step部分执行递归的计算；所以在pointer network中使用call 一次性处理编码器hidden state到解码器的通路（上图中的time-distributed dense layer），减少递归的计算量。

get_initial_state部分初始化需要递归的函数值。

compute_output_shape是一个utility function，在运行模型summary时被调用，打印输出维度。

get_config也是一个utility function，在运行模型 save 和 load 时被调用。

最重要的递归计算部分在step，输入采用标准RNN格式 （output, [hidden state, cell state]），第二项的数组必须提供可以递归的symbolic variable，以实现反向传播。所以在pointer network中，根据拓扑结构，按照LSTM的输出参数标准格式（output, [hidden state, cell state]），在第一部分返回probability vector $\alpha$，在第二部分返回数组[update state z, reset state r]。

**Build** 部分代码实现如下：

```python
def build(self, input_shape):
    """ input_shape: shape of the encoder output. 
        Assuming the encoder is an LSTM, 
        input_shape = (batchsize, timestep, encoder hiddensize)          
    """

    self.batch_size, self.timesteps, self.input_dim = input_shape
    self.output_dim = self.timesteps

    if self.stateful:
        super(AttentionPointer, self).reset_states()

    self.states = [None, None]  # z, s_p

    # Matrices for creating the probability vector alpha
    self.V_a = self.add_weight(shape=(self.output_dim,),
                               name='V_a',
                               initializer=self.kernel_initializer,
                               regularizer=self.kernel_regularizer,
                               constraint=self.kernel_constraint)
    self.W_a = self.add_weight(shape=(self.units, self.output_dim),
                               name='W_a',
                               initializer=self.kernel_initializer,
                               regularizer=self.kernel_regularizer,
                               constraint=self.kernel_constraint)
    self.U_a = self.add_weight(shape=(self.input_dim, self.output_dim),
                               name='U_a',
                               initializer=self.kernel_initializer,
                               regularizer=self.kernel_regularizer,
                               constraint=self.kernel_constraint)
    self.b_a = self.add_weight(shape=(self.output_dim,),
                               name='b_a',
                               initializer=self.bias_initializer,
                               regularizer=self.bias_regularizer,
                               constraint=self.bias_constraint)

    # Matrices for the r (reset) gate
    self.U_r = self.add_weight(shape=(self.units, self.units),
                               name='U_r',
                               initializer=self.recurrent_initializer,
                               regularizer=self.recurrent_regularizer,
                               constraint=self.recurrent_constraint)
    self.W_r = self.add_weight(shape=(self.output_dim, self.units),
                               name='W_r',
                               initializer=self.recurrent_initializer,
                               regularizer=self.recurrent_regularizer,
                               constraint=self.recurrent_constraint)
    self.b_r = self.add_weight(shape=(self.units, ),
                               name='b_r',
                               initializer=self.bias_initializer,
                               regularizer=self.bias_regularizer,
                               constraint=self.bias_constraint)

    # Matrices for the z (update) gate
    self.U_z = self.add_weight(shape=(self.units, self.units),
                               name='U_z',
                               initializer=self.recurrent_initializer,
                               regularizer=self.recurrent_regularizer,
                               constraint=self.recurrent_constraint)
    self.W_z = self.add_weight(shape=(self.output_dim, self.units),
                               name='W_z',
                               initializer=self.recurrent_initializer,
                               regularizer=self.recurrent_regularizer,
                               constraint=self.recurrent_constraint)
    self.b_z = self.add_weight(shape=(self.units, ),
                               name='b_z',
                               initializer=self.bias_initializer,
                               regularizer=self.bias_regularizer,
                               constraint=self.bias_constraint)

    # Matrices for the proposal
    self.U_p = self.add_weight(shape=(self.units, self.units),
                               name='U_p',
                               initializer=self.recurrent_initializer,
                               regularizer=self.recurrent_regularizer,
                               constraint=self.recurrent_constraint)
    self.W_p = self.add_weight(shape=(self.output_dim, self.units),
                               name='W_p',
                               initializer=self.recurrent_initializer,
                               regularizer=self.recurrent_regularizer,
                               constraint=self.recurrent_constraint)
    self.b_p = self.add_weight(shape=(self.units, ),
                               name='b_p',
                               initializer=self.bias_initializer,
                               regularizer=self.bias_regularizer,
                               constraint=self.bias_constraint)

    # For creating the initial state:
    # input to the pointer network is its own output, therefore
    # use output_dim to initialize states.
    self.W_s = self.add_weight(shape=(self.output_dim, self.units),
                               name='W_s',
                               initializer=self.recurrent_initializer,
                               regularizer=self.recurrent_regularizer,
                               constraint=self.recurrent_constraint)

    self.input_spec = [
        InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))]
    self.built = True
```

**Call** 部分代码实现如下：

```python
def call(self, x):        
    # x is the hidden state of encoder.
    self.x_seq = x

    # a_ij = softmax(V_a^T tanh(W_a \cdot s_{t-1} + U_a \cdot h_t))
    # apply a dense layer over the time dimension of the sequence 
    # (get the U_a \cdot h_t) part).
    self._uxpb = _time_distributed_dense(self.x_seq, self.U_a, b=self.b_a,
                                         input_dim=self.input_dim,
                                         timesteps=self.timesteps,
                                         output_dim=self.output_dim)        
    x = self._uxpb        
    return super(AttentionPointer, self).call(x)
```

**Get_initial_state** 部分代码实现如下：

```python
def get_initial_state(self, inputs):
    """ initialize z0 and s_tp0""" 

    # inputs[:,0] has shape (batchsize, input_dim)
    # W_s has shape (input_dim, units)
    # s0 has shape (batchsize, units)
    s0 = activations.tanh(K.dot(inputs[:, 0], self.W_s)) 
    self.stm2 = s0

    # initialize output of shape (batchsize,output_dim)
    y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
    y0 = K.sum(y0, axis=(1, 2))  # (samples, )
    y0 = K.expand_dims(y0)  # (samples, 1)
    y0 = K.tile(y0, [1, self.output_dim]) # (batchsize, output_dim)

    # initialize update gate and proposal instead of hidden and cell, 
    # so that it can be back-propagated. 
    # W_z/W_r/W_p has shape (output_dim, units), 
    # therefore K.dot(y0, W_z/W_r/W_p) has shape (batchsize, units)
    # U_z/U_r/U_p has shape (units, units), 
    # therefore K.dot(s0, U_z/U_r) has shape (batchsize, units)
    # b_z/b_r/b_p has shape (units, )
    # therefore z0/r0 has shape (batchsize, units)       
    z0 = activations.sigmoid( K.dot(y0, self.W_z) + K.dot(s0, self.U_z)
                              + self.b_z )                
    r0 = activations.sigmoid( K.dot(y0, self.W_r) + K.dot(s0, self.U_r)
                              + self.b_r )

    # r0*s0 has shape (batchsize, unit), '*' is element-wise multiplication
    # therefore s_tp has shape (batchsize, units)
    s_tp0 = activations.tanh( K.dot(y0, self.W_p) 
            + K.dot((r0 * s0), self.U_p) + self.b_p )        
    return [z0, s_tp0]
```

**Step** 部分代码实现如下：

```python
def step(self, x, states):
    """ get the previous hidden state of the decoder from states = [z, s_p]
        alignment model:
            waStm1 = W_a \dot s_{t-1}
            uaHt = U_a \dot h_t
            tmp = tanh(waStm1 + uaHt)
            e_ij = V_a^T * tmp
            vector of length = timestep is: u_t = softmax(e_tj)
    """
    atm1 = x
    ztm1, s_tpm1 = states

    # old hidden state:
    # shape (batchsize, units)
    stm1 = (1 - ztm1) * self.stm2 + ztm1 * s_tpm1

    # shape (batchsize, timesteps, units)
    _stm = K.repeat(stm1, self.timesteps)

    # shape (batchsize, timesteps, output_dim)
    _Wxstm = K.dot(_stm, self.W_a)

    # calculate the attention probabilities:
    # self._uxpb has shape (batchsize, timesteps, output_dim)
    # V_a has shape (output_dim, )
    # after K.expand_dims it is (output_dim, 1)
    # therefore et has shape (batchsize, timesteps, 1)
    et = K.dot(activations.tanh(_Wxstm + self._uxpb),
               K.expand_dims(self.V_a))
    at = K.exp(et)
    at_sum = K.sum(at, axis=1)
    at_sum_repeated = K.repeat(at_sum, self.timesteps)
    at /= at_sum_repeated  # vector of shape (batchsize, timesteps, 1)

    # reset gate:
    rt = activations.sigmoid( K.dot(atm1, self.W_r) + K.dot(stm1, self.U_r)
                              + self.b_r )
    # update gate:
    zt = activations.sigmoid( K.dot(atm1, self.W_z) + K.dot(stm1, self.U_z)
                              + self.b_z )
    # proposal hidden state:
    s_tp = activations.tanh( K.dot(atm1, self.W_p) 
                             + K.dot((rt * stm1), self.U_p) + self.b_p )
    yt = activations.softmax(at)

    if self.return_probabilities:
        return at, [zt, s_tp]
    else:
        return yt, [zt, s_tp]
```

订制层 class 的完整代码比较长，有兴趣的童鞋请开小窗。

### 三种递归神经网络LSTM, attention network, pointer network 解决排序问题的性能比较：

**测试案例设计**：

随机生成一万组8位的整数序列，序列的元素集合为0-9。例如：

x0|x1|x2|x3|x4|x5|x6|x7
--|--|--|--|--|--|--|--
9|7|4|1|6|2|5|3
9|2|4|1|3|8|7|0
5|9|1|0|8|3|2|4
9|5|3|1|2|7|0|6
7|3|1|9|0|8|2|4
6|4|8|7|1|0|2|3
4|8|1|2|9|5|6|3
6|5|7|8|0|3|4|1
8|9|7|0|1|3|2|5
3|9|6|2|0|7|4|1

80% 作为训练集，10%验证集，10%测试集。

目标序列为原始序列对应的倒排顺序序列，例如：

y0|y1|y2|y3|y4|y5|y6|y7
--|--|--|--|--|--|--|--
0|1|4|7|2|6|3|5
0|5|3|6|4|1|2|7
2|0|6|7|1|4|5|3
0|3|4|6|5|1|7|2
2|4|6|0|7|1|5|3
2|3|0|1|6|7|5|4
4|1|7|6|0|3|2|5
2|3|1|0|7|5|4|6
1|0|2|7|6|4|5|3
4|0|2|5|7|1|3|6

由于pointer network返回的是分类器各个分类的概率，所以在把目标数组传递给神经网络前需要做离散处理：

```python
import numpy as np
from keras.utils.np_utils import to_categorical
y_ = np.loadtxt('../data/y_8.csv', delimiter=',', dtype=int)
# re-code target sequence into one-hot dummies:
y = []
for yy in y_[0:2,]:
    y.append(to_categorical(yy))
y = np.asarray(y)

print(y_[0])
print(y[0])
```

搭建模型：

```python
main_input = Input(shape=(n_steps, 1), name='main_input')
masked = Masking(mask_value=-1)(main_input)
enc = Bidirectional(LSTM(hidden_size, return_sequences=True), merge_mode='concat')(masked)
dropout = Dropout(rate=dropoutRate)(enc)
dec = AttentionPointer(hidden_size, return_probabilities=True)(dropout)
model = Model(inputs=main_input, outputs=dec)
model.summary()
```

输出模型结构：

![alt](doc/modelSummary.JPG)

**测试结果**：

编码器统一采用bidirectional LSTM，隐藏层为64节点；解码器分别采用LSTM，attention network和pointer network，分别用epoch = 100，序列长度为8，用相同训练集和验证集得到的结果为：

验证集准确率：

![alt](doc/val_acc.JPG)

验证集loss:

![alt](doc/val_loss.JPG)

测试集输出序列排序结果举例：

y0|y1|y2|y3|y4|y5|y6|y7
--|--|--|--|--|--|--|--
8|7|6|5|4|3|2|1
8|7|5|4|3|2|1|0
9|8|6|5|3|2|1|0
9|8|7|6|4|3|2|0
9|8|6|5|4|3|2|1
8|7|6|5|4|3|1|0
9|8|7|6|4|3|2|0
9|8|7|6|3|2|1|0
9|8|7|4|3|2|1|0
9|8|7|4|3|2|1|0

可以看到，pointer network的收敛更快，准确率更高。

**应用扩展**：

通过加入masking层，可以实现不同长度输入序列的排序。对目标序列的masking可以保证反向传播忽略被mask的元素。

需要具体代码实现的请小窗。
