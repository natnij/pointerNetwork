# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:42:15 2017

@author: Nat
"""
from keras import backend as K
import os
#from importlib import reload
#def set_keras_backend(backend):
#    if K.backend() != backend:
#        os.environ['KERAS_BACKEND'] = backend
#        reload(K)
#        assert K.backend() == backend
#set_keras_backend("theano")
from keras import regularizers, constraints, initializers, activations
from keras.layers import Dropout
from keras.layers.recurrent import Recurrent, _time_distributed_dense
from keras.engine import InputSpec
from keras.callbacks import Callback

class AttentionPointer(Recurrent):
    """ This is a modified version based on an attention network.
    compared to an attention network, 
    1. context vectors are dropped
    2. input to the ptrNet decoder is its own output (input from the 
       encoder is merely saved in a time_distributed_dense layer).
    3. for back propagation,  alpha, [z, s_p] values are returned in each step.
       z: update gate; s_p: proposal; alpha: the probability vector for output    
    
    Reference:
        Attention network: https://arxiv.org/pdf/1409.0473.pdf
        Pointer network: https://arxiv.org/pdf/1506.03134.pdf        
    Input: hidden state (return_sequence=True) from one or multiple 
        keras LSTM cells as encoder
    Output:
        based on the input dictionary, output a new sequence of the 
        same length and from the same dictionary.
    """
    def __init__(self, units,
                 return_probabilities=True,
                 name='AttentionPointer',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        
        self.units = units
        self.output_dim = units
        self.return_probabilities = return_probabilities
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionPointer, self).__init__(**kwargs)
        self.name = name
        self.return_sequences = True  # must return sequences

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

    def call(self, x):        
        # x is the hidden state of encoder.
        self.x_seq = x

        # attention model:
        # a_ij = softmax(V_a^T tanh(W_a \dot s_{t-1} + U_a \dot h_t))
        # apply a dense layer over the time dimension of the sequence 
        # (get the U_a \dot h_t) part).
        self._uxpb = _time_distributed_dense(self.x_seq, self.U_a, b=self.b_a,
                                             input_dim=self.input_dim,
                                             timesteps=self.timesteps,
                                             output_dim=self.output_dim)        
        x = self._uxpb        
        return super(AttentionPointer, self).call(x)

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

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_probabilities:
            return (None, self.timesteps, self.timesteps)
        else:
            return (None, self.timesteps, self.output_dim)

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.units,
            'return_probabilities': self.return_probabilities
        }
        base_config = super(AttentionPointer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MyCallback(Callback):

    def __init__(self, validation_data=None):
        # validation_data[0]: input validation dataset
        # validation_data[1]: target validation dataset
        # validation_data[2]: sample weight of validation set
        self.val_data = validation_data
        
    def on_train_begin(self, logs={}):
        self.my_masked_val_losses = []
        self.my_unmasked_val_losses = []
    
    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.val_data[0])
        target = self.val_data[1]
        masked, unmasked = myCategorical_crossentropy(pred, target)
        self.my_masked_val_losses.append(masked)
        self.my_unmasked_val_losses.append(unmasked)

def normalize(ar):
    smallNr = np.power(10.0, -14)
    ar = np.array(ar)
    s = np.sum(ar, axis=1)
    s = s.reshape(-1,1).repeat(ar.shape[1], axis=1)
    ar = ar / s
    ar = np.clip(ar, smallNr, 1.0 - smallNr)
    ar = ar.reshape([1, ar.shape[0], ar.shape[1]])
    return ar

def myCategorical_crossentropy(pred, target):
    """ manual calculation of loss in case of masked input and output 
        same calculation as in 
        https://github.com/fchollet/keras/blob
            /85fe6427a50f6def7777413a4bc8b286f7ab8f8d/keras
            /backend/tensorflow_backend.py#L2729
        input: probability
    """
    pred_ = pred.tolist()
    pred_ = np.concatenate([normalize(x) for x in pred_])
    unmasked_loss = - (np.sum(target * np.log(pred_)) 
                        / (pred.shape[0] * pred.shape[1]))
    masked_loss = - (np.sum(target * np.log(pred_)) 
                        / (len(pred[pred != 0].tolist()) / pred.shape[2]))
    return masked_loss, unmasked_loss

def allocatePosition(element):
    """ utility function to avoid duplicate output """
    a = []
    for i in np.arange(len(element)):
        idx = np.argmax(element[i])
        while idx in a:
            element[i][idx] = -1
            idx = np.argmax(element[i])
        a.append(idx)
    return a
    
def sortInput(seq, pointer):
    o = list(zip(list(pointer), seq))
    a = [i[1] for i in sorted(o)]
    return a

def preProcess(fileName, n_steps):
    """ utility function to mask input and target datasets. """
    XX = None
    YY = None
    for i in fileName:
        x_file = 'x_{}.csv'.format(i)
        y_file = 'y_{}.csv'.format(i)
        x = np.loadtxt(x_file, delimiter=',', dtype=int)
        y_ = np.loadtxt(y_file, delimiter=',', dtype=int)
        # to avoid normal zero values, since it's used also as padding.        
        x = x + 1
        
        # re-code target sequence into one-hot dummies:
        y = []
        for yy in y_:
            y.append(to_categorical(yy))
        y = np.asarray(y)
        
        # expand x into shape of (batchsize, timestep, # features)
        # because keras LSTM requires input to have three axes.
        if len(x.shape) < 3:
            x = np.expand_dims(x, axis=2)

        # to enable different input length: 
        # use maxlen as accepted input shape; 
        # zero-pad or truncate if actual input shape is different.
        X = preprocessing.sequence.pad_sequences(x, 
                        padding='post', truncating='post', maxlen=n_steps)
        Y = np.zeros([y.shape[0], y.shape[1], n_steps])
        # pad dim[2]
        for j in np.arange(y.shape[0]):
            Y[j] = preprocessing.sequence.pad_sequences(y[j], 
                        padding='post', truncating='post', maxlen=n_steps)
        # pad dim[1]
        Y = preprocessing.sequence.pad_sequences(Y, 
                        padding='post', truncating='post', maxlen=n_steps)            

        if XX is None:
            XX = X
            YY = Y
        else:
            XX = np.concatenate((XX, X), axis=0)
            YY = np.concatenate((YY, Y), axis=0)

    return XX, YY

def splitSet(dfX, dfY, ratio):
    """ utility function to split trainset, validation set and test set
        input: ratio: (trainset ratio, val_set ratio, test set ratio)
            has to add up to 1.
        output: XX_train, XX_val, XX_test, YY_train, YY_val, YY_test
    """
    shape = dfX.shape[0]
    trainNr = int(shape * ratio[0])
    valNr = int(shape * (ratio[0] + ratio[1]))
    idx = np.random.choice(np.arange(shape),size=shape,replace=False)
    XX_train = dfX[[idx[:trainNr]]]
    YY_train = dfY[[idx[:trainNr]]]
    XX_val = dfX[[idx[trainNr: valNr]]]
    YY_val = dfY[[idx[trainNr: valNr]]]
    XX_test = dfX[[idx[valNr:]]]
    YY_test = dfY[[idx[valNr:]]]
    return XX_train, XX_val, XX_test, YY_train, YY_val, YY_test

def cropOutputs(x, n_steps):
    """
    https://stackoverflow.com/questions/46653322
        /how-to-use-masking-layer-to-mask-input-output-in-lstm-autoencoders
    This is to mask output at the same position as 
    the input to the model.
    x[0] is not-masked, decoded output from the model 
        with shape (batchsize, timestep, timestep)
    x[1] is masked input to the model (zero-padded) 
        with shape (batchsize, timestep, nr.features)
    """
    # in case input sequence is multi-dimensional with features
    x_ = x[1][:, :, 0]
    x_ = K.expand_dims(x_, axis=2)
    # padding = 1 for actual data in inputs, 0 for masked data(zero-padded)
    padding =  K.cast( K.not_equal(x_, 0), dtype=K.floatx())
    # if you have zeros for non-padded data, 
    # they will lose their backpropagation.
    # Pad last digits in each timestep
    rowPadding = K.squeeze(padding, axis=2)
    rowPadding = K.repeat(rowPadding, n_steps)
    y = x[0] * rowPadding
    return y

# test if it compiles with number sequencing from biggest to smallest
if __name__ == '__main__':
    os.chdir('./data')   
    from keras import preprocessing
    from keras.layers import Input, LSTM, Masking, Lambda, Bidirectional
    from keras.models import Model
    from keras.utils.np_utils import to_categorical
    import numpy as np
    import pandas as pd
    
    # params:
    n_steps = 10
    fileName = [5, 8]
    ratio = [0.8, 0.1, 0.1]  # split ratio between trainset, valset and testset
    batch_size = 100
    nr_epoch = 100
    hidden_size = 64
    dropoutRate = 0.1

    # read data:
    XX, YY = preProcess(fileName, n_steps)
    XX_train,XX_val,XX_test,YY_train,YY_val,YY_test = splitSet(XX, YY, ratio)
    validation_data = (XX_val, YY_val)
        
    # modeling:
    main_input = Input(shape=(n_steps, 1), name='main_input')
    masked = Masking(mask_value=-1)(main_input)
    enc = Bidirectional(LSTM(hidden_size, return_sequences=True), 
                                                merge_mode='concat')(masked)
    dropout = Dropout(rate=dropoutRate)(enc)
    dec = AttentionPointer(hidden_size, return_probabilities=True)(dropout)
    perf = Lambda(cropOutputs,arguments={'n_steps':n_steps}, 
                        output_shape=(n_steps,n_steps))([dec,main_input])
    model = Model(inputs=main_input, outputs=perf)
    model.summary()
    
    model.compile(optimizer='adadelta',
          loss='categorical_crossentropy', metrics=['accuracy'])
    
    my_callbacks = MyCallback(validation_data=validation_data)
    
    # ignore the Theano error messages, if any
    model.fit(XX_train, YY_train, epochs=nr_epoch, 
                        batch_size=batch_size,
                        validation_data=validation_data,
                        callbacks=[my_callbacks])
        
    p = model.predict(XX_test)
    # output loss and metrics
    evaluated = model.evaluate(XX_test, YY_test)    
    masked_loss, unmasked_loss = myCategorical_crossentropy(p, YY_test)
    
    history = model.history.history
    # tidy up the results and sort input sequence by learning result
    pPrime = p.tolist()
    pPrime = list(map(allocatePosition, pPrime))
    result = [sortInput(i, j) 
                    for i, j in list(zip(np.squeeze(XX_test, axis=2), pPrime))]
    result[0:5]
    
    for i in history.keys():
        pd.DataFrame(history[i]).plot(title=i)