import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, AveragePooling2D,  Reshape
from keras.layers import Conv1D, Conv2D,  DepthwiseConv2D
from keras.layers import BatchNormalization, LayerNormalization
from keras.layers import Add, Concatenate, Lambda, Input, Permute
from keras.constraints import max_norm
from keras import backend as K
import math

def MMANet(n_classes, in_chans=22, in_samples=1125, n_Slice = 5,diff = 1,eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=8, eegn_dropout=0.3,
           tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,tcn_activation='elu'):

    input_1 = Input(shape=(1, in_chans, in_samples))  # TensorShape([None, 1, C, T])
    input_2 = Permute((3, 2, 1))(input_1)
    regRate = .25
    numFilters = eegn_F1
    F2 = numFilters * eegn_D

    # EEGIA Module
    block1 = EEGIA_block(input_layer=input_2, F1=eegn_F1, D=eegn_D,
                         kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                         in_chans=in_chans, dropout=eegn_dropout)

    # ablation study of EEGIA Module
    # block1 = EEGIA_block_Ablation(input_2) #ablation study of EEGIA Module

    # Multi-Head attention
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)
    block1 = attention_block(block1)

    # ablation study of PTSA Module
    # outs = TCN_block(input_layer=block1, input_dimension=F2, depth=tcn_depth, kernel_size=tcn_kernelSize, filters=tcn_filters,
    #                  dropout=tcn_dropout, activation=tcn_activation)
    # out = Lambda(lambda x: x[:, -1, :])(outs)
    # dense = Dense(n_classes, name='dense', kernel_constraint=max_norm(regRate))(out)
    # softmax = Activation('softmax', name='softmax')(dense)

    # PTSA module
    block2 = SequenceAttention(block1, num_slicedseq=n_Slice, D=diff)
    sw_concat = []
    for i in range(n_Slice ):
        block3 = block2[:,:,:,i]
        block3 = TCN_block(input_layer=block3, input_dimension=F2, depth=tcn_depth,
                            kernel_size = tcn_kernelSize, filters = tcn_filters,
                            dropout = tcn_dropout, activation = tcn_activation)
        block3 = Lambda(lambda x: x[:, -1, :])(block3)
        sw_concat.append(Dense(n_classes, kernel_constraint = max_norm(regRate))(block3))

    if len(sw_concat) > 1:
        sw_concat = tf.keras.layers.Average()(sw_concat[:])
    else:
        sw_concat = sw_concat[0]
    softmax = Activation('softmax', name='softmax')(sw_concat)

    return Model(inputs=input_1, outputs=softmax)


def EEGIA_block(input_layer, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22, dropout=0.1):

    F2= F1*D
    block1 = EEGChannelAttention(9)(input_layer) # ablation of ECA
    block1 = MultiConv_Block(depth=1,input_layer=block1,F1=F1,kernLength=kernLength)#Inception结构的多尺度卷积并行
    block2 = DepthwiseConv2D((1, in_chans), use_bias = False, 
                                    depth_multiplier = D,
                                    data_format='channels_last',
                                    depthwise_constraint = max_norm(1.))(block1)
    block2 = BatchNormalization(axis = -1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8,1),data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    block3 = Conv2D(F2, (16, 1),
                            data_format='channels_last',
                            use_bias = False, padding = 'same')(block2)
    block3 = BatchNormalization(axis = -1)(block3)
    block3 = Activation('elu')(block3)
    
    block3 = AveragePooling2D((poolSize,1),data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3

def EEGIA_block_Ablation(input_layer):
    F2 = 32
    block1 = Conv2D(16, (1, 1),
                    data_format='channels_last',
                    use_bias=False, padding='same')(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)
    block2 = DepthwiseConv2D((1, 22), use_bias=False,
                             depth_multiplier=2,
                             data_format='channels_last',
                             depthwise_constraint=max_norm(1.))(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
    block2 = Dropout(0.1)(block2)
    block3 = Conv2D(F2, (16, 1),
                    data_format='channels_last',
                    use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)

    block3 = AveragePooling2D((7, 1), data_format='channels_last')(block3)
    block3 = Dropout(0.1)(block3)
    return block3

def TCN_block(input_layer,input_dimension,depth,kernel_size,filters,dropout,activation='relu'):

    block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=1,activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=1,activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if(input_dimension != filters):
        conv = Conv1D(filters,kernel_size=1,padding='same')(input_layer)
        added = Add()([block,conv])
    else:
        added = Add()([block,input_layer])
    out = Activation(activation)(added)
    
    for i in range(depth-1):
        block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=2**(i+1),activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=2**(i+1),activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)
        
    return out


def ConvAttend_Block(input_layer, F1, kernLength):
    block = Conv2D(F1, (kernLength, 1), padding = 'same',data_format='channels_last',use_bias = False)(input_layer) #提取多频率尺度信息
    block = BatchNormalization(axis=-1)(block)
    block = Conv2D(1, (1, 1), padding = 'same',data_format='channels_last',use_bias = False)(block)#调整特征图维度
    return block

def Inception_Attend_Block(input_layer, F1=8, kernLength=64):
    block1 = ConvAttend_Block(input_layer=input_layer,F1=F1, kernLength=int(kernLength/4))
    block2 = ConvAttend_Block(input_layer=input_layer,F1=F1, kernLength=int(kernLength/8))
    block3 = ConvAttend_Block(input_layer=input_layer,F1=F1, kernLength=int(kernLength/16))
    block = Concatenate(axis=3)([block1,block2,block3])
    block = ChannelAttention()(block) # ablation of CA
    block = Conv2D(F1, (1, 1), padding = 'same',data_format='channels_last',use_bias = False)(block)
    block = BatchNormalization(axis = -1)(block)
    return block

def MultiConv_Block(depth,input_layer,F1=8,kernLength=64):
    block = Inception_Attend_Block(input_layer=input_layer, F1=F1, kernLength=kernLength)
    for i in range(depth-1):
        block = Inception_Attend_Block(input_layer=block, F1=F1, kernLength=kernLength)
    return block



def SequenceAttention(input_seq,num_slicedseq,D):
    len_slicedseq = input_seq.shape[1]-D*(num_slicedseq-1)
    seq = []
    input_seq = tf.expand_dims(input_seq,axis = 3)
    for i in range(num_slicedseq):
        std = i*D
        end = len_slicedseq+i*D
        if i == 0:
            seq = input_seq[:,std:end,:,:]
        else:
            seq = tf.keras.layers.Concatenate()([seq,input_seq[:,std:end,:,:]])
    seq = ChannelAttention()(seq)
    return seq


class EEGChannelAttention(tf.keras.Model):
    def __init__(self, L):
        super().__init__()
        self.L = L
    def build(self, input_shape):

        self.W_channel = self.add_weight(shape=(input_shape[2], 1, self.L),
                                         initializer='random_normal',
                                         trainable=True,
                                         name='channel_weight')
    def call(self, inputs):
        input_attened = []
        W_channel_softmax = tf.nn.softmax(self.W_channel, axis=0)
        inputs_permuted = Permute((3, 2, 1))(inputs)  # shape[none,channel,C,T]

        for i in range(self.L):
            if i == 0:
                input_attened = W_channel_softmax[:, :, i] * inputs_permuted
            else:
                input_attened = Concatenate(axis=1)([input_attened, W_channel_softmax[:, :, i] * inputs_permuted])

        input_attened = Permute((3, 2, 1))(input_attened)
        return input_attened


class ChannelAttention(tf.keras.Model):
    def __init__(self):
        super().__init__()
    def build(self, input_shape):
        self.W = self.add_weight(shape=(1, 1, input_shape[3]),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='FreAtten_weight')
    def call(self, inputs):
        W_softmax = tf.nn.softmax(self.W)
        out = inputs * W_softmax
        return out

def mhlsa_block(input_feature, key_dim=8, num_heads=2, dropout=0.5):
    x = LayerNormalization(epsilon=1e-6)(input_feature)
    NUM_PATCHES = input_feature.shape[1]
    diag_attn_mask = 1 - tf.eye(NUM_PATCHES)
    diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)
    x = MultiHeadAttention_LSA(key_dim=key_dim, num_heads=num_heads, dropout=dropout)(x, x, attention_mask=diag_attn_mask)
    x = Dropout(0.3)(x)
    mha_feature = Add()([input_feature, x])
    return mha_feature

class MultiHeadAttention_LSA(tf.keras.layers.MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable=True)
    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1.0 / self.tau)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores

def attention_block(net):
    in_sh = net.shape  # dimensions of the input tensor
    in_len = len(in_sh)
    expanded_axis = 3  # defualt = 3
    if (in_len > 3):
        net = Reshape((in_sh[1], -1))(net)
    net = mhlsa_block(net)
    if (in_len == 3 and len(net.shape) == 4):
        net = K.squeeze(net, expanded_axis)
    elif (in_len == 4 and len(net.shape) == 3):
        net = Reshape((in_sh[1], in_sh[2], in_sh[3]))(net)
    return net
