# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, AveragePooling3D,\
#      Flatten, Dense, Dropout, Activation, BatchNormalization, Reshape, Lambda, LSTM, InputLayer, GlobalAveragePooling2D, CuDNNLSTM, TimeDistributed
# from tensorflow.keras import backend as K
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
# import tensorflow as tf
from keras import Sequential
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, AveragePooling3D, GlobalAveragePooling1D, \
     Flatten, Dense, Dropout, Activation, BatchNormalization, Reshape, Lambda, LSTM, InputLayer, GlobalAveragePooling2D, CuDNNLSTM, TimeDistributed, concatenate
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling2D)
from keras import backend as K
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet import MobileNet
from keras.layers import Input, Flatten, Dropout, Concatenate, Activation, Dense
from keras.layers import Convolution3D, MaxPooling3D, AveragePooling3D
from keras.layers import GlobalMaxPooling3D, GlobalAveragePooling3D


def CNN3D(inp_shape, nb_classes, k_size=(3,3,3), drop_rate=0):
    NB_AXIS = 1
    ROW_AXIS = 2
    COL_AXIS = 3
    data = Input(shape=inp_shape)

    x = Conv3D(filters=(64), kernel_size=k_size, strides=(1,1,1), padding='same', activation='relu')(data)
    x = MaxPooling3D(pool_size=(1,2,2), strides=(2,2,2))(x)    
    x = Dropout(drop_rate)(x)    
    
    x = Conv3D(filters=(128), kernel_size=k_size, strides=(1,1,1), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x)
    x = Dropout(drop_rate)(x)    
    
    x = Conv3D(filters=(256), kernel_size=k_size, strides=(1,1,1), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x)
    x = Dropout(drop_rate)(x)    

    x = Conv3D(filters=(512), kernel_size=k_size, strides=(1,1,1), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x)
    x = Dropout(drop_rate)(x)    

    block_shape = K.int_shape(x)
    x = AveragePooling3D(pool_size=(block_shape[NB_AXIS], block_shape[ROW_AXIS], block_shape[COL_AXIS]),strides=(1, 1, 1))(x)
    x = Flatten()(x)

    output = Dense(nb_classes, activation="softmax")(x)
    
    model = Model(data, output)

    return model


# Model inspired from https://arxiv.org/pdf/1412.0767.pdf
def CNN3D_dense(inp_shape, nb_classes, k_size=(3,3,3)):
    data = Input(shape=inp_shape)

    x = Conv3D(filters=(64), kernel_size=k_size, strides=(1,1,1), padding='same', activation='relu')(data)
    x = MaxPooling3D(pool_size=(1,2,2), strides=(2,2,2))(x)    
    
    x = Conv3D(filters=(128), kernel_size=k_size, strides=(1,1,1), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x)
    
    x = Conv3D(filters=(256), kernel_size=k_size, strides=(1,1,1), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x)

    x = Conv3D(filters=(512), kernel_size=k_size, strides=(1,1,1), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x)

    x = Flatten()(x)
    
    x = Dense(512, activation="relu")(x)
    x = Dense(512, activation="relu")(x)

    output = Dense(nb_classes, activation="softmax")(x)
    
    model = Model(data, output)
    
    return model


def CNN3D_lite(inp_shape, nb_classes):
    """
    Lite C3D Model + LSTM
    L2 Normalisation of C3D Lite Feature Vectors 
    3M Parameters
    Able to Run on the Jetson Nano at 8FPS

    # From https://github.com/patrickjohncyh/ibm-waldo/blob/master/2-MachineLearning/server-training/Models.py
    """

    model = tf.keras.Sequential()

    model.add(InputLayer(input_shape=inp_shape))
    model.add(Conv3D(32, 3,strides=(1,2,2), activation='relu', padding='same', name='conv1', input_shape=inp_shape))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='same', name='pool1'))
    
    model.add(Conv3D(64, 3, activation='relu', padding='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool2'))
    
    model.add(Conv3D(128, 3, activation='relu', padding='same', name='conv3a'))
    model.add(Conv3D(128, 3, activation='relu', padding='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(3,2,2), strides=(2,2,2), padding='valid', name='pool3'))
    
    model.add(Conv3D(128, 3, activation='relu', padding='same', name='conv4a'))
    model.add(Conv3D(128, 3, activation='relu', padding='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool4'))

    model.add(Reshape((2,384)))

    model.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
    model.add(LSTM(512, return_sequences=False,
                   input_shape= (2,384),
                   dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model


def mobilenetonly():
    # model = MobileNetV2( weights ='imagenet', include_top = True)
    # model = Model(inputs = model.input, outputs = model.get_layer('global_average_pooling2d_1').output )
    # #model.add(Dense(12, activation='softmax'))

    # model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, 3, padding='same'), input_shape=(16, 64, 96, 3), name='Conv')) # shape = (frames, width, height, channel)
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), name='MaxPooling')))
    model.add(TimeDistributed(Flatten(), name='Flatten_1'))
    model.add(LSTM(20, return_sequences=True, name='LSTM'))
    model.add(TimeDistributed(Dense(6, activation='linear'), name='Dense'))
    model.add(GlobalAveragePooling1D(name='average'))

    return model


def lrcn(inp_shape, nb_classes):
    """Build a CNN into RNN.
    Starting version from:
        https://github.com/udacity/self-driving-car/blob/master/
            steering-models/community-models/chauffeur/models.py
    Heavily influenced by VGG-16:
        https://arxiv.org/abs/1409.1556
    Also known as an LRCN:
        https://arxiv.org/pdf/1411.4389.pdf
    """
    print(inp_shape)
    model = Sequential()

    model.add(InputLayer(input_shape=inp_shape))

    model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2),
        activation='relu', padding='same')))
    model.add(TimeDistributed(Conv2D(32, (3,3),
        kernel_initializer="he_normal", activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(64, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(64, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(128, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(128, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(256, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(256, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(512, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(512, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Flatten()))

    model.add(Dropout(0.5))
    model.add(LSTM(256, return_sequences=False, dropout=0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    print(model.summary())

    return model


def Hanuman(inp_shape, nb_classes):

    def firemodule(x, filters, name="firemodule"):
        squeeze_filter, expand_filter1, expand_filter2 = filters
        squeeze = Convolution3D(squeeze_filter, (1, 1, 1), activation='relu', padding='same', name=name + "/squeeze1x1x1")(x)
        expand1 = Convolution3D(expand_filter1, (1, 1, 1), activation='relu', padding='same', name=name + "/expand1x1x1")(squeeze)
        expand2 = Convolution3D(expand_filter2, (3, 3, 3), activation='relu', padding='same', name=name + "/expand3x3x3")(squeeze)
        x = Concatenate(axis=-1, name=name)([expand1, expand2])
        return x

    img_input = Input(shape=inp_shape)

    x = Convolution3D(64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same", activation="relu", name='conv1')(img_input)
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='maxpool1', padding="valid")(x)

    x = firemodule(x, (16, 64, 64), name="fire2")
    x = firemodule(x, (16, 64, 64), name="fire3")

    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='maxpool3', padding="valid")(x)
    x = firemodule(x, (32, 128, 128), name="fire4")
    x = firemodule(x, (32, 128, 128), name="fire5")
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), name='maxpool5', padding="valid")(x)
    x = firemodule(x, (48, 192, 192), name="fire6")
    x = firemodule(x, (48, 192, 192), name="fire7")
    x = firemodule(x, (64, 256, 256), name="fire8")
    x = firemodule(x, (64, 256, 256), name="fire9")

    x = GlobalMaxPooling3D(name="maxpool10")(x)
    x = Dense(nb_classes, init='normal')(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x)

    print(model.summary())

    return model