from keras.layers import Input, Conv2D, BatchNormalization, Dropout, Activation, Add, Flatten, Dense, MaxPooling2D
from keras import regularizers
from keras.models import Model

def residual_block(x, num_filter, kernel_size=5, dropout=0.5, reg=0, input_size = (400, 1, 22)):
    shortcut = Conv2D(filters=num_filter, kernel_size=(1, 1), padding="valid", input_shape=input_size, kernel_regularizer=regularizers.L2(reg))(x)
    x = Conv2D(filters=num_filter, kernel_size=(kernel_size,1), padding='same', input_shape=input_size, kernel_regularizer=regularizers.L2(reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=num_filter, kernel_size=(kernel_size,1), padding='same', input_shape=input_size, kernel_regularizer=regularizers.L2(reg))(x)
    x = BatchNormalization()(x)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

# ResNet: One initial convolution block, followed by residual blocks, followed by affine
def ResNetCustom(reg=1e-3, pool_size=3, dropout=0.5, 
                 first_conv_size=10,
                 first_conv_num=25,
                 res_filter_sizes=[5 for _ in range(4)], 
                 res_num_filters=[25, 50, 100, 200], 
                 affine_layer_sizes=[], 
                 input_shape=(400, 1, 22), 
                 output_classes=4):
    input_layer = Input(shape=input_shape)
    output_layer = Conv2D(filters=first_conv_num, kernel_size=(first_conv_size,1), padding='same', activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.L2(reg)) (input_layer)
    output_layer = MaxPooling2D(pool_size=(pool_size,1), padding='same') (output_layer)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Dropout(dropout)(output_layer)
    for num_filter, kernel_size in zip(res_num_filters, res_filter_sizes):
        output_layer = residual_block(output_layer, num_filter=num_filter,
                                      kernel_size=kernel_size,
                                      dropout=dropout,
                                      reg = reg,
                                      input_size=input_shape)
        output_layer = MaxPooling2D(pool_size=(pool_size,1), padding='same') (output_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Dropout(dropout)(output_layer)

    output_layer = Flatten()(output_layer) # Flattens the input
    for affine in affine_layer_sizes:
        output_layer = Dense(affine, activation='relu', kernel_regularizer=regularizers.L2(reg))(output_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Dropout(dropout)(output_layer)
    output_layer = Dense(output_classes, activation='softmax', kernel_regularizer=regularizers.L2(reg))(output_layer) # Output FC layer with softmax activation

    return Model(inputs=input_layer, outputs=output_layer)