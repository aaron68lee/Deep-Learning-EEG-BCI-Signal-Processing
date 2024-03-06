from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Dropout
from keras.layers import Conv2D,BatchNormalization,MaxPooling2D,AveragePooling2D,Reshape

def CNN(reg=0, pool_size=3, use_max_pool=True, dropout=0.5, dropout_on_conv=False, filters=[10 for _ in range(4)], affine_layer_sizes=[], num_filters=[25, 50, 100, 200], input_shape=(500, 1, 22), output_classes=4):
    model = Sequential()
    model.add(Conv2D(filters=num_filters[0], kernel_size=(filters[0],1), padding='same', activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.L2(reg)))
    if use_max_pool:
        model.add(MaxPooling2D(pool_size=(pool_size,1), padding='same')) # Read the keras documentation
    else:
        model.add(AveragePooling2D(pool_size=(pool_size,1), padding='same')) # Read the keras documentation
    model.add(BatchNormalization())
    if dropout_on_conv:
        model.add(Dropout(dropout))
    for filter, num_filter in zip(filters[1:], num_filters[1:]):
        model.add(Conv2D(filters=num_filter, kernel_size=(filter,1), padding='same', activation='relu', kernel_regularizer=regularizers.L2(reg)))
        if use_max_pool:
            model.add(MaxPooling2D(pool_size=(pool_size,1), padding='same')) # Read the keras documentation
        else:
            model.add(AveragePooling2D(pool_size=(pool_size,1), padding='same')) # Read the keras documentation
        model.add(BatchNormalization())
        if dropout_on_conv:
            model.add(Dropout(dropout))
    model.add(Flatten()) # Flattens the input
    for affine in affine_layer_sizes:
        model.add(Dense(affine, activation='relu', kernel_regularizer=regularizers.L2(reg))) 
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
    model.add(Dense(output_classes, activation='softmax', kernel_regularizer=regularizers.L2(reg))) # Output FC layer with softmax activation
    return model