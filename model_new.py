import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GRU, Dense, Bidirectional
from tensorflow import keras


def CRNN_Model():
    mel_input = keras.Input(shape=(802, 100), name="mel_input")
    x = mel_input

    input_layer = tf.keras.layers.Reshape((802, 100, 1))(x)

    x = Conv2D(64, (1, 3), padding='same', activation='relu')(input_layer)
    x = MaxPooling2D((1, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (1, 11), padding='same', strides=(1, 1), dilation_rate=(5, 1), activation='relu')(x)
    x = MaxPooling2D((1, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(16, (1, 11), padding='same', strides=(1, 1), dilation_rate=(5, 1), activation='relu')(x)
    x = MaxPooling2D((1, 2))(x)
    x = BatchNormalization()(x)

    _, _, sx, sy = x.shape
    x = tf.keras.layers.Reshape((-1, int(sx * sy)))(x)

    x = Bidirectional(GRU(80, return_sequences=True))(x)
    x = BatchNormalization()(x)

    x = Bidirectional(GRU(40, return_sequences=True))(x)
    x = BatchNormalization()(x)

    pred = Dense(2, activation='sigmoid')(x)

    model = keras.Model(inputs=[mel_input], outputs=[pred])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=[keras.losses.BinaryCrossentropy()], metrics=['binary_accuracy']
    )

    model.summary()

    return model
