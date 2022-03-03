import numpy as np
import sklearn
import tensorflow as tf


for file in 'data':

    with open('data' + file, 'rb') as f:
        inp_mat = np.load(f)
        out_mat = np.load(f)

    inp_mat, out_mat = sklearn.utils.shuffle(inp_mat, out_mat)
    action = ['left', 'right', 'up', 'down', 'stop']
    out_mat = tf.keras.utils.to_categorical(out_mat, 5)
    inn = int(inp_mat.shape[0] * .85)
    x_train, y_train, x_test, y_test = inp_mat[:inn], out_mat[:inn], inp_mat[inn:], out_mat[inn:]
    if 'window_5' in file:
        WINDOW_SIZE = 5
    elif 'window_7' in file:
        WINDOW_SIZE = 7
    elif 'window_11' in file:
        WINDOW_SIZE = 11
    else:
        WINDOW_SIZE = 9
    kern = (WINDOW_SIZE - 1) / 2
    # change shape based on window grid size
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(filters=int(WINDOW_SIZE * WINDOW_SIZE), kernel_size=int(kern), activation='relu',
                                   input_shape=[WINDOW_SIZE, WINDOW_SIZE, 1]))
    cnn.add(tf.keras.layers.Conv2D(filters=int(WINDOW_SIZE * WINDOW_SIZE), kernel_size=int(kern), activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=WINDOW_SIZE))
    # cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    # cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    cnn.add(tf.keras.layers.Dropout(0.2))
    cnn.add(tf.keras.layers.Dense(units=5, activation='softmax'))
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = cnn.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))

    fname = file.split('.')[0]
    # SAVE MODEL

    model_json = cnn.to_json()
    with open('models/' + fname + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    cnn.save_weights('models/' + fname + ".h5")
    print("Saved model to disk")
