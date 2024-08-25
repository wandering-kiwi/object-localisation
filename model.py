from timeit import default_timer as timer
import data
import os

print('importing tensorflow')
start = timer()
import tensorflow as tf
keras=tf.keras

end = timer()
print('')
print('tensorflow imported: ' + str(end - start))

# get data in form of x_train, x_test, y_train, y_test
print('loading cached data')
start = timer()
x_train, y_train, x_test, y_test = data.load_data(data.path)
end = timer()
print('data loaded: ' + str(end - start))


start = timer()
print('creating model')

model = tf.keras.Sequential(
    [
    tf.keras.layers.Input(shape=(640, 640, 3)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax")
]
)
model.summary()
end = timer()
print('model created: ' + str(end - start))