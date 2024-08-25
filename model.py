from timeit import default_timer as timer
import data
import os

print('import tensorflow')
start = timer()
import tensorflow as tf

end = timer()
print('Elapsed time: ' + str(end - start))
# get data in form of x_train, x_test, y_train, y_test
print('load cached data')
start = timer()
x_train, y_train, x_test, y_test = data.load_data(data.path)

end = timer()
print('Elapsed time: ' + str(end - start))

print(x_train[0].shape)
# keras=tf.keras
# model = keras.Sequential([
#     keras.layers.Input(shape=(640, 640, 3)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(128, activation=tf.nn.relu),
#     keras.layers.Dense(2, activation=tf.nn.softmax)
# ])