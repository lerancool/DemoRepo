import tensorflow as tf
import numpy as np




import copy
a = [1, 2, 3, 4, ['a', 'b']]
b = a
c = copy.copy(a)
d = copy.deepcopy(a)
a.append(5)
a[4].append('c')


print("a = ", a)
print("b = ", b)
print("c = ", c)
print("d = ", d)

exit()


test = np.array([[1, 2, 3],
                 [2, 3, -5],
                 [4, 7, 1]])

# test = np.array([[1, 2, 6],
#                  [2, 3, 7],
#                  [1, 4, 5]
#                  ])

print(np.linalg.matrix_rank(test))

exit()

m = np.matrix([[1, 2, 3],
               [4, 5, 6]])
a = np.array([[1, 2, 3],
              [4, 5, 6]])

tf.keras.layers.Conv3D()


def build_modle():
    model = tf.keras.Sequential()

    conv_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1])
    model.add(conv_1)
    return model

    model = build_modle()
