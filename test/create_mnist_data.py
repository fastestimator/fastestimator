import numpy as np
import tensorflow as tf
import os
import pandas as pd
from PIL import Image
import multiprocessing
from multiprocessing import Process
import time

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_names = []
y_names = []
x_train = x_train[:800]
y_train = y_train[:800]
x_test = x_test[:200]
y_test = y_test[:200]

data_dir = os.path.join(os.path.abspath('.'), 'mnist')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

num_examples = len(y_train) + len(y_test)

for i in range(len(y_train)):
    y_names.append(y_train[i])
    x_names.append(os.path.join(data_dir, 'train_' + str(i) + '.png'))

for i in range(len(y_test)):
    y_names.append(y_test[i])
    x_names.append(os.path.join(data_dir, 'val_' + str(i) + '.png'))

df = pd.DataFrame(data={'x': x_names[:len(y_train)], 'y': y_names[:len(y_train)]})
df.to_csv('annotation_train.csv', index=False)
df = pd.DataFrame(data={'x': x_names[len(y_train):], 'y': y_names[len(y_train):]})
df.to_csv('annotation_val.csv', index=False)

def write_images(ids, x, x_names):
    for i in ids:
        img = Image.fromarray(x[i])
        img.save(x_names[i])

num_cpu = multiprocessing.cpu_count()

ids = {}
z = 0
examples_per_process = num_examples // num_cpu

while True:
    start = z * examples_per_process
    if start >= num_examples:
        break
    end = (z + 1) * examples_per_process
    if end > num_examples:
        end = num_examples
    ids[z] = range(start, end)
    z += 1

x = np.vstack((x_train, x_test))
processes = [Process(target=write_images, args=(ids[i], x, x_names)) for i in ids]
for p in processes:
    p.start()

for p in processes:
    p.join()

