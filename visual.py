import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class_names = ['Anger', 'Hate', 'Scary', 'Happy', 'Sad', 'Shack', 'Middle']

# Read CSV
train_csv_set = pd.read_csv('train.csv', names=('labels', 'features'))
train_csv_set = train_csv_set[1:]  # drop column name

train_csv_set = train_csv_set.drop(train_csv_set.index[59])

# Separate train and validate data
train_csv_separate = [train_csv_set[(train_csv_set['labels'].eq('0'))],
                      train_csv_set[(train_csv_set['labels'].eq('1'))],
                      train_csv_set[(train_csv_set['labels'].eq('2'))],
                      train_csv_set[(train_csv_set['labels'].eq('3'))],
                      train_csv_set[(train_csv_set['labels'].eq('4'))],
                      train_csv_set[(train_csv_set['labels'].eq('5'))],
                      train_csv_set[(train_csv_set['labels'].eq('6'))]]

# Transfer to numpy-array
train_data = []
train_labels = []
for train_features, train_label in zip(train_csv_set['features'], train_csv_set['labels']):
    train_data.append(np.array(train_features.split(' ')).astype(np.int))
    train_labels.append(int(train_label))

train_data = np.array(train_data)
train_data = train_data.reshape([-1, 48, 48])

train_data

#plt.figure(figsize=(10, 10))

#shift = 51
#for i in range(25):
#    plt.subplot(5, 5, i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_data[i+shift], cmap="gray")
#    plt.xlabel(class_names[train_labels[i+shift]])
#plt.show()

plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(train_data[60], cmap="gray")
plt.xlabel(class_names[train_labels[60]])
plt.show()


