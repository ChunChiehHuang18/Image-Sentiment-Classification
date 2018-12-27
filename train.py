import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator


datetime_path = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
MODEL_PATH = 'vgg-like-' + datetime_path + '.h5'

checkpoint_path = "Training/cp-{epoch:03d}-{val_acc:.2f}.h5"

es_fit_CallBack = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=3)

es_fit_gen_CallBack = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=20)

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph',
                                         histogram_freq=0,
                                         write_images=True,
                                         write_graph=True,
                                         write_grads=True,
                                         batch_size=32)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 monitor='val_loss',
                                                 save_best_only=True,)


datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    fill_mode='nearest')

# labels count [3062, 355, 3188, 5529, 3692, 2376, 3798]
# Read CSV
train_csv_set = pd.read_csv('train.csv', names=('labels', 'features'))
train_csv_set = train_csv_set[1:]  # drop column name


# Separate train and validate data
train_csv_separate = [train_csv_set[(train_csv_set['labels'].eq('0'))],
                      train_csv_set[(train_csv_set['labels'].eq('1'))],
                      train_csv_set[(train_csv_set['labels'].eq('2'))],
                      train_csv_set[(train_csv_set['labels'].eq('3'))],
                      train_csv_set[(train_csv_set['labels'].eq('4'))],
                      train_csv_set[(train_csv_set['labels'].eq('5'))],
                      train_csv_set[(train_csv_set['labels'].eq('6'))]]

train_csv_set = pd.DataFrame()
validate_csv_set = pd.DataFrame()
for train_group in train_csv_separate:
    train, validate = train_test_split(train_group, test_size=0.15)
    train_csv_set = train_csv_set.append(train)
    validate_csv_set = validate_csv_set.append(validate)

# Shuffle
#train_csv_set, validate_csv_set = train_csv_set.sample(frac=1), validate_csv_set.sample(frac=1)

# Transfer to numpy-array
train_data = []
train_labels = []
validate_data = []
validate_labels = []
for train_features, train_label in zip(train_csv_set['features'], train_csv_set['labels']):
    train_data.append(np.array(train_features.split(' ')).astype(np.int))
    train_labels.append(train_label)

for validate_features, validate_label in zip(validate_csv_set['features'], validate_csv_set['labels']):
    validate_data.append(np.array(validate_features.split(' ')).astype(np.int))
    validate_labels.append(validate_label)

train_data, validate_data = np.array(train_data), np.array(validate_data)
train_data, validate_data = train_data.reshape([-1, 48, 48, 1]) / 255.0, validate_data.reshape([-1, 48, 48, 1]) / 255.0
# ----- cal class_weights -----
class_weights = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
# ----- cal class_weights -----
train_labels, validate_labels = keras.utils.to_categorical(np.array(train_labels), num_classes=7),\
                                keras.utils.to_categorical(np.array(validate_labels), num_classes=7)

# data augmentation
datagen.fit(train_data)

def create_vgg(model):
    model.add(keras.layers.Conv2D(16, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(16, (3, 3), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(32, (3, 3), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(7))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('softmax'))

    adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()


if __name__ == '__main__':

    model = keras.Sequential()
    create_vgg(model)
    model.fit(train_data, train_labels, epochs=1000, batch_size=32,
              validation_data=(validate_data, validate_labels),
              callbacks=[cp_callback, tbCallBack, es_fit_CallBack])
              #class_weight={0: class_weights[0], 1: class_weights[1], 2: class_weights[2],
              #              3: class_weights[3], 4: class_weights[4], 5: class_weights[5],
              #              6: class_weights[6]})
    model.fit_generator(datagen.flow(x=train_data, y=train_labels, batch_size=32),
                        epochs=300,
                        steps_per_epoch=len(train_data),
                        validation_data=(validate_data, validate_labels),
                        callbacks=[cp_callback, tbCallBack, es_fit_gen_CallBack])

    test_loss, test_acc = model.evaluate(validate_data, validate_labels, batch_size=32)
    print('Test accuracy:', test_acc)

    # Evaluate
    val_predict_output = model.predict(validate_data)
    val_output = list()
    val_labels = list()
    for output in val_predict_output:
        val_output.append(np.argmax(output))

    for validate_label in validate_labels:
        val_labels.append(np.argmax(validate_label))

    con_mat = tf.confusion_matrix(val_labels, val_output)

    with tf.Session():
        print('Confusion Matrix: \n\n', tf.Tensor.eval(con_mat, feed_dict=None, session=None))

    model.save(MODEL_PATH)

