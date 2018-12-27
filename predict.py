from tensorflow import keras
import pandas as pd
import numpy as np
import csv



TARGET_MODEL = 'Best/64%_0.9799_VGG-7/cp-005-0.65.h5'

print('Loading ' + TARGET_MODEL + ' model')
model = keras.models.load_model(TARGET_MODEL)
model.summary()
#test_loss, test_acc = model.evaluate(validate_data, validate_labels, batch_size=32)
#print('Test accuracy:', test_acc)

print('Predict test data')
# Read Test CSV
test_csv_set = pd.read_csv('test.csv', names=('index', 'features'))
test_csv_set = test_csv_set[1:]  # drop column name

test_data = []
for data in test_csv_set['features']:
    test_data.append(np.array(data.split(' ')).astype(np.int))

test_data = np.array(test_data)
test_data = test_data.reshape([-1, 48, 48, 1]) / 255.0

predict_output = model.predict(test_data)

test_output = list()
for output in predict_output:
    test_output.append(np.argmax(output))

with open('predict.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'label'])
    i = 1
    for output in test_output:
        writer.writerow([i, output])
        i += 1


