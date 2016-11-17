import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
# from keras.utils.visualize_util import plot

INPUT_SHAPE = (17 * 26 * 256,)

DO_TRAINING = True
features_dir = '../data-livdet-2015'
train_dir = '../data-livdet-2015/Training/Digital_Persona'
test_dir = '../data-livdet-2015/Testing/Digital_Persona'
PRE_TRAINED_WEIGHTS_FILE = 'pretrain.h5'

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=INPUT_SHAPE))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
print('Trainable weights', model.trainable_weights)
# plot(model, to_file='model.png')

if DO_TRAINING:
    train_data = np.concatenate((np.load(features_dir + '/train_fake.npy'), np.load(features_dir + '/train_live.npy')))
    train_labels = [1] * 1000 + [0] * 1000

    test_data = np.concatenate((np.load(features_dir + '/test_fake.npy'), np.load(features_dir + '/test_live.npy')))
    test_labels = [1] * 1500 + [0] * 1000

    model.fit(train_data, train_labels,
              batch_size=32,
              nb_epoch=50,
              verbose=2,
              validation_data=(test_data, test_labels))

    model.save_weights(PRE_TRAINED_WEIGHTS_FILE)
else:
    test_data = np.concatenate((np.load(features_dir + '/test_fake.npy'), np.load(features_dir + '/test_live.npy')))
    test_labels = np.array([1] * 1500 + [0] * 1000)

    model.load_weights(PRE_TRAINED_WEIGHTS_FILE)
    predicted = model.predict(test_data)
    predicted = np.array([0 if x < 0.5 else 1 for x in predicted])

    n_ok = np.sum(predicted == test_labels)
    print 'Validation accuracy = {:.2f} ({:d}/{:d})'.format(float(n_ok) / len(predicted), n_ok, len(predicted))
