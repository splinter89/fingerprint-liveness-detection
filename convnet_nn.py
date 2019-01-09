import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, merge
from sklearn import preprocessing
# from keras.utils.visualize_util import plot

INPUT_SHAPE = (17 * 26 * 256,)
AUX_INPUT_SHAPE = (4096,)

DO_TRAINING = True
features_dir = '../data-livdet-2015/z_features_252x324'
aux_features_dir = '../data-livdet-2015/z_other_features/BSIF-DigPer-2015-features'
PRE_TRAINED_WEIGHTS_FILE = 'pretrain.h5'

# model = Sequential()
# model.add(Dense(32, activation='relu', input_shape=INPUT_SHAPE))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

main_input = Input(shape=INPUT_SHAPE)
# aux_input = Input(shape=AUX_INPUT_SHAPE)
# x = merge([main_input, aux_input], mode='concat')
x = main_input
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
# model = Model(input=[main_input, aux_input], output=predictions)
model = Model(input=main_input, output=predictions)
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
print('Trainable weights', model.trainable_weights)
# plot(model, to_file='model.png')

test_fake = np.load(features_dir + '/test_fake.npy')
test_live = np.load(features_dir + '/test_live.npy')
test_data = np.concatenate((test_fake, test_live))
test_labels = np.array([1] * len(test_fake) + [0] * len(test_live))
del test_fake, test_live
preprocessing.scale(test_data, copy=False)

# aux_test_data = np.concatenate((
#     np.loadtxt(aux_features_dir + '/Data_2015_BSIF_7_12_motion_Test_Spoof_DigPerson.txt'),
#     np.loadtxt(aux_features_dir + '/Data_2015_BSIF_7_12_motion_Test_Real_DigPerson.txt')))
# preprocessing.scale(aux_test_data, copy=False)

if DO_TRAINING:
    train_fake = np.load(features_dir + '/train_fake.npy')
    train_live = np.load(features_dir + '/train_live.npy')
    train_data = np.concatenate((train_fake, train_live))
    train_labels = [1] * len(train_fake) + [0] * len(train_live)
    del train_fake, train_live
    preprocessing.scale(train_data, copy=False)

    # aux_train_data = np.concatenate((
    #     np.loadtxt(aux_features_dir + '/Data_2015_BSIF_7_12_motion_Train_Spoof_DigPerson.txt'),
    #     np.loadtxt(aux_features_dir + '/Data_2015_BSIF_7_12_motion_Train_Real_DigPerson.txt')))
    # preprocessing.scale(aux_train_data, copy=False)

    # model.fit([train_data, aux_train_data], train_labels,
    model.fit(train_data, train_labels,
              batch_size=32,
              nb_epoch=50,
              verbose=2,
              # validation_data=([test_data, aux_test_data], test_labels))
              validation_data=(test_data, test_labels))

    model.save_weights(PRE_TRAINED_WEIGHTS_FILE)
else:
    model.load_weights(PRE_TRAINED_WEIGHTS_FILE)

# predicted = model.predict([test_data, aux_test_data])
predicted = model.predict(test_data)
predicted = np.array([0 if x < 0.5 else 1 for x in predicted])

n_ok = np.sum(predicted == test_labels)
print 'Validation accuracy = {:.2f} ({:d}/{:d})'.format(float(n_ok) / len(predicted), n_ok, len(predicted))
