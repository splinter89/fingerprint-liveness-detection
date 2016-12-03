""" Code for fine-tuning Inception V3 for a new task """

import glob
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Flatten, Dropout, AveragePooling2D
# from keras.utils.visualize_util import plot
import inception_v3 as inception

IMG_SIZE = (299, 299)

train_dir = '../data-livdet-2015/Training/Digital_Persona'
test_dir = '../data-livdet-2015/Testing/Digital_Persona'
PRE_TRAINED_WEIGHTS_FILE = 'fingerprints_pretrain.h5'
DO_TRAINING = True

base_model = inception.InceptionV3(include_top=True)

# x = base_model.get_layer('mixed10').output
# x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
# x = Flatten(name='flatten')(x)
x = base_model.get_layer(name='flatten').output
x = Dense(32, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid', name='predictions')(x)

model = Model(input=base_model.input, output=predictions)
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

print('Loaded Inception model')
model.summary()
print('Trainable weights', model.trainable_weights)
# plot(model, to_file='fingerprints_model.png')

if DO_TRAINING:
    # Data generators for feeding training/testing images to the model
    train_data_generator = ImageDataGenerator(rescale=1. / 255)
    # All images will be resized to 299x299 Inception V3 input
    train_generator = train_data_generator.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='binary')
    test_data_generator = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_data_generator.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        samples_per_epoch=250,
        nb_epoch=1,     # TODO
        validation_data=test_generator,
        verbose=2,
        nb_val_samples=200)

    model.save_weights(PRE_TRAINED_WEIGHTS_FILE)
else:
    model.load_weights(PRE_TRAINED_WEIGHTS_FILE)

img_paths = glob.glob('../data-livdet-2015/Testing/Digital_Persona/*/*.png')
predicted = {}
for img_path in img_paths:
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = inception.preprocess_input(x)
    predicted[img_path] = model.predict(x)

    if len(predicted) % 200 == 0:
        print len(predicted)

    print img_path
    print predicted
    quit(0)
