import os
from keras.preprocessing.image import ImageDataGenerator

train_dir = '../data-livdet-2015/Training/Digital_Persona'
target_dir = '../data-livdet-2015/Training_augmented/Digital_Persona'
TARGET_SIZE = (252, 324)
N_BATCHES = 100
BATCH_SIZE = 32

data_generator = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')
for k in ['Fake', 'Live']:
    if not os.path.exists(target_dir + '/' + k):
        os.mkdir(target_dir + '/' + k)

    generator = data_generator.flow_from_directory(
            train_dir,
            target_size=TARGET_SIZE,
            classes=[k],
            batch_size=BATCH_SIZE,
            save_to_dir=target_dir + '/' + k,
            save_format='png')

    i = 0
    for batch in generator:
        i += 1
        if i > N_BATCHES:
            break
