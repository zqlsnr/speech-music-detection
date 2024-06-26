import tensorflow as tf
import os
from DataGenerate import DataGenerator
from CustomCallback import MyCustomCallback
from model_new import CRNN_Model
from Datapartition import datapartition

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

log_dir = "./audio-seg-tf2/src/logs"

mel_dir = "./musan/Mel_Files_0425_num_5120_music_0.4_speech_0.4_noise_0.2"

val_dir = "./musan/val_0425_num_5120_music_0.4_speech_0.4_noise_0.2"

test_dir = "./musan/test_0425_num_5120_music_0.4_speech_0.4_noise_0.2"

model_path = "./model_0425_num_5120_music_0.4_speech_0.4_noise_0.2.h5"

m_samples = 2048 * 2

"""
Load the individual numpy arrays into partition
"""

partition = datapartition(mel_dir, val_dir, test_dir, m_train=m_samples)

# Parameters
params = {'dim': (1,),
          'batch_size': 128,
          'n_classes': 2,
          'shuffle': True}

# Generators
training_generator = DataGenerator(partition[0], **params)
validation_generator = DataGenerator(partition[1], **params)
test_generator = DataGenerator(partition[2], **params)

"""
The CRNN developed for audio segmentation.
"""
model = CRNN_Model()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Ensure the directory for the model path is already created.

history = model.fit(training_generator, validation_data=validation_generator, epochs=30,
                    callbacks=[tensorboard_callback, MyCustomCallback(model_path, test_generator, patience=15)])

model.evaluate(validation_generator)
