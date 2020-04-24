import csv
import random
import os
import numpy as np
from tqdm.auto import tqdm
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv2D, MaxPooling2D, Flatten, Reshape, Lambda, TimeDistributed, Masking
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint

# Seed random
random.seed(13243252)

# Data max length
IMG_SIZE = (100, 132)
IMG_DIMS = (100, 132, 3)
MAX_LEN = 40

# Based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(keras.utils.Sequence):
  'Generates data for Keras'
  def __init__(self, list_IDs, labels, filepaths, preload=False, shuffle=True):
    'Initialization'
    self.labels = labels
    self.filepaths = filepaths
    self.list_IDs = list_IDs
    self.shuffle = shuffle

    self.preload = preload
    self.preloaded = np.empty(len(self.filepaths), dtype=object)
    if self.preload:
      for ID in tqdm(list_IDs):
        X = self.load_clip(ID)
        self.preloaded[ID] = X
  
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return len(self.list_IDs)

  def __getitem__(self, index):
    'Generate one batch of data'
    # Generate index
    ID = self.indexes[index]

    # Generate data
    X = []
    if self.preload:
      X = self.preloaded[ID]
    else:
      X = self.load_clip(ID)
    y = self.labels[ID]
  
    # extended_X = self.extend_clip(X)
  
    return np.array([X]), np.array([y])

  def load_clip(self, ID):
    X = []
    for (path, dirs, files) in os.walk(self.filepaths[ID]):
      for frame in files:
        frame_processed = resize(img_to_array(load_img(path + "/" + frame)), IMG_DIMS).astype("float16")
        X.append(frame_processed)
    
    return np.array(X)

  def extend_clip(self, clip):
    clip_shape = clip.shape
    extended_len = MAX_LEN
    if clip_shape[0] >= extended_len:
      return clip[:extended_len]
    else:
      # first zeros, then the clip + masking in the network
      return np.concatenate([np.tile(np.zeros(IMG_DIMS), (MAX_LEN - clip_shape[0],1,1,1)), clip])

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.copy(self.list_IDs)
    if self.shuffle:
      np.random.shuffle(self.indexes)

def createDataGenerator(csv_name):
    labels = []
    filepaths = []
    with open(csv_name, "rt") as csv_file:
      reader = csv.reader(csv_file, delimiter=';')
      for row in reader:
        if len(row) < 2:
            continue
        labels.append(row[1])
        filepaths.append("/home/samygarg/hanuman/20bn-jester-v1/" + str(row[0]))
    ids = list(range(len(labels)))
    labels = LabelBinarizer().fit_transform(labels)
    return DataGenerator(ids, labels, filepaths)

train_generator = createDataGenerator("/home/samygarg/hanuman/20bn-jester-v1/annotations/jester-v1-train.csv")
test_generator = createDataGenerator("/home/samygarg/hanuman/20bn-jester-v1/annotations/jester-v1-validation.csv")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
NUM_CLASSES = 27

preprocess = Sequential()
preprocess.add(Masking(mask_value=0.))
preprocess.add(Conv2D(64, kernel_size=(4, 4), strides=(1, 1), activation='relu'))
preprocess.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
preprocess.add(Dropout(0.5))
preprocess.add(Conv2D(64, kernel_size=(4, 4), strides=(1, 1), activation='relu'))
preprocess.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
preprocess.add(Dropout(0.5))
preprocess.add(Conv2D(64, kernel_size=(4, 4), strides=(1, 1), activation='relu'))
preprocess.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
preprocess.add(Dropout(0.5))
preprocess.add(Conv2D(64, kernel_size=(4, 4), strides=(1, 1), activation='relu'))
preprocess.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
preprocess.add(Dropout(0.5))
preprocess.add(Flatten())
preprocess.add(Dense(256))

recurrent = Sequential()
recurrent.add(LSTM(512))
recurrent.add(Dropout(0.25))
recurrent.add(Dense(256))
recurrent.add(Dropout(0.25))
recurrent.add(Dense(NUM_CLASSES))

model = Sequential()
model.add(TimeDistributed(preprocess, input_shape=(MAX_LEN, IMG_DIMS[0], IMG_DIMS[1], IMG_DIMS[2])))
model.add(recurrent)
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=1e-4, rho=0.9), metrics=["acc"])

filepath="/home/samygarg/hanuman/model/weights-improvement-{epoch:03d}-{val_accuracy:.6f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy',verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit_generator(train_generator, initial_epoch=0, epochs=10, validation_data=test_generator, callbacks=callbacks_list)