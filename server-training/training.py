import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model


from utils.Generator import DataGeneratorIA
from Models import c3d_super_lite, mobilenetonly, lrcn

# Load Training Labels
df_jest = pd.read_csv( 'jester-train-six-actions.csv',
					index_col = None,
					header=None,
					sep=';',
					names=['Folder','Action', 'Frames'])

df_jest = df_jest[(df_jest['Frames']>=30)]
df_jest = df_jest.drop(columns='Frames')
df_jest.loc[:,'Path'] = '/home/samygarg/hanuman/20bn-jester-v1'


# Training Split: 80% Training , 20% Validation
dftrain = df_jest.head(int(len(df_jest)*0.8)).copy()
dfval = df_jest.tail(int(len(df_jest)*0.2)).copy()

# Print Categories
print('Categories : ')
print(sorted((dftrain['Action'].unique())))

# Convert Labels to One-Hot
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(dftrain['Action'])
dftrain['Action'] = integer_encoded

# Convert Labels to One-Hot
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(dfval['Action'])
dfval['Action'] = integer_encoded

# Load Model
model = lrcn()

model_file_format_best = os.path.join('radhakrishna/model-lrcn.best.hdf5') 

# Fit Model
model.fit_generator(
 	DataGeneratorIA(dftrain,dim=(112,112),augment=True),
 	validation_data=DataGeneratorIA(dfval,dim=(112,112),augment=True),
 	verbose=1,
 	epochs=50,
 	callbacks=[ModelCheckpoint(model_file_format_best,
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                mode='min',
                                period=1)]
 )