# Train a model to generate pitches

from music21 import *
import numpy as np
import os
import os.path

from contextlib import redirect_stdout

from helper import loadChorales, createPitchVocabularies, loadModelAndWeights
from config import sequence_length, latent_dim, pitch_generator_dir

# disable GPU processing as the network doesn't fit in my card's memory
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# ----------------------------------------------

from keras.utils import to_categorical
from keras import backend as K
from keras.losses import mse, binary_crossentropy
from keras.models import model_from_json
from keras.optimizers import RMSprop
from keras import objectives
from keras import initializers
from keras.callbacks import ModelCheckpoint

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Dropout, Lambda
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, TensorBoard, CSVLogger

# gaussian sampling
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# VAE network loss (reconstruction + KL-divergence)
def vae_loss(y_true, y_pred):
    xent_loss = objectives.categorical_crossentropy(y_true, y_pred)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
    loss = xent_loss + kl_loss
    return loss  

# create the vocabulary
note_vocab, note_names_vocab, note_vocab_categorical = createPitchVocabularies()

# size of the categorical representation of a note (since we do not use an embedding this can change, if we change the vocabulary)
note_categorical_size = note_vocab_categorical.shape[0]

# note to integer and reversal dictinaries used to make categorical data
note_to_int = dict((note, number) for number, note in enumerate(note_vocab))
int_to_note = dict((number, note) for number, note in enumerate(note_vocab))

# load Bach chorales
print('loading chorales...')
notes = loadChorales()
only_notes = [chord[0] for (chord, _) in notes]         # discard durations

# preapre data for the network

network_input = []
network_output = []

if len(notes) <= sequence_length:
    raise Exception("Notes from pieces are less than the given sequence length")

# prepare series of sequences and for each one the target output
for i in range(0, len(notes) - sequence_length):
    sequence_in = only_notes[i:i + sequence_length]
    sequence_out = only_notes[i + sequence_length]
    
    categorical_in = np.reshape([note_vocab_categorical[note_to_int[x]] for x in sequence_in], (sequence_length, -1))
    categorical_out = np.reshape([note_vocab_categorical[note_to_int[sequence_out]]], (1, -1))
    
    # append the categorical representation of each note
    network_input.append(categorical_in)
    network_output.append([categorical_out])
    
network_input = np.array(network_input)
network_output = np.reshape(np.array(network_output), (-1,note_categorical_size))

# split data to train and test
train_index = int(0.85 * len(network_input))

network_input_train = network_input[:train_index]
network_input_test = network_input[train_index:]

network_output_train = network_output[:train_index]
network_output_test = network_output[train_index:]

# define the music generator network

# at first we have a series of LSTM layers for progression of notes
x = Input(shape=(sequence_length, note_categorical_size,), name='generator_input')
h = LSTM(256, return_sequences=True, name='h_lstm_1')(x)
h = Dropout(0.2)(h)
h = LSTM(512, return_sequences=True, name='h_lstm_2')(h)
h = Dropout(0.1)(h)
h = LSTM(512, return_sequences=True, name='h_lstm_3')(h)
h = LSTM(256, return_sequences=False, name='h_lstm_4')(h)

# then use a VAE for non-deterministic generation of notes
z_mean = Dense(latent_dim, name='z_mean', kernel_initializer='uniform')(h)
z_log_var = Dense(latent_dim, name='z_log_var', kernel_initializer='uniform')(h)

z = Lambda(sampling, output_shape=(latent_dim, ), name='z')([z_mean, z_log_var])
decoded_z = Dense(note_categorical_size, activation="softmax", name='generator_output', kernel_initializer='uniform')(z)

# end-to-end generator
generator = Model(x, decoded_z)

# compile and print generator summary
optimizer = RMSprop(lr=0.001)
generator.compile(optimizer=optimizer, loss=vae_loss)
generator.summary()

# save the model
os.makedirs(pitch_generator_dir, exist_ok=True)
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, pitch_generator_dir, "model.json"), "w") as json_file:
    json_file.write(generator.to_json())

# save the architecture as text
with open(os.path.join(dir_path, pitch_generator_dir, "arch.txt"), "w") as f:
    with redirect_stdout(f):
        generator.summary()

# wegihts will be saved every epoch
filepath = os.path.join(dir_path, pitch_generator_dir, "weights-{epoch:02d}.h5")   

checkpoint = ModelCheckpoint(
    filepath, monitor='val_loss', 
    verbose=0,        
    save_best_only=False,        
    mode='min'
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
csv_logger = CSVLogger(os.path.join(dir_path, pitch_generator_dir, "trainning.csv"))

callbacks_list = [checkpoint, reduce_lr, csv_logger]

print('training...')
# train the generator network
generator.fit(x=network_input_train, y=network_output_train, shuffle=False, initial_epoch=0, epochs=200, batch_size=16, callbacks=callbacks_list, validation_data=(network_input_test, network_output_test))

print('SUCCESS')