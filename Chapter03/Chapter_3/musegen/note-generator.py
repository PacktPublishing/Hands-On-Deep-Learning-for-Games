# Extension of the pitch-generator model
# Generates notes (pitches and durations)

from music21 import *
import numpy as np
import os
import os.path

from contextlib import redirect_stdout

from helper import loadChorales, createPitchSpecificVocabularies, loadModelAndWeights, createDurationVocabularySpecific
from config import sequence_length, latent_dim_p, latent_dim_d, note_generator_dir

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
from keras.layers import Input, Dense, LSTM, Dropout, Lambda, TimeDistributed, Concatenate
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, TensorBoard, CSVLogger

# Gaussian sampling for sequences
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    seq = K.int_shape(z_mean)[1]
    dim = K.int_shape(z_mean)[2]
    
    epsilon = K.random_normal(shape=(batch, seq, dim))
        
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# pitches VAE loss
def vae_p_loss(y_true, y_pred):
    xent_loss = objectives.categorical_crossentropy(y_true, y_pred)
    kl_loss = - 0.5 * K.mean(1 + z_log_var_p - K.square(z_mean_p) - K.exp(z_log_var_p))
    loss = xent_loss + kl_loss
    return loss

# durations VAE loss
def vae_d_loss(y_true, y_pred):
    xent_loss = objectives.categorical_crossentropy(y_true, y_pred)
    kl_loss = - 0.5 * K.mean(1 + z_log_var_d - K.square(z_mean_d) - K.exp(z_log_var_d))
    loss = xent_loss + kl_loss
    return loss

# load Bach chorales
print('loading chorales...')
notes = loadChorales()

# create the vocabularies for pitches and durations
note_vocab, note_name_vocab, note_vocab_categorical = createPitchSpecificVocabularies([x[0] for (x, _) in notes])
duration_vocab = createDurationVocabularySpecific([d for (_, d) in notes])
duration_vocab_categorical = to_categorical(range(len(duration_vocab)))

note_to_int = dict((note, number) for number, note in enumerate(note_vocab))
int_to_note = dict((number, note) for number, note in enumerate(note_vocab))

duration_to_int = dict((dur, number) for number, dur in enumerate(duration_vocab))

duration_dim = duration_vocab.shape[0]
pitch_dim = np.array(note_vocab).shape[0]

# preapre data for the network

if len(notes) <= sequence_length:
    raise Exception("Notes from pieces are less than the given sequence length")

durations_input = []
durations_output = []
pitches_input = []
pitches_output = []

# prepare pitches data

only_notes = [x[0] for (x, _) in notes]

for i in range(0, len(notes) - sequence_length):
    sequence_in = only_notes[i:i + sequence_length]
    sequence_out = only_notes[i + sequence_length]
    
    categorical_in = np.reshape([note_vocab_categorical[note_to_int[x]] for x in sequence_in], (sequence_length, -1))
    categorical_out = np.reshape([note_vocab_categorical[note_to_int[sequence_out]]], (1, -1))
    
    pitches_input.append(categorical_in)
    pitches_output.append([categorical_out])
    
    
pitches_input = np.array(pitches_input)
pitches_output = np.reshape(np.array(pitches_output), (-1, pitch_dim))


# prepare durations data

only_durations = [x for (_, x) in notes]

for i in range(0, len(notes) - sequence_length):
    sequence_in = only_durations[i:i + sequence_length]
    sequence_out = only_durations[i + sequence_length]
    
    categorical_in = np.reshape([duration_vocab_categorical[[duration_to_int[x]]] for x in sequence_in], (sequence_length, -1))
    categorical_out = np.reshape([duration_vocab_categorical[duration_to_int[sequence_out]]], (1, -1))
    
    durations_input.append(categorical_in)
    durations_output.append([categorical_out])
    
durations_input = np.array(durations_input)
durations_input = np.reshape(durations_input, (-1, sequence_length, duration_dim))

durations_output = np.reshape(np.array(durations_output), (-1, duration_dim))

# split data to train and test
train_index = int(0.80 * len(pitches_input))

pitches_input_train = pitches_input[:train_index]
pitches_input_test = pitches_input[train_index:]

pitches_output_train = pitches_output[:train_index]
pitches_output_test = pitches_output[train_index:]

durations_input_train = durations_input[:train_index]
durations_input_test = durations_input[train_index:]

durations_output_train = durations_output[:train_index]
durations_output_test = durations_output[train_index:]

# define the note generator network

# layers for pitches generation (independent of durations) 
x_p = Input(shape=(sequence_length, pitch_dim,), name='pitches_input')
h = LSTM(256, return_sequences=True, name='h_lstm_p_1')(x_p)
h = LSTM(512, return_sequences=True, name='h_lstm_p_2')(h)
h = LSTM(256, return_sequences=True, name='h_lstm_p_3')(h)

# VAE for pitches
z_mean_p = TimeDistributed(Dense(latent_dim_p, kernel_initializer='uniform'))(h)
z_log_var_p = TimeDistributed(Dense(latent_dim_p, kernel_initializer='uniform'))(h)
z_p = Lambda(sampling)([z_mean_p, z_log_var_p])
z_p = TimeDistributed(Dense(pitch_dim, kernel_initializer='uniform', activation='softmax'))(z_p)

# layers for durations generation (independent of pitches) 
x_d = Input(shape=(sequence_length, duration_dim, ), name='durations_input')
h = LSTM(128, return_sequences=True)(x_d)
h = LSTM(256, return_sequences=True)(h)
h = LSTM(128, return_sequences=True)(h)

# VAE for durations
z_mean_d = TimeDistributed(Dense(latent_dim_d, kernel_initializer='uniform'))(h)
z_log_var_d = TimeDistributed(Dense(latent_dim_d, kernel_initializer='uniform'))(h)
z_d = Lambda(sampling)([z_mean_d, z_log_var_d])
z_d = TimeDistributed(Dense(duration_dim, kernel_initializer='uniform', activation='softmax'))(z_d)

# Concatenate layer to correlate and change if necessary the two generated components
conc = Concatenate(axis=-1)([z_p, z_d])
latent = TimeDistributed(Dense(pitch_dim + duration_dim, kernel_initializer='uniform'))(conc)
latent = LSTM(256, return_sequences=False)(latent)

# final output layers for the two components
o_p = Dense(pitch_dim, activation='softmax', name='pitches_output', kernel_initializer='uniform')(latent)
o_d = Dense(duration_dim, activation='softmax', name='durations_output', kernel_initializer='uniform')(latent)

#               layer summary

#       Pitches LSTM    Durations LSTM
#             |              |
#       Pitches VAE     Durations VAE
#             \              /
#            Dense + Shared LSTM
#             /              \
#       Pitches out     Durations out


# end-to-end model
note_generator = Model(inputs=[x_p, x_d], outputs=[o_p, o_d])

# compile and print generator summary
losses = {'durations_output': vae_d_loss, 'pitches_output': vae_p_loss}
note_generator.compile(optimizer='rmsprop', loss=losses)
note_generator.summary()

# save the model
os.makedirs(note_generator_dir, exist_ok=True)
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, note_generator_dir, "model.json"), "w") as json_file:
    json_file.write(note_generator.to_json())

# save the architecture as text
with open(os.path.join(dir_path, note_generator_dir, "arch.txt"), "w") as f:
    with redirect_stdout(f):
        note_generator.summary()

# train the network
inputs = {'pitches_input': pitches_input_train, 'durations_input': durations_input_train}
outputs = {'pitches_output': pitches_output_train, 'durations_output': durations_output_train}

inputs_v = {'pitches_input': pitches_input_test, 'durations_input': durations_input_test}
output_v = {'pitches_output': pitches_output_test, 'durations_output': durations_output_test}

filepath = os.path.join(dir_path, note_generator_dir, "weights-{epoch:02d}.h5")

checkpoint = ModelCheckpoint(
    filepath, monitor='val_loss', 
    verbose=0,        
    save_best_only=False,        
    mode='min'
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
csv_logger = CSVLogger(os.path.join(dir_path, note_generator_dir, "trainning.csv"))

callbacks_list = [checkpoint, reduce_lr, csv_logger]

note_generator.fit(x=inputs, y=outputs, validation_data=(inputs_v, output_v), shuffle=True, initial_epoch=0, epochs=200, batch_size=32, callbacks=callbacks_list)