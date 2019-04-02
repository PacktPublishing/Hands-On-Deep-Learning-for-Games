from music21 import note, instrument, stream
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path

from helper import createPitchVocabularies, loadChorales
from config import note_embedding_dim, note_embedding_dir

from keras.models import Model
from keras.layers import Input, Dense


# create the vocabulary
note_vocab, note_names_vocab, note_vocab_categorical = createPitchVocabularies()

# note to integer and reversal dictinaries used to make categorical data
note_to_int = dict((note, number) for number, note in enumerate(note_vocab))
int_to_note = dict((number, note) for number, note in enumerate(note_vocab))


# define the autoencoder model parts
input_shape = note_vocab_categorical.shape[0]

note_input = Input(shape=(input_shape,))
note_encoded = Dense(note_embedding_dim, kernel_initializer='uniform', activation='relu')(note_input)
note_decoded = Dense(input_shape, kernel_initializer='uniform', activation='sigmoid')(note_encoded)

# define the full autoencoder
note_autoencoder = Model(note_input, note_decoded)

# define the encoder part of the network
note_encoder = Model(note_input, note_encoded)

#define the decoder part of the network
note_encoded_input = Input(shape=(note_embedding_dim,))
note_decoder_layer = note_autoencoder.layers[-1]

note_decoder = Model(note_encoded_input, note_decoder_layer(note_encoded_input))

# compile and print autoencoder summary
note_autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')
note_autoencoder.summary()


# prepare data for the network

# load Bach chorales
print('loading chorales...')
notes = loadChorales()

# map notes to categorical ones
notes_categorical = []
for (chord, dur) in notes:
    for _note in chord:
        notes_categorical.append(note_vocab_categorical[note_to_int[_note]])
        
notes_categorical = np.reshape(notes_categorical, (len(notes_categorical), -1))

# prepare train and test samples
shuffled_notes = notes_categorical
np.random.shuffle(shuffled_notes)

train_index = int(0.8 * len(shuffled_notes))
test_index = int(0.2 * len(shuffled_notes))

notes_train = shuffled_notes[0:train_index]
notes_test = shuffled_notes[train_index:]

# train the autoencoder network
note_autoencoder.fit(x=notes_train, y=notes_train, epochs=50, batch_size=128, shuffle=True, validation_data=(notes_test, notes_test))

# save the model for future use
os.makedirs(note_embedding_dir, exist_ok=True)
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, note_embedding_dir, "full-model.json"), "w") as json_file:
    json_file.write(note_autoencoder.to_json())

with open(os.path.join(dir_path, note_embedding_dir, "encoder-model.json"), "w") as json_file:
    json_file.write(note_encoder.to_json())

with open(os.path.join(dir_path, note_embedding_dir, "decoder-model.json"), "w") as json_file:
    json_file.write(note_decoder.to_json())
  
note_autoencoder.save_weights(os.path.join(dir_path, note_embedding_dir, 'full-weights.h5'))
note_encoder.save_weights(os.path.join(dir_path, note_embedding_dir, 'encoder-weights.h5'))
note_decoder.save_weights(os.path.join(dir_path, note_embedding_dir, 'decoder-weights.h5'))

# fun visualization of generated 2D-embedding
if note_embedding_dim == 2:
    import matplotlib.pyplot as plt
    space = note_encoder.predict(note_vocab_categorical)
    x_axis = [x[0] for x in space]
    y_axis = [x[1] for x in space]

    text = note_names_vocab

    fig, ax = plt.subplots()
    ax.scatter(x_axis, y_axis)

    for i, txt in enumerate(text):
        ax.annotate(txt, (x_axis[i], y_axis[i]))

print('SUCCESS')