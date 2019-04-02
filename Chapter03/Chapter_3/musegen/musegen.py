# Currently this script is configured to use the note-generator model.

from config import sequence_length, output_dir, note_generator_dir
from helper import loadChorales, loadModelAndWeights, createPitchSpecificVocabularies, createDurationVocabularySpecific
from music21 import note, instrument, stream, duration
import numpy as np
import os

# disable GPU processing
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# ----------------------------------------------

from keras.utils import to_categorical

# select the epoch to use when loading the weights of the model generator
generator_epoch = 43

# how many notes to generate ('end' marks are created along the way and the result is split into pieces)
number_of_notes = 200

# load chorales to create the vocabularies
print('loading chorales...')
notes = loadChorales()

# create the vocabulary
note_vocab, note_names_vocab, note_vocab_categorical = createPitchSpecificVocabularies([x[0] for (x, _) in notes])
duration_vocab = createDurationVocabularySpecific([d for (_, d) in notes])
duration_vocab_categorical = to_categorical(range(len(duration_vocab)))

note_to_int = dict((note, number) for number, note in enumerate(note_vocab))
int_to_note = dict((number, note) for number, note in enumerate(note_vocab))

duration_to_int = dict((dur, number) for number, dur in enumerate(duration_vocab))

duration_dim = duration_vocab.shape[0]
pitch_dim = np.array(note_vocab).shape[0]

print('loading networks...')
dir_path = os.path.dirname(os.path.realpath(__file__))
generator = loadModelAndWeights(os.path.join(dir_path, note_generator_dir, 'model.json'),
                               os.path.join(dir_path, note_generator_dir, 'weights-{:02d}.hdf5'.format(generator_epoch)))

# make a melody!!!
pitch_input = np.eye(pitch_dim)[np.random.choice(pitch_dim, size=sequence_length)]
duration_input = np.eye(duration_dim)[np.random.choice(duration_dim, size=sequence_length)]

print('generating output...')

# generate notes
generator_output = []

for _ in range(number_of_notes):
    # reshape inputs
    pi = np.reshape(pitch_input, (1, sequence_length, pitch_dim))
    di = np.reshape(duration_input, (1, sequence_length, duration_dim))

    # make prediction
    pitch_pred, dur_pred = generator.predict({'pitches_input': pi, 'durations_input': di}, verbose=0)

    generator_output.append((pitch_pred, dur_pred))

    pitch_input = np.vstack([pitch_input, pitch_pred])
    pitch_input = pitch_input[1:len(pitch_input)]

    duration_input = np.vstack([duration_input, dur_pred])
    duration_input = duration_input[1:len(duration_input)]


output_notes = [(int_to_note[np.argmax(n)], duration_vocab[np.argmax(d)]) for (n, d) in generator_output]
output_notes = np.array(output_notes)
output_notes = np.reshape(output_notes, (-1, 2))

# output_notes contains: pitch values in midi format (integers), 'rest' marks, 'end' marks

# split the generated notes into pieces based on 'end' marks
indices = []
for (ind, (n, _)) in enumerate(output_notes):
    if n == 'end':
        indices.append(ind)
indices = np.insert(np.reshape(indices, (-1)), 0, 0)
    
pieces = [output_notes]
if len(indices) > 1:
    pieces = ([ output_notes[(indices[j] + 1):indices[j + 1] ] for j in range(len(indices) - 1)])

print('writing output to disk...')

os.makedirs(os.path.join(dir_path, output_dir, 'note-generator'), exist_ok=True)

# output pieces to midi files
for index, notes in enumerate(pieces):
    midi_notes = []
    offset = 0
    for n, d in notes:
        # since a duration of 0 is included in the vocabulary (for the 'end' marks), the network may generate a 0 duration for other notes
        # naively correct and report this erroneous behaviour
        if abs(float(d)) < 0.001:
            print('found zero duration')
            d = '1.0'
        if n == 'rest':
            new_note = note.Rest()
            new_note.duration = duration.Duration(float(d))
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            midi_notes.append(new_note)
        else:
            new_note = note.Note(int(n))
            new_note.duration = duration.Duration(float(d))
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            midi_notes.append(new_note)
        offset += float(d)
        
    midi_stream = stream.Stream(midi_notes)
    midi_stream.write('midi', fp=os.path.join(dir_path, output_dir, 'note-generator', 'sample-{}.mid'.format(index)))