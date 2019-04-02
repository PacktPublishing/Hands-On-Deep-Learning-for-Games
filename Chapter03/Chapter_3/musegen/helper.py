from keras.utils import to_categorical
from keras.models import model_from_json
from music21 import corpus, chord, note, pitch, interval
from config import maxChorales
import os.path
import numpy as np

def getChoralesIterator():
    iterator = corpus.chorales.Iterator()
    if maxChorales > iterator.highestNumber:
        raise Exception('Not that many chorales')

    return iterator

# loads chorales into flat array (see parseToFlatArray for definition)
def loadChorales():
    notes = []
    iterator = getChoralesIterator()

    # load notes of chorales
    for chorale in iterator[1:maxChorales]:             # iterator is 1-based                        
        transpose_to_C_A(chorale.parts[0])
        notes = notes + parseToFlatArray(chorale.parts[0])
        notes.append((['end'], 0.0))                        # mark the end of the piece
    
    return notes

# loads chorales as above along with the key signature of each one
def loadChoralesWithKeys():
    notes = []
    iterator = getChoralesIterator()
    orig_keys = []

    # load notes of chorales
    for chorale in iterator[1:maxChorales]:             # iterator is 1-based   
        k = chorale.parts[0].analyze('key')    
        orig_keys.append(k.tonicPitchNameWithCase)                 
        transpose_to_C_A(chorale.parts[0])
        notes = notes + parseToFlatArray(chorale.parts[0])
        notes.append((['end'], 0.0))                        # mark the end of the piece
    
    return notes, orig_keys

# if the given chord is a note (singleton chord) then returns a list of itself, otherwise concatenates all notes in a list
def chordToNotes(notes):
    if isinstance(notes, chord.Chord):
        notes.duration.quarterLength
        return [str(x.midi) for x in notes.pitches]
    elif isinstance(notes, note.Note):
        return [str(notes.pitch.midi)]                      # use midi naming to avoid problems with accidentals (same note different name due to accidental)
    else: # this is a rest
        return ['rest']
    
# transform a midi score to flat array (=> array of notes and their durations) 
def parseToFlatArray(score):
    notes = []
    for _note in score.flat.notesAndRests:
        # when the note is tied, add the duration to the last note instead of creating a new one
        if (_note.tie is not None) and (_note.tie.type == 'continue' or _note.tie.type == 'stop'):
            (n, dur) = notes[-1]
            notes[-1] = (n, dur + _note.duration.quarterLength)
        else:
            notes.append((chordToNotes(_note), _note.duration.quarterLength))
    return notes


# transposes a stream object (part) from its given basePitch to the targetPitch (transform happens in place)
def transpose_to(stream, basePitch, targetPitch):
    i = interval.Interval(pitch.Pitch(basePitch), pitch.Pitch(targetPitch))
    return stream.transpose(i, inPlace=True)

# create notes vocabulary from A2 to A6 with pitch and midi names
def createPitchVocabulary():
    n = note.Note('A2')
    note_vocab = []
    note_names_vocab = []

    while n.pitch.nameWithOctave != "B6":
        note_vocab.append(str(n.pitch.midi))
        note_names_vocab.append(n.pitch.nameWithOctave)
        n.transpose(1, inPlace=True)
    
    # append the special marks for rest and end of piece
    note_vocab.append('rest')
    note_vocab.append('end')

    return note_vocab, note_names_vocab

# transforms a score to C major when the score key is major and to A minor when the score key is minor
def transpose_to_C_A(score):
    k = score.analyze('key')
    if k.mode == 'major':
        transpose_to(score, k.tonic.name, 'C')
    elif k.mode == 'minor':
        transpose_to(score, k.tonic.name, 'A')

# creates note vocabularies and categorical vocabularies
def createPitchVocabularies():
    note_vocab, note_names_vocab = createPitchVocabulary()
    note_vocab_categorical = to_categorical(range(len(note_vocab)))
    return note_vocab, note_names_vocab, note_vocab_categorical

def createPitchSpecificVocabularies(pitches):
    note_vocab, note_names_vocab = createPitchVocabularySpecific(pitches)
    note_vocab_categorical = to_categorical(range(len(note_vocab)))
    return note_vocab, note_names_vocab, note_vocab_categorical

def createPitchVocabularySpecific(pitches):
    distinct = np.unique(pitches)  
    note_vocab = []
    note_names_vocab = []
    
    for n in distinct:
        if n != 'rest' and n != '_' and n != 'end':            
            note_vocab.append(n)
        else:
            note_vocab.append(n)
            note_names_vocab.append(n)
            
    return note_vocab, note_names_vocab

# create a vocabulary from the given durations
def createDurationVocabularySpecific(durations):
    duration_vocab = np.unique(durations)
    return duration_vocab

# load a saved model and its weights
def loadModelAndWeights(model_file, weights_file):    
    if os.path.exists(model_file) == False:
        raise Exception("model file not found")

    if os.path.exists(weights_file) == False:
        raise Exception("weights file not found")

    _file = open(model_file, 'r')
    json = _file.read()
    _file.close()
    model = model_from_json(json)
    model.load_weights(weights_file)

    return model