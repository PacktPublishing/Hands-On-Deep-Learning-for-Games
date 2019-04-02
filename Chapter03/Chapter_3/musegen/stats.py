from music21 import corpus, note, instrument, stream
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path
from collections import Counter

from helper import parseToFlatArray, transpose_to_C_A, loadChorales, loadChoralesWithKeys

# load Bach chorales
print('loading chorales...')
notes, keys = loadChoralesWithKeys()

# calculate statistics 

note_stats = Counter([n[0] for (n, _) in notes])
duration_stats = Counter([dur for (_, dur) in notes])
keys_stats = Counter(keys)

print("Note statistics:")

for key, value in sorted(note_stats.items()):
    k = key
    if(key != 'end' and key != 'rest'):
        k = note.Note(int(key)).pitch.nameWithOctave
    print ("%s: %s" % (k, value))

print("Duration statistics:")                       # the 'end' mark has 0.0 duration!
for key, value in sorted(duration_stats.items()):
    print ("%s: %s" % (key, value))

print("Key Signature statistics:")
for key, value in sorted(keys_stats.items()):
    print ("%s: %s" % (key, value))

# calculate chorale length stats
notes_per_chorale = sum(v for k, v in note_stats.items() if k != 'end') / note_stats['end']
print('notes per chorale: ', notes_per_chorale)

average_chorale_duration = sum(v for k, v in duration_stats.items() if k != 0.0) / duration_stats[0.0]
print('average chorale length: ', average_chorale_duration)

note_named_stats = Counter([note.Note(int(n)).pitch.name for n in note_stats.values() if n != 'end' and n != 'rest'])

# Plots

plt.figure(1)

# plot pitch statistics
plt.subplot(221)

labels, values = zip(*note_named_stats.items())
indexes = np.arange(len(labels))
width = 0.8

plt.bar(indexes, values, width)
plt.xticks(indexes + width / 20, labels)

# plot duration statistics
plt.subplot(222)

labels, values = zip(*duration_stats.items())
indexes = np.arange(len(labels))
width = 0.8

plt.bar(indexes, values, width)
plt.xticks(indexes + width / 20, labels)

# plot key signature statistics
plt.subplot(223)

labels, values = zip(*keys_stats.items())
indexes = np.arange(len(labels))
width = 0.8

plt.bar(indexes, values, width)
plt.xticks(indexes + width / 20, labels)


plt.show()


print('SUCCESS')