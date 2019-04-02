# configuration file 

# max chorales to be used for statistics, training etc...
maxChorales = 300

# dimension of notes embedding --- not used anymore
note_embedding_dim = 16

# sequence length for generator LSTM
sequence_length = 128

# latent dimension of VAE (used in pitch-generator)
latent_dim = 512

# latent dimensions for pitches and durations (used in note-generator)
latent_dim_p = 512
latent_dim_d = 256

# directory for saving the note embedding network model --- not used anymore
note_embedding_dir = "models/note-embedding"

# directory for saving the generator network model
pitch_generator_dir = 'models/pitch-generator'

# directory for saving the note generator network model
note_generator_dir = 'models/note-generator'

# directory for saving generated music samples
output_dir = 'samples'