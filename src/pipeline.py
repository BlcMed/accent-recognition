from sklearn.pipeline import Pipeline
from split_silence_transformer import SplitSilenceTransformer

audio,
sr,
threshold_percentage,
min_silence_duration,
frame_length_energy,
hop_length,

pipeline = Pipeline(
    ("split_silence_transformer", SplitSilenceTransformer())    
)