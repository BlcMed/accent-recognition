from sklearn.pipeline import Pipeline
from load_config import load_constants_from_yaml
from split_silence_transformer import SplitSilenceTransformer
from mfcc_transformer import MfccTransformer
from expander_transformer import ExpanderTransformer


def make_pipeline(sampling_rating,
                  threshold_percentage,
                  min_silence_duration,
                  frame_length_energy,
                  hop_length,
                  n_mfcc, 
                  segment_duration,
                  segment_overlap):
    pipeline = Pipeline(
        [
            ("split_silence_transformer", SplitSilenceTransformer(
                variables=['audio', 'labels'],
                sampling_rating=sampling_rating,
                threshold_percentage=threshold_percentage,
                min_silence_duration=min_silence_duration,
                frame_length_energy=frame_length_energy,
                hop_length=hop_length)),
            
            ("mfcc_transformer", MfccTransformer(variables=["audio", "labels"],
                sampling_rating=sampling_rating, 
                n_mfcc=n_mfcc,
                duration=segment_duration,
                overlap=segment_overlap)),
            
            ("expander_transformer", ExpanderTransformer(n_mfcc=n_mfcc, variables_to_repeat=["labels"]))
        ]
    )
    return pipeline