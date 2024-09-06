from load_config import load_constants_from_yaml
from sklearn.pipeline import Pipeline
from custom_transformers.split_silence_transformer import SplitSilenceTransformer
from custom_transformers.mfcc_transformer import MfccTransformer
from custom_transformers.expander_transformer import ExpanderTransformer
from custom_transformers.label_encoder_transformer import LabelEncoderTransformer
from custom_transformers.standard_scaler import StandardScalerTransformer


constants = load_constants_from_yaml('constants.yml')

SAMPLING_RATING = constants["SAMPLING_RATING"]
FRAME_LENGTH_ENERGY = constants["FRAME_LENGTH_ENERGY"]
THRESHOLD_PERCENTAGE = constants["THRESHOLD_PERCENTAGE"]
MIN_SILENCE_DURATION = constants["MIN_SILENCE_DURATION"]
HOP_LENGTH = constants["HOP_LENGTH"]
SEGMENT_DURATION = constants["SEGMENT_DURATION"]
SEGMENT_OVERLAP = constants["SEGMENT_OVERLAP"]
N_MFCC = constants["N_MFCC"]
CONSIDERED_ACCENTS = constants["CONSIDERED_ACCENTS"]

pipeline = Pipeline(
    [
        ("split_silence_transformer", SplitSilenceTransformer(
            variables=['audio', 'label'],
            sampling_rating=SAMPLING_RATING,
            threshold_percentage=THRESHOLD_PERCENTAGE,
            min_silence_duration=MIN_SILENCE_DURATION,
            frame_length_energy=FRAME_LENGTH_ENERGY,
            hop_length=HOP_LENGTH)),
        
        ("mfcc_transformer", MfccTransformer(variables=["audio", "label"],
            sampling_rating=SAMPLING_RATING, 
            n_mfcc=N_MFCC,
            duration=SEGMENT_DURATION,
            overlap=SEGMENT_OVERLAP)),
        
        ("expander_transformer", ExpanderTransformer(n_mfcc=N_MFCC, columns_to_remain=["label"])),

        ("label_encoder", LabelEncoderTransformer(variable='label')),
        
        ("scaler", StandardScalerTransformer(n_mfccs=N_MFCC))
    ]
)
